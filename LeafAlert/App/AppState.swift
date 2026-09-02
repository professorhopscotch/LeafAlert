import SwiftUI
import Combine
import CoreImage

/// Top-level observable state shared across the app via EnvironmentObject.
@MainActor
final class AppState: ObservableObject {

    // MARK: - Engines

    let captureEngine = CaptureEngine()
    let inferenceEngine = InferenceEngine()
    let alertEngine = AlertEngine()

    // MARK: - Stores

    let detectionLogStore = DetectionLogStore()

    // MARK: - Published State

    @Published var isPatrolling = false
    /// The detection the feedback card shows. Lives `detectionCardLifetime`
    /// (or until the user answers/dismisses it) — long enough to act on.
    @Published var lastDetection: DetectionResult? {
        didSet { if lastDetection == nil { boxDetection = nil } }
    }
    /// The detection whose bounding box is drawn. Expires after
    /// `detectionBoxLifetime`, independently of the card: the box is a spatial
    /// claim that goes stale in seconds; the card is not.
    @Published private(set) var boxDetection: DetectionResult?
    @AppStorage("hasShownDisclaimer") var hasShownDisclaimer = false

    // MARK: - Settings (synced from UserDefaults via AppStorage)

    // Default 0.50 (neutral). Held-out eval showed the old 0.65 caught only ~44%
    // of toxic plants; 0.50 with per-class thresholds + an "uncertain" band raises
    // that materially. See ToxicityThresholds.
    @AppStorage("sensitivityThreshold") var sensitivityThreshold: Double = 0.50
    @AppStorage("audioAlertsEnabled") var audioAlertsEnabled = true
    @AppStorage("screenDimLevel") var screenDimLevel: Double = 0.7
    @AppStorage("batterySaverEnabled") var batterySaverEnabled = false
    @AppStorage("debugSaveFrames") var debugSaveFrames = false

    // MARK: - Private

    private var settingsCancellables = Set<AnyCancellable>()
    private var engineCancellables = Set<AnyCancellable>()
    /// Non-isolated helper for JPEG conversion on background queues.
    private nonisolated let imageConverter = ImageConverter()

    // MARK: - Init

    init() {
        // Heal any out-of-range value an older debug build may have persisted
        // (its slider allowed 0.10–0.95); see ToxicityThresholds.sensitivityRange.
        let range = ToxicityThresholds.sensitivityRange
        if !range.contains(sensitivityThreshold) {
            sensitivityThreshold = min(max(sensitivityThreshold, range.lowerBound), range.upperBound)
        }
        observeSettingsChanges()
        forwardEngineChanges()
    }

    // MARK: - Patrol Lifecycle

    /// Starts the patrol pipeline: camera → inference → alert.
    func startPatrol() {
        syncSettingsToEngines()
        inferenceEngine.loadModel()
        detectionLogStore.requestLocationPermission()

        captureEngine.onFrameCaptured = { [weak self] pixelBuffer, captureContext in
            guard let self else { return }
            let saveFrames = self.debugSaveFrames

            self.inferenceEngine.classify(pixelBuffer: pixelBuffer) { [weak self] result in
                guard let self else { return }
                // Signal to the capture engine that processing is done so it can send the next frame.
                self.captureEngine.markFrameProcessingComplete()

                // Save every captured frame to disk when debug mode is on.
                if saveFrames {
                    let jpegForDebug = self.imageConverter.jpegData(from: pixelBuffer)
                    // Record the model's full verdict, not just a label: the active
                    // learning selector ranks this pool by confidence, and the
                    // near-miss frames (toxic class present but below its alert
                    // threshold) are the ones that reveal misses.
                    let sensitivity = Float(self.sensitivityThreshold)
                    let severity = result.map {
                        ToxicityThresholds.severity(plantType: $0.plantType,
                                                    confidence: $0.confidence,
                                                    sensitivity: sensitivity)
                    }
                    DebugFrameSaver.shared.save(
                        jpegData: jpegForDebug,
                        detection: result,
                        severity: severity,
                        context: captureContext
                    )
                }

                guard let result else { return }
                let loggedSeverity = ToxicityThresholds.severity(plantType: result.plantType,
                                                                 confidence: result.confidence,
                                                                 sensitivity: Float(self.sensitivityThreshold))
                DataRecorder.shared.logEvent(
                    "detection",
                    details: "class=\(result.plantType) conf=\(String(format: "%.3f", result.confidence)) sev=\(loggedSeverity) trigger=\(captureContext.trigger.rawValue)"
                )
                let imageData = self.imageConverter.jpegData(from: pixelBuffer)
                DispatchQueue.main.async {
                    self.handleDetection(result, imageData: imageData)
                }
            }
        }

        captureEngine.start()
        isPatrolling = true
    }

    /// The main-actor half of a detection: alerts, the on-screen card and box,
    /// and the log. Shared by the live pipeline and the DEBUG injection hook so
    /// the two paths cannot drift apart.
    private func handleDetection(_ result: DetectionResult, imageData: Data?) {
        // A late completion from an in-flight inference must not mutate state
        // or fire alerts after the user stopped the patrol.
        guard isPatrolling else { return }

        alertEngine.process(result)

        // Surface the warning card for any actionable toxic detection — both
        // full `.alert`s and `.uncertain` near-misses (shown with hedged "verify
        // visually" framing). Only truly sub-floor `.ignore` detections stay
        // silent. A safety app must never present a confident all-clear.
        if InferenceEngine.toxicLabels.contains(result.plantType),
           ToxicityThresholds.severity(
               plantType: result.plantType,
               confidence: result.confidence,
               sensitivity: Float(sensitivityThreshold)
           ).isActionable {
            lastDetection = result
            boxDetection = result
            scheduleBoxExpiry(for: result)
            scheduleCardExpiry(for: result)
        }

        // Log every detection regardless of gating so the map/history records
        // everything.
        detectionLogStore.save(result: result, imageData: imageData)
    }

    #if DEBUG
    /// Test hook: pushes a synthetic detection through the exact live path
    /// (alerts, card, box, log) so the detection UX can be exercised and
    /// UI-tested where there is no camera. `-injectDetection poison_ivy:0.72`.
    func injectDebugDetection(plantType: String, confidence: Float) {
        let result = DetectionResult(
            plantType: plantType,
            confidence: confidence,
            boundingBox: CGRect(x: 0.3, y: 0.35, width: 0.4, height: 0.3)
        )
        handleDetection(result, imageData: Self.placeholderJPEG())
    }

    private static func placeholderJPEG() -> Data? {
        let size = CGSize(width: 480, height: 640)
        return UIGraphicsImageRenderer(size: size).image { ctx in
            UIColor(red: 0.20, green: 0.45, blue: 0.20, alpha: 1).setFill()
            ctx.fill(CGRect(origin: .zero, size: size))
        }.jpegData(compressionQuality: 0.7)
    }
    #endif

    /// Stops the patrol pipeline.
    func stopPatrol() {
        captureEngine.stop()
        // A session recording is tied to the patrol: finalize it now so the .mov
        // is playable and metadata.json is written, instead of leaving an
        // unfinalized file behind when the user taps Stop out of habit.
        if DataRecorder.shared.isRecording {
            DataRecorder.shared.stop()
        }
        boxExpiryTask?.cancel()
        boxExpiryTask = nil
        cardExpiryTask?.cancel()
        cardExpiryTask = nil
        lastDetection = nil
        // Do NOT nil out captureEngine.onFrameCaptured here: it is read on the
        // capture queue while this runs on the main actor, and clearing it mid-
        // flight can strand CaptureEngine.isProcessingFrame (set before the
        // callback, cleared only inside it) — permanently dropping frames on the
        // next patrol. The `guard self.isPatrolling` check inside the inference
        // completion already neutralizes any late frame, and
        // markFrameProcessingComplete() still runs first to free the pipeline.
        isPatrolling = false
    }

    /// How long a detection's on-screen box stays valid.
    ///
    /// The bounding box describes where something was seen in ONE captured frame.
    /// The camera keeps moving, so after a moment that rectangle is a confident
    /// spatial claim about a scene that no longer exists — a hiker could see a box
    /// on the left and steer right while the plant is actually behind them. Captures
    /// run at roughly 1 Hz, so a couple of seconds keeps the box present between
    /// consecutive detections while still expiring a stale one.
    nonisolated static let detectionBoxLifetime: TimeInterval = 2.5

    /// How long the feedback card stays up without a newer detection.
    ///
    /// Deliberately much longer than the box: the card is what the user answers
    /// ("Correct" / "Wrong"), and that feedback is the only signal the active-
    /// learning loop gets. Tying the card to the box lifetime made it vanish in
    /// 2.5 s — before anyone could tap it.
    nonisolated static let detectionCardLifetime: TimeInterval = 20

    private var boxExpiryTask: Task<Void, Never>?
    private var cardExpiryTask: Task<Void, Never>?

    /// Clears the box once it can no longer be trusted, unless a newer detection
    /// has already replaced it.
    private func scheduleBoxExpiry(for result: DetectionResult) {
        boxExpiryTask?.cancel()
        boxExpiryTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(Self.detectionBoxLifetime * 1_000_000_000))
            guard !Task.isCancelled, let self else { return }
            if self.boxDetection?.id == result.id {
                self.boxDetection = nil
            }
        }
    }

    /// Clears the card after `detectionCardLifetime`, unless replaced or answered.
    private func scheduleCardExpiry(for result: DetectionResult) {
        cardExpiryTask?.cancel()
        cardExpiryTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(Self.detectionCardLifetime * 1_000_000_000))
            guard !Task.isCancelled, let self else { return }
            if self.lastDetection?.id == result.id {
                self.lastDetection = nil
            }
        }
    }

    /// Marks the first-launch disclaimer as shown.
    func markDisclaimerShown() {
        hasShownDisclaimer = true
    }

    // MARK: - Private

    private func syncSettingsToEngines() {
        alertEngine.sensitivityThreshold = Float(sensitivityThreshold)
        alertEngine.audioAlertsEnabled = audioAlertsEnabled
        captureEngine.isBatterySaverEnabled = batterySaverEnabled
    }

    /// Forwards objectWillChange from child engines so SwiftUI views that read
    /// nested engine properties (e.g. `appState.captureEngine.pipelineActive`) update correctly.
    private func forwardEngineChanges() {
        captureEngine.objectWillChange
            // The engine publishes motion telemetry at ~10 Hz, and every view
            // observing AppState re-evaluates on each forward. Cap it at 5 Hz —
            // ample for debug readouts — to halve main-thread churn on patrol.
            .throttle(for: .milliseconds(200), scheduler: DispatchQueue.main, latest: true)
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &engineCancellables)
    }

    /// Watches UserDefaults for settings changes and live-syncs them to engines during an active patrol.
    private func observeSettingsChanges() {
        let defaults = UserDefaults.standard

        defaults.publisher(for: \.sensitivityThreshold)
            .dropFirst()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] newValue in
                guard let self, self.isPatrolling else { return }
                self.alertEngine.sensitivityThreshold = Float(newValue)
            }
            .store(in: &settingsCancellables)

        defaults.publisher(for: \.audioAlertsEnabled)
            .dropFirst()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] newValue in
                guard let self, self.isPatrolling else { return }
                self.alertEngine.audioAlertsEnabled = newValue
            }
            .store(in: &settingsCancellables)

        defaults.publisher(for: \.batterySaverEnabled)
            .dropFirst()
            .receive(on: DispatchQueue.main)
            .sink { [weak self] newValue in
                guard let self, self.isPatrolling else { return }
                self.captureEngine.isBatterySaverEnabled = newValue
            }
            .store(in: &settingsCancellables)
    }
}

// MARK: - Image Conversion (non-isolated, safe to call from any queue)

private final class ImageConverter: Sendable {
    private let ciContext = CIContext()

    func jpegData(from pixelBuffer: CVPixelBuffer) -> Data? {
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        return ciContext.jpegRepresentation(of: ciImage, colorSpace: colorSpace, options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.85])
    }
}

// MARK: - UserDefaults KVO Key Paths

/// Extends UserDefaults with `@objc dynamic` key-path accessors so Combine's
/// `publisher(for:)` can observe the AppStorage-backed settings.
extension UserDefaults {
    @objc dynamic var sensitivityThreshold: Double {
        double(forKey: "sensitivityThreshold")
    }

    @objc dynamic var audioAlertsEnabled: Bool {
        bool(forKey: "audioAlertsEnabled")
    }

    @objc dynamic var batterySaverEnabled: Bool {
        bool(forKey: "batterySaverEnabled")
    }
}
