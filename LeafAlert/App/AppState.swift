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
    @Published var lastDetection: DetectionResult?
    @AppStorage("hasShownDisclaimer") var hasShownDisclaimer = false

    // MARK: - Settings (synced from UserDefaults via AppStorage)

    @AppStorage("sensitivityThreshold") var sensitivityThreshold: Double = 0.65
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
                    DebugFrameSaver.shared.save(
                        jpegData: jpegForDebug,
                        classification: result?.plantType,
                        context: captureContext
                    )
                }

                guard let result else { return }
                DataRecorder.shared.logEvent(
                    "detection",
                    details: "class=\(result.plantType) conf=\(String(format: "%.3f", result.confidence))"
                )
                let imageData = self.imageConverter.jpegData(from: pixelBuffer)
                DispatchQueue.main.async {
                    // A late completion from an in-flight inference must not mutate
                    // state or fire alerts after the user stopped the patrol.
                    guard self.isPatrolling else { return }

                    self.alertEngine.process(result)

                    // Only surface the warning card / bounding box for alert-worthy
                    // detections: a toxic class at or above the sensitivity threshold.
                    // This mirrors AlertEngine.process's own gating. Safe plants and
                    // sub-threshold detections are still logged, just not surfaced.
                    if InferenceEngine.toxicLabels.contains(result.plantType)
                        && result.confidence >= Float(self.sensitivityThreshold) {
                        self.lastDetection = result
                    }

                    // Log every detection regardless of gating so the map/history
                    // records everything.
                    self.detectionLogStore.save(result: result, imageData: imageData)
                }
            }
        }

        captureEngine.start()
        isPatrolling = true
    }

    /// Stops the patrol pipeline.
    func stopPatrol() {
        captureEngine.stop()
        // Do NOT nil out captureEngine.onFrameCaptured here: it is read on the
        // capture queue while this runs on the main actor, and clearing it mid-
        // flight can strand CaptureEngine.isProcessingFrame (set before the
        // callback, cleared only inside it) — permanently dropping frames on the
        // next patrol. The `guard self.isPatrolling` check inside the inference
        // completion already neutralizes any late frame, and
        // markFrameProcessingComplete() still runs first to free the pipeline.
        isPatrolling = false
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
            .receive(on: DispatchQueue.main)
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
