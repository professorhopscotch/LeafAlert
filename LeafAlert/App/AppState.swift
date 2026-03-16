import SwiftUI
import Combine

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
    @Published var hasShownDisclaimer: Bool

    /// Tracks when the last detection occurred so we can auto-expire stale banners.
    private var lastDetectionTime: Date = .distantPast

    /// How long a detection banner stays visible before auto-clearing (seconds).
    private let detectionExpiryInterval: TimeInterval = 10.0

    /// Timer that clears stale detection banners.
    private var detectionExpiryTimer: Timer?

    // MARK: - Settings (synced from UserDefaults via AppStorage)

    @AppStorage("sensitivityThreshold") var sensitivityThreshold: Double = 0.65
    @AppStorage("audioAlertsEnabled") var audioAlertsEnabled = true
    @AppStorage("screenDimLevel") var screenDimLevel: Double = 0.7
    @AppStorage("batterySaverEnabled") var batterySaverEnabled = false

    // MARK: - Private

    private var settingsCancellables = Set<AnyCancellable>()

    // MARK: - Init

    init() {
        self.hasShownDisclaimer = UserDefaults.standard.bool(forKey: "hasShownDisclaimer")
        observeSettingsChanges()
    }

    // MARK: - Patrol Lifecycle

    /// Starts the patrol pipeline: camera → inference → alert.
    func startPatrol() {
        syncSettingsToEngines()
        inferenceEngine.loadModel()
        detectionLogStore.requestLocationPermission()

        captureEngine.onFrameCaptured = { [weak self] pixelBuffer in
            self?.inferenceEngine.classify(pixelBuffer: pixelBuffer) { result in
                guard let self else { return }
                // Signal to the capture engine that processing is done so it can send the next frame.
                self.captureEngine.markFrameProcessingComplete()

                guard let result else { return }
                self.alertEngine.process(result)
                DispatchQueue.main.async {
                    self.lastDetection = result
                    self.lastDetectionTime = Date()
                    self.detectionLogStore.save(result: result, imageData: nil)
                }
            }
        }

        captureEngine.start()
        isPatrolling = true
        startDetectionExpiryTimer()
    }

    /// Stops the patrol pipeline.
    func stopPatrol() {
        captureEngine.stop()
        isPatrolling = false
        detectionExpiryTimer?.invalidate()
        detectionExpiryTimer = nil
    }

    /// Marks the first-launch disclaimer as shown.
    func markDisclaimerShown() {
        hasShownDisclaimer = true
        UserDefaults.standard.set(true, forKey: "hasShownDisclaimer")
    }

    // MARK: - Private

    private func syncSettingsToEngines() {
        alertEngine.sensitivityThreshold = Float(sensitivityThreshold)
        alertEngine.audioAlertsEnabled = audioAlertsEnabled
        captureEngine.isBatterySaverEnabled = batterySaverEnabled
    }

    /// Periodically checks if the detection banner should auto-expire.
    private func startDetectionExpiryTimer() {
        detectionExpiryTimer?.invalidate()
        detectionExpiryTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            if self.lastDetection != nil &&
               Date().timeIntervalSince(self.lastDetectionTime) >= self.detectionExpiryInterval {
                self.lastDetection = nil
            }
        }
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
