import SwiftUI
import Combine

/// Top-level observable state shared across the app via EnvironmentObject.
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

    // MARK: - Settings (synced from UserDefaults via AppStorage)

    @AppStorage("sensitivityThreshold") var sensitivityThreshold: Double = 0.65
    @AppStorage("audioAlertsEnabled") var audioAlertsEnabled = true
    @AppStorage("screenDimLevel") var screenDimLevel: Double = 0.7
    @AppStorage("batterySaverEnabled") var batterySaverEnabled = false

    // MARK: - Init

    init() {
        self.hasShownDisclaimer = UserDefaults.standard.bool(forKey: "hasShownDisclaimer")
    }

    // MARK: - Patrol Lifecycle

    /// Starts the patrol pipeline: camera → inference → alert.
    func startPatrol() {
        syncSettingsToEngines()
        inferenceEngine.loadModel()
        detectionLogStore.requestLocationPermission()

        captureEngine.onFrameCaptured = { [weak self] pixelBuffer in
            self?.inferenceEngine.classify(pixelBuffer: pixelBuffer) { result in
                guard let self, let result else { return }
                DispatchQueue.main.async {
                    self.lastDetection = result
                }
                self.alertEngine.process(result)
                self.detectionLogStore.save(result: result, imageData: nil)
            }
        }

        captureEngine.start()
        isPatrolling = true
    }

    /// Stops the patrol pipeline.
    func stopPatrol() {
        captureEngine.stop()
        isPatrolling = false
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
}
