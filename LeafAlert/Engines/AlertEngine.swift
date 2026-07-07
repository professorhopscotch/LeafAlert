import UIKit
import AVFoundation

/// Produces haptic and audio alerts based on detection confidence levels.
final class AlertEngine: ObservableObject {

    // MARK: - Configuration

    /// Confidence threshold below which detections are silently logged.
    var sensitivityThreshold: Float = 0.65

    /// Whether audio alerts are enabled.
    var audioAlertsEnabled = true

    // MARK: - Private Properties

    private var lastAlertTime: Date = .distantPast
    private let cooldownInterval: TimeInterval = 8.0
    private var audioPlayer: AVAudioPlayer?

    // MARK: - Alert Delivery

    /// Evaluates a detection result and fires the appropriate haptic/audio cue.
    ///
    /// A full `.alert` fires haptic + (optional) audio. An `.uncertain` near-miss
    /// fires only a soft, silent haptic nudge to prompt a visual check without the
    /// full alarm. `.ignore` is silent.
    /// - Returns: The severity that was evaluated.
    @discardableResult
    func process(_ result: DetectionResult) -> DetectionSeverity {
        // Only consider toxic plant detections.
        guard InferenceEngine.toxicLabels.contains(result.plantType) else { return .ignore }
        // Sanity check: ignore impossible confidence values.
        guard result.confidence <= 1.0 else { return .ignore }

        let severity = ToxicityThresholds.severity(
            plantType: result.plantType,
            confidence: result.confidence,
            sensitivity: sensitivityThreshold
        )
        guard severity != .ignore else { return .ignore }

        // Cooldown so we don't buzz continuously on the same plant.
        let now = Date()
        guard now.timeIntervalSince(lastAlertTime) >= cooldownInterval else { return severity }
        lastAlertTime = now

        switch severity {
        case .alert:
            let level = AlertLevel(confidence: result.confidence, threshold: sensitivityThreshold)
            fireHaptic(for: level)
            if audioAlertsEnabled {
                playAudio(for: level)
            }
        case .uncertain:
            // Soft, silent nudge for a near-miss: enough to prompt a visual check.
            fireHaptic(for: .low)
        case .ignore:
            break
        }
        return severity
    }

    // MARK: - Alert Levels

    enum AlertLevel {
        case low
        case medium
        case high

        init(confidence: Float, threshold: Float) {
            switch confidence {
            case ..<0.75:
                self = .low
            case 0.75..<0.90:
                self = .medium
            default:
                self = .high
            }
        }
    }

    // MARK: - Haptics

    private func fireHaptic(for level: AlertLevel) {
        switch level {
        case .low:
            let generator = UIImpactFeedbackGenerator(style: .light)
            generator.impactOccurred()
        case .medium:
            let generator = UIImpactFeedbackGenerator(style: .medium)
            generator.impactOccurred()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
                generator.impactOccurred()
            }
        case .high:
            let generator = UINotificationFeedbackGenerator()
            generator.notificationOccurred(.warning)
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                generator.notificationOccurred(.warning)
            }
        }
    }

    // MARK: - Audio

    private func playAudio(for level: AlertLevel) {
        switch level {
        case .low:
            break  // No audio for low-confidence detections
        case .medium:
            playSystemSound()
        case .high:
            playSystemSound()
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
                self?.playSystemSound()
            }
        }
    }

    private func playSystemSound() {
        AudioServicesPlaySystemSound(1007)  // Standard system "chirp"
    }
}
