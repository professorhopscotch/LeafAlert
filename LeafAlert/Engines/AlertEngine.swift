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

    /// Evaluates a detection result and fires the appropriate haptic/audio alert.
    /// - Parameter result: The detection result from the inference engine.
    /// - Returns: The alert level that was triggered, or nil if suppressed.
    @discardableResult
    func process(_ result: DetectionResult) -> AlertLevel? {
        // Ignore negative classifications
        guard result.plantType != "negative" else { return nil }

        // Below threshold — silent log only
        guard result.confidence >= sensitivityThreshold else { return nil }

        // Cooldown check
        let now = Date()
        guard now.timeIntervalSince(lastAlertTime) >= cooldownInterval else { return nil }
        lastAlertTime = now

        let level = AlertLevel(confidence: result.confidence, threshold: sensitivityThreshold)
        fireHaptic(for: level)
        if audioAlertsEnabled {
            playAudio(for: level)
        }
        return level
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
