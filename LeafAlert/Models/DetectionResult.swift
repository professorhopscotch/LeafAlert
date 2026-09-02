import Foundation
import CoreGraphics

/// Result from a single inference pass on a camera frame.
struct DetectionResult: Identifiable, Sendable {
    let id: UUID
    let plantType: String
    let confidence: Float
    let boundingBox: CGRect
    let timestamp: Date

    init(
        id: UUID = UUID(),
        plantType: String,
        confidence: Float,
        boundingBox: CGRect,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.plantType = plantType
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.timestamp = timestamp
    }
}

/// How strongly a toxic detection should be surfaced to the user.
///
/// For a safety app the dangerous error is a MISSED toxic plant, so a
/// mid-confidence detection is surfaced as `.uncertain` ("verify visually")
/// rather than silently ignored. The app never presents a confident all-clear.
enum DetectionSeverity: Equatable {
    /// High enough confidence for a full haptic + audio warning.
    case alert
    /// "Possible toxic plant — verify visually." Soft cue only, no audio.
    case uncertain
    /// Below the noise floor — logged, but not surfaced.
    case ignore

    var isActionable: Bool { self != .ignore }
}

/// Per-class decision thresholds for toxic-plant alerts.
///
/// These come from held-out evaluation (scripts/evaluate_model.py): the current
/// model separates poison_ivy / poison_oak weakly, so they alert at a lower bar,
/// while poison_sumac is better calibrated and alerts higher. IMPORTANT: with the
/// current data-limited model these thresholds trade recall against a real
/// false-alarm rate — raising recall toward 80% pushes false alarms high. The
/// durable fix is more/better training data, not threshold tuning; tune here only
/// as a stopgap and re-derive after each retrain.
enum ToxicityThresholds {

    /// Per-class base alert thresholds at the default sensitivity (0.50), re-derived
    /// for the v5 model (train_v5.py) on the frozen held-out set via
    /// scripts/evaluate_model.py. poison_ivy/oak alert lower (weaker separation),
    /// poison_sumac higher (well-calibrated). Re-derive after every retrain.
    static let baseAlert: [String: Float] = [
        "poison_ivy": 0.40,
        "poison_oak": 0.40,
        "poison_sumac": 0.52,
    ]
    private static let defaultAlert: Float = 0.50

    /// The neutral sensitivity value at which the base thresholds apply.
    static let neutralSensitivity: Float = 0.50

    /// Allowed range for the user-facing sensitivity slider. The Settings slider
    /// AND the debug Live Controls slider must both use this: the debug one used
    /// to allow 0.10–0.95, so a single flick persisted a value the Settings
    /// slider could not even display and that silently over- or de-sensitised
    /// every later patrol until someone noticed.
    static let sensitivityRange: ClosedRange<Double> = 0.3...0.8

    /// Confidence band immediately below the alert bar that is still surfaced as
    /// `.uncertain` instead of ignored — so a near-miss toxic plant is never a
    /// silent all-clear.
    static let uncertaintyMargin: Float = 0.20

    /// The user's sensitivity slider biases every class threshold: values above
    /// the neutral 0.50 are stricter (fewer alerts), below are more sensitive.
    static func alertThreshold(for plantType: String, sensitivity: Float) -> Float {
        let base = baseAlert[plantType] ?? defaultAlert
        return min(max(base + (sensitivity - neutralSensitivity), 0.15), 0.95)
    }

    /// Classifies a toxic detection's confidence into an actionable severity.
    static func severity(plantType: String, confidence: Float, sensitivity: Float) -> DetectionSeverity {
        let threshold = alertThreshold(for: plantType, sensitivity: sensitivity)
        if confidence >= threshold { return .alert }
        if confidence >= threshold - uncertaintyMargin { return .uncertain }
        return .ignore
    }
}
