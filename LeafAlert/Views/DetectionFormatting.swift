import SwiftUI

/// Shared formatting helpers used by HomeView and PatrolMapView.
enum DetectionFormatting {

    /// The safety disclaimer shown across multiple screens.
    static let safetyDisclaimer =
        "This app assists identification — always verify visually before touching any plant."

    /// Converts a snake_case plant type ID to a human-readable title.
    static func plantDisplayName(_ raw: String) -> String {
        raw.replacingOccurrences(of: "_", with: " ").capitalized
    }

    /// A hedged headline for a detection. The app must never imply certainty, so
    /// a full alert reads "Likely …" and an uncertain near-miss reads "Possible …".
    static func detectionHeadline(_ plantType: String, severity: DetectionSeverity) -> String {
        let name = plantDisplayName(plantType)
        switch severity {
        case .alert:     return "Likely \(name)"
        case .uncertain: return "Possible \(name)"
        case .ignore:    return name
        }
    }

    /// A hedged sub-line that always points the user back to visual verification
    /// and never presents the raw confidence as a certainty.
    static func detectionSubtitle(confidence: Float, severity: DetectionSeverity) -> String {
        switch severity {
        case .alert:
            return "~\(Int(confidence * 100))% match · always verify before touching"
        case .uncertain:
            return "Low confidence · verify visually before touching"
        case .ignore:
            return "Verify visually before touching"
        }
    }

    /// Returns a color representing the severity of a confidence value. Bands are
    /// tuned to the model's actual (softmax) confidence distribution.
    static func confidenceColor(_ confidence: Float) -> Color {
        switch confidence {
        case ..<0.50:  return .yellow
        case 0.50..<0.70: return .orange
        default: return .red
        }
    }

    /// Formats a date as a relative string (e.g. "2 hours ago").
    static func relativeTimestamp(_ date: Date) -> String {
        Self.relativeDateFormatter.localizedString(for: date, relativeTo: .now)
    }

    // MARK: - Private

    private static let relativeDateFormatter: RelativeDateTimeFormatter = {
        let formatter = RelativeDateTimeFormatter()
        formatter.unitsStyle = .full
        return formatter
    }()
}
