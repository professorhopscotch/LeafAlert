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

    /// Returns a color representing the severity of a confidence value.
    static func confidenceColor(_ confidence: Float) -> Color {
        switch confidence {
        case ..<0.75:  return .yellow
        case 0.75..<0.90: return .orange
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
