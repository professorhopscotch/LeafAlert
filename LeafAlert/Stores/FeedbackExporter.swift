import Foundation
import UIKit

/// Exports user feedback (confirmed/corrected detections with images) to iCloud Drive
/// for ingestion into the training pipeline.
///
/// Files are written to the app's iCloud ubiquity container at:
///   iCloud Drive/LeafAlert/feedback/
///     ├── <timestamp>_<label>_<status>.jpg
///     ├── manifest.json
///     └── ...
final class FeedbackExporter {

    static let shared = FeedbackExporter()

    /// The subdirectory within the iCloud container for feedback data
    private let feedbackDirectoryName = "feedback"

    private init() {}

    // MARK: - Public API

    /// Exports a single feedback entry (image + metadata) to iCloud Drive.
    /// Call this whenever the user submits feedback on a detection.
    func exportFeedback(
        imageData: Data?,
        originalPrediction: String,
        correctedLabel: String,
        feedbackStatus: String,
        confidence: Float,
        latitude: Double,
        longitude: Double,
        timestamp: Date
    ) {
        guard let containerURL = FileManager.default.url(forUbiquityContainerIdentifier: nil) else {
            print("[FeedbackExporter] iCloud container not available. Feedback not exported.")
            return
        }

        let feedbackDir = containerURL
            .appendingPathComponent("Documents")
            .appendingPathComponent(feedbackDirectoryName)

        // Create directory if needed
        try? FileManager.default.createDirectory(at: feedbackDir, withIntermediateDirectories: true)

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        let timestampStr = formatter.string(from: timestamp)
            .replacingOccurrences(of: ":", with: "-")

        // Save image
        let filename = "\(timestampStr)_\(correctedLabel)_\(feedbackStatus).jpg"
        if let imageData = imageData {
            let imageURL = feedbackDir.appendingPathComponent(filename)
            try? imageData.write(to: imageURL)
        }

        // Build manifest entry
        let entry: [String: Any] = [
            "filename": filename,
            "originalPrediction": originalPrediction,
            "correctedLabel": correctedLabel,
            "feedbackStatus": feedbackStatus,
            "confidence": confidence,
            "timestamp": formatter.string(from: timestamp),
            "latitude": latitude,
            "longitude": longitude
        ]

        // Append to manifest.json (read-modify-write)
        let manifestURL = feedbackDir.appendingPathComponent("manifest.json")
        appendToManifest(entry: entry, at: manifestURL)
    }

    // MARK: - Private

    private func appendToManifest(entry: [String: Any], at url: URL) {
        var manifest: [String: Any]

        if let existingData = try? Data(contentsOf: url),
           let existing = try? JSONSerialization.jsonObject(with: existingData) as? [String: Any] {
            manifest = existing
        } else {
            manifest = ["version": 1, "entries": []]
        }

        var entries = manifest["entries"] as? [[String: Any]] ?? []
        entries.append(entry)
        manifest["entries"] = entries

        if let data = try? JSONSerialization.data(withJSONObject: manifest, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
