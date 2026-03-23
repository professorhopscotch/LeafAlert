import Foundation
import UIKit

/// Exports user feedback (confirmed/corrected detections with images) to the app's
/// Documents directory for transfer to the training pipeline via AirDrop or Files app.
///
/// Files are written to:
///   <App Documents>/feedback/
///     ├── <timestamp>_<label>_<status>.jpg
///     ├── manifest.json
///     └── ...
///
/// To transfer: open Files app → On My iPhone → LeafAlert → feedback/
/// Then AirDrop or share the folder to your Mac.
final class FeedbackExporter {

    static let shared = FeedbackExporter()

    private let feedbackDirectoryName = "feedback"
    /// Serializes all file I/O to prevent concurrent read-modify-write races on manifest.json.
    private let ioQueue = DispatchQueue(label: "com.leafalert.feedback-io")

    private init() {}

    /// The local feedback directory URL.
    var feedbackDirectoryURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent(feedbackDirectoryName)
    }

    // MARK: - Public API

    /// Exports a single feedback entry (image + metadata) to the local feedback directory.
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
        ioQueue.async { [self] in
            let feedbackDir = feedbackDirectoryURL

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

            // Append to manifest.json (read-modify-write, serialized by ioQueue)
            let manifestURL = feedbackDir.appendingPathComponent("manifest.json")
            self.appendToManifest(entry: entry, at: manifestURL)

            // Auto-upload to sync server if available
            FeedbackUploader.shared.uploadEntry(
                imageData: imageData,
                metadata: entry,
                filename: filename
            )
        }
    }

    /// Returns the feedback directory URL for sharing via UIActivityViewController.
    /// Returns nil if no feedback has been exported yet.
    func shareableURL() -> URL? {
        let dir = feedbackDirectoryURL
        guard FileManager.default.fileExists(atPath: dir.path) else { return nil }
        return dir
    }

    /// Returns the count of feedback entries in the manifest.
    var feedbackCount: Int {
        let manifestURL = feedbackDirectoryURL.appendingPathComponent("manifest.json")
        guard let data = try? Data(contentsOf: manifestURL),
              let manifest = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let entries = manifest["entries"] as? [[String: Any]] else {
            return 0
        }
        return entries.count
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
