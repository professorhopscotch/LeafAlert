import Foundation
import UIKit

/// Exports user feedback (confirmed/corrected detections with images) to the app's
/// Documents directory AND optionally to a user-chosen iCloud Drive folder.
///
/// Local files are written to:
///   <App Documents>/feedback/
///     ├── <timestamp>_<label>_<status>.jpg
///     ├── manifest.json
///     └── ...
///
/// When the user picks an iCloud Drive folder via the document picker in Settings,
/// feedback is also written there. iCloud syncs it to the Mac automatically.
final class FeedbackExporter: ObservableObject {

    static let shared = FeedbackExporter()

    private let feedbackDirectoryName = "feedback"
    /// Serializes all file I/O to prevent concurrent read-modify-write races on manifest.json.
    private let ioQueue = DispatchQueue(label: "com.leafalert.feedback-io")

    /// Whether an iCloud sync folder has been configured.
    @Published private(set) var hasSyncFolder = false
    /// Display name of the sync folder (e.g., "iCloud Drive/LeafAlert").
    @Published private(set) var syncFolderName: String?

    private init() {
        loadBookmark()
    }

    /// The local feedback directory URL.
    var feedbackDirectoryURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent(feedbackDirectoryName)
    }

    // MARK: - iCloud Sync Folder

    /// The bookmarked iCloud Drive folder URL (security-scoped).
    private var syncFolderURL: URL?

    /// Read-only access to the sync folder URL for other components (e.g. DataRecorder).
    /// Caller is responsible for managing security-scoped resource access.
    var configuredSyncFolderURL: URL? { syncFolderURL }

    private var bookmarkURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent(".sync_folder_bookmark")
    }

    /// Save a user-picked folder URL as the sync destination.
    /// Copies any existing local feedback into the new sync folder.
    func setSyncFolder(_ url: URL) {
        guard url.startAccessingSecurityScopedResource() else { return }
        defer { url.stopAccessingSecurityScopedResource() }

        // Create a bookmark so we can access this folder across app launches.
        guard let bookmarkData = try? url.bookmarkData(
            options: .minimalBookmark,
            includingResourceValuesForKeys: nil,
            relativeTo: nil
        ) else { return }

        try? bookmarkData.write(to: bookmarkURL, options: .atomic)
        syncFolderURL = url

        DispatchQueue.main.async {
            self.hasSyncFolder = true
            self.syncFolderName = url.lastPathComponent
        }

        // Copy existing local feedback to the new sync folder
        ioQueue.async { [self] in
            copyExistingFeedbackToSyncFolder(url)
        }
    }

    /// Copies all existing local feedback files into the newly-configured sync folder.
    private func copyExistingFeedbackToSyncFolder(_ syncURL: URL) {
        guard syncURL.startAccessingSecurityScopedResource() else { return }
        defer { syncURL.stopAccessingSecurityScopedResource() }

        let localDir = feedbackDirectoryURL
        let cloudDir = syncURL.appendingPathComponent("feedback")
        try? FileManager.default.createDirectory(at: cloudDir, withIntermediateDirectories: true)

        let fm = FileManager.default
        guard let localFiles = try? fm.contentsOfDirectory(at: localDir, includingPropertiesForKeys: nil) else { return }

        for file in localFiles {
            let dest = cloudDir.appendingPathComponent(file.lastPathComponent)
            if !fm.fileExists(atPath: dest.path) {
                try? fm.copyItem(at: file, to: dest)
            }
        }
    }

    /// Remove the sync folder configuration.
    func clearSyncFolder() {
        try? FileManager.default.removeItem(at: bookmarkURL)
        syncFolderURL = nil
        DispatchQueue.main.async {
            self.hasSyncFolder = false
            self.syncFolderName = nil
        }
    }

    private func loadBookmark() {
        guard let bookmarkData = try? Data(contentsOf: bookmarkURL) else { return }
        var isStale = false
        guard let url = try? URL(
            resolvingBookmarkData: bookmarkData,
            bookmarkDataIsStale: &isStale
        ) else { return }

        syncFolderURL = url
        hasSyncFolder = true
        syncFolderName = url.lastPathComponent

        // Re-save if stale
        if isStale {
            setSyncFolder(url)
        }
    }

    // MARK: - Public API

    /// Exports a single feedback entry (image + metadata) to the local feedback directory
    /// and optionally to the iCloud sync folder.
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

            // Append to local manifest.json
            let manifestURL = feedbackDir.appendingPathComponent("manifest.json")
            self.appendToManifest(entry: entry, at: manifestURL)

            // Sync to iCloud Drive folder if configured
            self.syncToCloudFolder(imageData: imageData, filename: filename, entry: entry)

            // Auto-upload to sync server if available
            FeedbackUploader.shared.uploadEntry(
                imageData: imageData,
                metadata: entry,
                filename: filename
            )
        }
    }

    /// Returns the feedback directory URL for sharing via UIActivityViewController.
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

    private func syncToCloudFolder(imageData: Data?, filename: String, entry: [String: Any]) {
        guard let syncURL = syncFolderURL else { return }
        guard syncURL.startAccessingSecurityScopedResource() else { return }
        defer { syncURL.stopAccessingSecurityScopedResource() }

        let feedbackDir = syncURL.appendingPathComponent("feedback")
        try? FileManager.default.createDirectory(at: feedbackDir, withIntermediateDirectories: true)

        // Write image
        if let imageData {
            let imageURL = feedbackDir.appendingPathComponent(filename)
            try? imageData.write(to: imageURL)
        }

        // Append to cloud manifest
        let manifestURL = feedbackDir.appendingPathComponent("manifest.json")
        appendToManifest(entry: entry, at: manifestURL)
    }

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
