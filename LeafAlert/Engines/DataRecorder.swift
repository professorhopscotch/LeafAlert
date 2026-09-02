import AVFoundation
import CoreMotion
import UIKit

/// Records a complete sensor session to disk for offline playback and analysis.
///
/// Output structure (one folder per session):
///   Documents/recordings/<session_id>/
///     ├── video.mov              — continuous H.264 video at native frame rate
///     ├── imu.csv                — 100 Hz IMU samples (accel, gyro, attitude)
///                                 + motion_ts on the device uptime clock
///     ├── events.csv             — capture triggers, detections, gate decisions
///     └── metadata.json          — session info, settings, model version
///
/// Files are accessible via Files app → On My iPhone → LeafAlert → recordings/.
final class DataRecorder {

    static let shared = DataRecorder()

    // MARK: - State

    private let stateLock = NSLock()
    private var _isRecording = false
    private(set) var sessionID: String = ""
    private(set) var startTime: Date = .distantPast

    var isRecording: Bool {
        stateLock.lock(); defer { stateLock.unlock() }
        return _isRecording
    }

    var elapsedSeconds: TimeInterval {
        guard isRecording else { return 0 }
        return Date().timeIntervalSince(startTime)
    }

    var sessionFolderSize: Int64 {
        guard !sessionID.isEmpty else { return 0 }
        return Self.folderSize(at: sessionFolderURL)
    }

    // MARK: - Private

    private let recordQueue = DispatchQueue(label: "com.leafalert.recorder", qos: .userInitiated)
    private let ioQueue = DispatchQueue(label: "com.leafalert.recorder-io", qos: .utility)

    private var assetWriter: AVAssetWriter?
    private var videoInput: AVAssetWriterInput?
    private var sessionStarted = false
    private var firstFrameTime: CMTime = .zero

    private var imuFileHandle: FileHandle?
    private var eventsFileHandle: FileHandle?
    private var imuSampleCount: Int = 0
    private var eventCount: Int = 0

    /// If non-nil, the iCloud sync folder we hold a security scope on for the recording's lifetime.
    private var heldSyncFolderURL: URL?
    /// The base "recordings" directory chosen at start time (sync folder if available, else local).
    private var recordingsBaseURL: URL?

    private var sessionFolderURL: URL {
        guard let base = recordingsBaseURL else {
            // Fallback for queries outside an active session
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            return docs.appendingPathComponent("recordings").appendingPathComponent(sessionID)
        }
        return base.appendingPathComponent(sessionID)
    }

    private var localRecordingsURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("recordings")
    }

    private init() {}

    // MARK: - Lifecycle

    /// Start a new recording session. Returns true on success.
    @discardableResult
    func start(videoSettings: [String: Any]) -> Bool {
        stateLock.lock()
        guard !_isRecording else { stateLock.unlock(); return false }
        _isRecording = true
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        sessionID = formatter.string(from: Date())
        startTime = Date()
        stateLock.unlock()

        // Choose recording destination: iCloud sync folder if configured, else local Documents.
        if let syncRoot = FeedbackExporter.shared.configuredSyncFolderURL,
           syncRoot.startAccessingSecurityScopedResource() {
            heldSyncFolderURL = syncRoot
            recordingsBaseURL = syncRoot.appendingPathComponent("recordings")
        } else {
            heldSyncFolderURL = nil
            recordingsBaseURL = localRecordingsURL
        }

        let folder = sessionFolderURL
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)

        // Set up asset writer for video
        let videoURL = folder.appendingPathComponent("video.mov")
        try? FileManager.default.removeItem(at: videoURL)

        do {
            let writer = try AVAssetWriter(outputURL: videoURL, fileType: .mov)
            let input = AVAssetWriterInput(mediaType: .video, outputSettings: videoSettings)
            input.expectsMediaDataInRealTime = true
            if writer.canAdd(input) {
                writer.add(input)
            }
            self.assetWriter = writer
            self.videoInput = input
            self.sessionStarted = false
            writer.startWriting()
        } catch {
            print("[DataRecorder] Failed to create asset writer: \(error)")
            // Release the iCloud security scope if we acquired one above; the success
            // path releases it at stop(), but on this error path stop() never runs.
            if let held = heldSyncFolderURL {
                held.stopAccessingSecurityScopedResource()
                heldSyncFolderURL = nil
            }
            recordingsBaseURL = nil
            stateLock.lock(); _isRecording = false; stateLock.unlock()
            return false
        }

        // Set up IMU CSV
        let imuURL = folder.appendingPathComponent("imu.csv")
        let imuHeader = "timestamp_s,accel_x,accel_y,accel_z,user_accel_x,user_accel_y,user_accel_z,gravity_x,gravity_y,gravity_z,rot_x,rot_y,rot_z,roll,pitch,yaw,motion_ts\n"
        try? imuHeader.data(using: .utf8)?.write(to: imuURL)
        imuFileHandle = try? FileHandle(forWritingTo: imuURL)
        imuFileHandle?.seekToEndOfFile()

        // Set up events CSV
        let eventsURL = folder.appendingPathComponent("events.csv")
        let eventsHeader = "timestamp_s,event_type,details\n"
        try? eventsHeader.data(using: .utf8)?.write(to: eventsURL)
        eventsFileHandle = try? FileHandle(forWritingTo: eventsURL)
        eventsFileHandle?.seekToEndOfFile()

        imuSampleCount = 0
        eventCount = 0

        logEvent("session_start", details: "")
        return true
    }

    /// Stop the current recording session and finalize all files.
    func stop(completion: (() -> Void)? = nil) {
        stateLock.lock()
        guard _isRecording else { stateLock.unlock(); completion?(); return }
        _isRecording = false
        stateLock.unlock()

        logEvent("session_end", details: "samples=\(imuSampleCount) events=\(eventCount)")

        // Close CSVs
        ioQueue.async { [self] in
            try? imuFileHandle?.close()
            try? eventsFileHandle?.close()
            imuFileHandle = nil
            eventsFileHandle = nil
        }

        // Finalize video
        recordQueue.async { [self] in
            videoInput?.markAsFinished()
            assetWriter?.finishWriting { [self] in
                let duration = Date().timeIntervalSince(startTime)
                // Read the counters on ioQueue (the queue that mutates them) so the
                // snapshot reflects all writes and metadata.json can't under-count.
                var imuSamples = 0
                var events = 0
                ioQueue.sync {
                    imuSamples = imuSampleCount
                    events = eventCount
                }
                let metadata: [String: Any] = [
                    "session_id": sessionID,
                    "start_time": ISO8601DateFormatter().string(from: startTime),
                    "duration_s": duration,
                    "imu_samples": imuSamples,
                    "events": events,
                    "video_path": "video.mov",
                    "imu_path": "imu.csv",
                    "events_path": "events.csv",
                    "app_version": Bundle.main.infoDictionary?["CFBundleShortVersionString"] ?? "unknown"
                ]
                if let data = try? JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys]) {
                    let url = sessionFolderURL.appendingPathComponent("metadata.json")
                    try? data.write(to: url)
                }
                assetWriter = nil
                videoInput = nil

                // Release the iCloud security scope if we held one
                if let held = heldSyncFolderURL {
                    held.stopAccessingSecurityScopedResource()
                    heldSyncFolderURL = nil
                }
                recordingsBaseURL = nil

                completion?()
            }
        }
    }

    // MARK: - Sample Inputs

    /// Append a video frame to the recording. Safe to call from any queue.
    /// Should be called on every camera frame, regardless of inference gating.
    func appendVideoFrame(_ sampleBuffer: CMSampleBuffer) {
        guard isRecording else { return }
        recordQueue.async { [self] in
            guard let writer = assetWriter, let input = videoInput else { return }

            let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer)

            if !sessionStarted {
                writer.startSession(atSourceTime: pts)
                firstFrameTime = pts
                sessionStarted = true
            }

            if input.isReadyForMoreMediaData {
                input.append(sampleBuffer)
            }
        }
    }

    /// Append an IMU sample to the IMU CSV. Safe to call from any queue.
    func appendIMUSample(_ motion: CMDeviceMotion) {
        guard isRecording else { return }
        let elapsed = Date().timeIntervalSince(startTime)
        let row = String(
            format: "%.4f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.4f\n",
            elapsed,
            motion.userAcceleration.x + motion.gravity.x,
            motion.userAcceleration.y + motion.gravity.y,
            motion.userAcceleration.z + motion.gravity.z,
            motion.userAcceleration.x,
            motion.userAcceleration.y,
            motion.userAcceleration.z,
            motion.gravity.x,
            motion.gravity.y,
            motion.gravity.z,
            motion.rotationRate.x,
            motion.rotationRate.y,
            motion.rotationRate.z,
            motion.attitude.roll,
            motion.attitude.pitch,
            motion.attitude.yaw,
            motion.timestamp   // device uptime clock, pairs with capture `pts`/`apex_ts`
        )
        ioQueue.async { [self] in
            if let data = row.data(using: .utf8) {
                imuFileHandle?.write(data)
                imuSampleCount += 1
            }
        }
    }

    /// Append a tagged event to the events log. Safe to call from any queue.
    func logEvent(_ type: String, details: String) {
        guard isRecording else { return }
        let elapsed = Date().timeIntervalSince(startTime)
        // CSV-escape the details field
        let escaped = details.contains(",") || details.contains("\"")
            ? "\"\(details.replacingOccurrences(of: "\"", with: "\"\""))\""
            : details
        let row = String(format: "%.4f,%@,%@\n", elapsed, type, escaped)
        ioQueue.async { [self] in
            if let data = row.data(using: .utf8) {
                eventsFileHandle?.write(data)
                eventCount += 1
            }
        }
    }

    // MARK: - Session Management

    /// Lists all recorded sessions from both local and iCloud sync folder, most recent first.
    func listSessions() -> [SessionInfo] {
        var sessions: [SessionInfo] = []

        // Local
        sessions.append(contentsOf: listSessions(in: localRecordingsURL, isSynced: false))

        // iCloud sync folder (if configured)
        if let syncRoot = FeedbackExporter.shared.configuredSyncFolderURL,
           syncRoot.startAccessingSecurityScopedResource() {
            defer { syncRoot.stopAccessingSecurityScopedResource() }
            let syncRecordings = syncRoot.appendingPathComponent("recordings")
            sessions.append(contentsOf: listSessions(in: syncRecordings, isSynced: true))
        }

        return sessions.sorted { $0.id > $1.id }
    }

    private func listSessions(in dir: URL, isSynced: Bool) -> [SessionInfo] {
        guard let folders = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) else {
            return []
        }
        return folders
            .filter { (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) == true }
            .map { SessionInfo(id: $0.lastPathComponent, url: $0, sizeBytes: Self.folderSize(at: $0), isSynced: isSynced) }
    }

    /// Delete a recorded session by ID (checks both local and sync locations).
    func deleteSession(_ id: String) {
        // Try local
        let localURL = localRecordingsURL.appendingPathComponent(id)
        if FileManager.default.fileExists(atPath: localURL.path) {
            try? FileManager.default.removeItem(at: localURL)
            return
        }
        // Try sync folder
        if let syncRoot = FeedbackExporter.shared.configuredSyncFolderURL,
           syncRoot.startAccessingSecurityScopedResource() {
            defer { syncRoot.stopAccessingSecurityScopedResource() }
            let url = syncRoot.appendingPathComponent("recordings").appendingPathComponent(id)
            try? FileManager.default.removeItem(at: url)
        }
    }

    struct SessionInfo: Identifiable {
        let id: String
        let url: URL
        let sizeBytes: Int64
        let isSynced: Bool
    }

    private static func folderSize(at url: URL) -> Int64 {
        guard let enumerator = FileManager.default.enumerator(at: url, includingPropertiesForKeys: [.fileSizeKey]) else {
            return 0
        }
        var total: Int64 = 0
        for case let fileURL as URL in enumerator {
            let size = (try? fileURL.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            total += Int64(size)
        }
        return total
    }
}
