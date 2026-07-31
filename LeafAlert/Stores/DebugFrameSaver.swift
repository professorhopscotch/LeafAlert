import Foundation
import UIKit
import Accelerate

/// Saves every captured frame to Documents/debug_frames/ for offline review.
/// Files are named with a sequential counter and optional classification label.
/// Access via Files app → On My iPhone → LeafAlert → debug_frames/
final class DebugFrameSaver {

    static let shared = DebugFrameSaver()

    private let ioQueue = DispatchQueue(label: "com.leafalert.debug-frames", qos: .utility)
    /// Next filename index. Seeded from disk on first use — see `seedIndexIfNeeded`.
    private var frameIndex: Int = 0
    private var didSeedIndex = false
    /// Frame count kept in memory. `frameCount` used to stat the directory on every
    /// read, and the debug dashboard reads it from `body` at ~10 Hz on the main
    /// thread — which delays the main-run-loop capture timer, i.e. opening the
    /// diagnostics degraded the capture it was measuring.
    private let countLock = NSLock()
    private var _frameCount: Int = 0

    private var directoryURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("debug_frames")
    }

    private init() {}

    /// Continues numbering from whatever is already on disk.
    ///
    /// `frameIndex` used to restart at 0 every launch, so a second session
    /// overwrote the first session's frames and produced duplicate indices — which
    /// broke the review gallery's identity and made `capture_metrics.csv` rows
    /// impossible to join back to their images. Must be called on `ioQueue`.
    private func seedIndexIfNeeded() {
        guard !didSeedIndex else { return }
        didSeedIndex = true
        let files = (try? FileManager.default.contentsOfDirectory(atPath: directoryURL.path)) ?? []
        let jpgs = files.filter { $0.hasSuffix(".jpg") }
        let maxIndex = jpgs.compactMap { name -> Int? in
            guard let underscore = name.firstIndex(of: "_") else { return nil }
            return Int(name[name.startIndex..<underscore])
        }.max()
        frameIndex = (maxIndex.map { $0 + 1 }) ?? 0
        setFrameCount(jpgs.count)
    }

    private func setFrameCount(_ n: Int) {
        countLock.lock(); defer { countLock.unlock() }
        _frameCount = n
    }

    /// Saves a JPEG frame to disk with a unique sequential filename.
    ///
    /// Records the model's actual verdict alongside the pixels — predicted class,
    /// confidence and the resulting app severity — because this directory is the
    /// pool the active-learning selector ranks. A frame with no recorded confidence
    /// cannot be scored for informativeness, and collapsing "model saw nothing",
    /// "safe plant" and "toxic but below threshold" into one label throws away
    /// exactly the near-miss signal needed to find the model's misses.
    /// Safe to call from any queue.
    func save(jpegData: Data?,
              detection: DetectionResult?,
              severity: DetectionSeverity?,
              context: CaptureContext? = nil) {
        guard let jpegData else { return }
        ioQueue.async { [self] in
            let dir = directoryURL
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            seedIndexIfNeeded()

            let index = frameIndex
            frameIndex += 1

            // "no_detection" means the classifier put no toxic class above
            // safe_plants — distinct from a toxic class that merely fell short of
            // its alert threshold, which is recorded with its real name.
            let label = detection?.plantType ?? "no_detection"
            let blur = Self.laplacianVariance(jpegData: jpegData)
            let filename = String(format: "%05d_%@_b%.0f.jpg", index, label, blur)
            let url = dir.appendingPathComponent(filename)
            try? jpegData.write(to: url)
            setFrameCount(_frameCount + 1)

            appendMetrics(
                index: index,
                blur: blur,
                netAccel: context?.netAcceleration ?? 0,
                rotRate: context?.rotationRate ?? 0,
                trigger: context?.trigger.rawValue ?? "unknown",
                classification: label,
                confidence: detection?.confidence,
                severity: severity,
                timestamp: context?.timestamp ?? Date()
            )
        }
    }

    // MARK: - Metrics CSV

    private var metricsURL: URL {
        directoryURL.appendingPathComponent("capture_metrics.csv")
    }

    private func appendMetrics(
        index: Int,
        blur: Double,
        netAccel: Double,
        rotRate: Double,
        trigger: String,
        classification: String,
        confidence: Float?,
        severity: DetectionSeverity?,
        timestamp: Date
    ) {
        let url = metricsURL
        let fm = FileManager.default

        // Write header if file doesn't exist
        if !fm.fileExists(atPath: url.path) {
            let header = "frame,timestamp,blur_score,net_accel_g,rotation_rate_rad_s,trigger,classification,confidence,severity,filename\n"
            try? header.data(using: .utf8)?.write(to: url)
        }

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let ts = formatter.string(from: timestamp)

        let confStr = confidence.map { String(format: "%.4f", $0) } ?? ""
        let sevStr: String = {
            switch severity {
            case .alert: return "alert"
            case .uncertain: return "uncertain"
            case .ignore: return "ignore"
            case nil: return ""
            }
        }()
        // Carry the filename so a row can always be joined back to its image, even
        // if the naming scheme changes later.
        let filename = String(format: "%05d_%@_b%.0f.jpg", index, classification, blur)

        let row = "\(index),\(ts),\(String(format: "%.1f", blur)),\(String(format: "%.4f", netAccel)),\(String(format: "%.4f", rotRate)),\(trigger),\(classification),\(confStr),\(sevStr),\(filename)\n"

        if let handle = try? FileHandle(forWritingTo: url) {
            handle.seekToEndOfFile()
            handle.write(row.data(using: .utf8)!)
            handle.closeFile()
        }
    }

    /// Deletes all saved debug frames.
    func clearAll() {
        ioQueue.async { [self] in
            try? FileManager.default.removeItem(at: directoryURL)
            frameIndex = 0
            didSeedIndex = true   // fresh directory: numbering restarts from 0
            setFrameCount(0)
        }
    }

    /// Number of saved frames. O(1) in-memory read — deliberately does NOT touch the
    /// filesystem, because the debug dashboard reads this from `body` at ~10 Hz on
    /// the main thread, where a directory stat over thousands of JPEGs stalls the
    /// main-run-loop capture timer.
    var frameCount: Int {
        countLock.lock(); defer { countLock.unlock() }
        return _frameCount
    }

    /// Populates the in-memory count from disk. Call from `onAppear`, not `body`.
    func refreshFrameCount() {
        ioQueue.async { [self] in
            seedIndexIfNeeded()
            let files = (try? FileManager.default.contentsOfDirectory(atPath: directoryURL.path)) ?? []
            setFrameCount(files.filter { $0.hasSuffix(".jpg") }.count)
        }
    }

    /// Metadata for a single saved debug frame.
    struct FrameInfo: Identifiable {
        let id: Int
        let url: URL
        let index: Int
        let classification: String
        let blurScore: Double
        let filename: String
    }

    /// Lists all saved frames sorted by index (most recent first).
    func listFrames() -> [FrameInfo] {
        let dir = directoryURL
        guard let files = try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: nil
        ) else { return [] }

        return files
            .filter { $0.pathExtension == "jpg" }
            .compactMap { url -> FrameInfo? in
                let name = url.deletingPathExtension().lastPathComponent
                // Format: 00001_poison_ivy_b450.jpg → index=1, classification=poison_ivy, blur=450
                // Legacy format: 00001_poison_ivy.jpg → blur=0
                guard let underscoreIdx = name.firstIndex(of: "_") else { return nil }
                let idxStr = name[name.startIndex..<underscoreIdx]
                guard let idx = Int(idxStr) else { return nil }

                let rest = String(name[name.index(after: underscoreIdx)...])

                // Try to parse blur score from _bNNN suffix
                var classification = rest
                var blur: Double = 0
                if let blurRange = rest.range(of: "_b", options: .backwards) {
                    let blurStr = rest[rest.index(blurRange.upperBound, offsetBy: 0)...]
                    if let blurVal = Double(blurStr) {
                        blur = blurVal
                        classification = String(rest[rest.startIndex..<blurRange.lowerBound])
                    }
                }

                return FrameInfo(
                    id: idx,
                    url: url,
                    index: idx,
                    classification: classification,
                    blurScore: blur,
                    filename: url.lastPathComponent
                )
            }
            .sorted { $0.index > $1.index }
    }

    // MARK: - Blur Metric

    /// Computes the variance of the Laplacian of an image — a standard motion blur metric.
    /// Higher values = sharper image. Typical ranges: <100 = very blurry, 100-500 = moderate, >500 = sharp.
    static func laplacianVariance(jpegData: Data) -> Double {
        guard let uiImage = UIImage(data: jpegData),
              let cgImage = uiImage.cgImage else { return 0 }

        let width = cgImage.width
        let height = cgImage.height
        let count = width * height

        // Convert to grayscale float buffer
        guard let cfData = cgImage.dataProvider?.data,
              let ptr = CFDataGetBytePtr(cfData) else { return 0 }

        let bytesPerPixel = cgImage.bitsPerPixel / 8
        var grayscale = [Float](repeating: 0, count: count)

        for i in 0..<count {
            let offset = i * bytesPerPixel
            let r = Float(ptr[offset])
            let g = Float(ptr[offset + 1])
            let b = Float(ptr[offset + 2])
            grayscale[i] = 0.299 * r + 0.587 * g + 0.114 * b
        }

        // Apply 3x3 Laplacian kernel: [0,1,0; 1,-4,1; 0,1,0]
        var laplacian = [Float](repeating: 0, count: count)
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let idx = y * width + x
                let val = -4.0 * grayscale[idx]
                    + grayscale[idx - 1] + grayscale[idx + 1]
                    + grayscale[idx - width] + grayscale[idx + width]
                laplacian[idx] = val
            }
        }

        // Compute variance
        var mean: Float = 0
        var meanSq: Float = 0
        let n = Float((width - 2) * (height - 2))
        for y in 1..<(height - 1) {
            for x in 1..<(width - 1) {
                let val = laplacian[y * width + x]
                mean += val
                meanSq += val * val
            }
        }
        mean /= n
        meanSq /= n
        return Double(meanSq - mean * mean)
    }
}
