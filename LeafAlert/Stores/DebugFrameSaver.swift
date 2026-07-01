import Foundation
import UIKit
import Accelerate

/// Saves every captured frame to Documents/debug_frames/ for offline review.
/// Files are named with a sequential counter and optional classification label.
/// Access via Files app → On My iPhone → LeafAlert → debug_frames/
final class DebugFrameSaver {

    static let shared = DebugFrameSaver()

    private let ioQueue = DispatchQueue(label: "com.leafalert.debug-frames", qos: .utility)
    private var frameIndex: Int = 0

    private var directoryURL: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        return docs.appendingPathComponent("debug_frames")
    }

    private init() {}

    /// Saves a JPEG frame to disk with a sequential filename.
    /// Computes Laplacian variance (blur metric) and embeds it in the filename.
    /// Appends motion + blur metrics to capture_metrics.csv for calibration.
    /// Safe to call from any queue.
    func save(jpegData: Data?, classification: String?, context: CaptureContext? = nil) {
        guard let jpegData else { return }
        ioQueue.async { [self] in
            let dir = directoryURL
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

            let index = frameIndex
            frameIndex += 1

            let label = classification ?? "none"
            let blur = Self.laplacianVariance(jpegData: jpegData)
            let filename = String(format: "%05d_%@_b%.0f.jpg", index, label, blur)
            let url = dir.appendingPathComponent(filename)
            try? jpegData.write(to: url)

            // Append metrics row
            if let ctx = context {
                appendMetrics(
                    index: index,
                    blur: blur,
                    netAccel: ctx.netAcceleration,
                    rotRate: ctx.rotationRate,
                    trigger: ctx.trigger.rawValue,
                    classification: label,
                    timestamp: ctx.timestamp
                )
            }
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
        timestamp: Date
    ) {
        let url = metricsURL
        let fm = FileManager.default

        // Write header if file doesn't exist
        if !fm.fileExists(atPath: url.path) {
            let header = "frame,timestamp,blur_score,net_accel_g,rotation_rate_rad_s,trigger,classification\n"
            try? header.data(using: .utf8)?.write(to: url)
        }

        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let ts = formatter.string(from: timestamp)

        let row = "\(index),\(ts),\(String(format: "%.1f", blur)),\(String(format: "%.4f", netAccel)),\(String(format: "%.4f", rotRate)),\(trigger),\(classification)\n"

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
        }
    }

    /// Returns the number of saved frames.
    var frameCount: Int {
        let dir = directoryURL
        return (try? FileManager.default.contentsOfDirectory(atPath: dir.path))?.count ?? 0
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
