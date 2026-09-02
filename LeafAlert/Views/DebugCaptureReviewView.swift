import SwiftUI

/// Debug view for reviewing capture timing and image quality.
/// Shows live accelerometer data, capture statistics, and a gallery of saved frames.
struct DebugCaptureReviewView: View {
    @EnvironmentObject private var appState: AppState

    @State private var accelHistory: [Double] = []
    @State private var rotHistory: [Double] = []
    @State private var frames: [DebugFrameSaver.FrameInfo] = []
    @State private var selectedFrame: DebugFrameSaver.FrameInfo?

    private let maxHistorySize = 200

    var body: some View {
        ScrollView {
            VStack(spacing: 0) {
                accelerometerGraph
                statsBar
                frameGallery
            }
        }
        .navigationTitle("Capture Review")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear { loadFrames() }
        .onChange(of: appState.captureEngine.currentNetAccel) { _, newValue in
            accelHistory.append(newValue)
            if accelHistory.count > maxHistorySize {
                accelHistory.removeFirst(accelHistory.count - maxHistorySize)
            }
            rotHistory.append(appState.captureEngine.currentRotationRate)
            if rotHistory.count > maxHistorySize {
                rotHistory.removeFirst(rotHistory.count - maxHistorySize)
            }
        }
        // Refresh the gallery on a bounded cadence instead of on every capture:
        // listing and parsing the whole frame directory on each new frame ran on
        // the main thread at the capture rate, right on top of the capture timer.
        .task {
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 2_000_000_000)
                loadFrames()
            }
        }
        .fullScreenCover(item: $selectedFrame) { frame in
            FrameFullscreenView(frame: frame)
        }
    }

    // MARK: - Live Accelerometer Graph

    private var accelerometerGraph: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text("Motion")
                    .font(.caption.bold())
                Spacer()
                Text(String(format: "Accel %+.3fg", appState.captureEngine.currentNetAccel))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.green)
                Text(String(format: "Rot %.1f°/s", appState.captureEngine.currentRotationRate * 180.0 / .pi))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.cyan)
            }
            .padding(.horizontal)
            .padding(.top, 12)

            Canvas { context, size in
                let w = size.width
                let h = size.height
                let midY = h / 2
                let scale = h / 1.0 // +/-0.5g fills the view

                // Background
                context.fill(Path(CGRect(origin: .zero, size: size)), with: .color(.black.opacity(0.8)))

                // Zero line
                let zeroPath = Path { p in
                    p.move(to: CGPoint(x: 0, y: midY))
                    p.addLine(to: CGPoint(x: w, y: midY))
                }
                context.stroke(zeroPath, with: .color(.gray.opacity(0.4)), style: StrokeStyle(lineWidth: 1, dash: [4, 4]))

                // Noise floor lines at +/-0.03g
                let noiseFloor = 0.03
                for sign in [-1.0, 1.0] {
                    let y = midY - CGFloat(sign * noiseFloor) * scale
                    let noisePath = Path { p in
                        p.move(to: CGPoint(x: 0, y: y))
                        p.addLine(to: CGPoint(x: w, y: y))
                    }
                    context.stroke(noisePath, with: .color(.orange.opacity(0.4)), style: StrokeStyle(lineWidth: 1, dash: [2, 3]))
                }

                // Acceleration line
                guard accelHistory.count > 1 else { return }
                let step = w / CGFloat(maxHistorySize - 1)

                let linePath = Path { p in
                    for (i, val) in accelHistory.enumerated() {
                        let x = w - CGFloat(accelHistory.count - 1 - i) * step
                        let y = midY - CGFloat(val) * scale
                        let clamped = min(max(y, 0), h)
                        if i == 0 { p.move(to: CGPoint(x: x, y: clamped)) }
                        else { p.addLine(to: CGPoint(x: x, y: clamped)) }
                    }
                }
                context.stroke(linePath, with: .color(.green), lineWidth: 1.5)

                // Rotation rate line (scaled: 0 at bottom, ~3 rad/s at top)
                if rotHistory.count > 1 {
                    let rotScale = h / 3.0 // 3 rad/s fills full height
                    let rotPath = Path { p in
                        for (i, val) in rotHistory.enumerated() {
                            let x = w - CGFloat(rotHistory.count - 1 - i) * step
                            let y = h - CGFloat(val) * rotScale
                            let clamped = min(max(y, 0), h)
                            if i == 0 { p.move(to: CGPoint(x: x, y: clamped)) }
                            else { p.addLine(to: CGPoint(x: x, y: clamped)) }
                        }
                    }
                    context.stroke(rotPath, with: .color(.cyan.opacity(0.7)), lineWidth: 1)

                    // Rotation threshold line
                    let threshY = h - CGFloat(1.5) * rotScale // default 1.5 rad/s
                    let threshPath = Path { p in
                        p.move(to: CGPoint(x: 0, y: threshY))
                        p.addLine(to: CGPoint(x: w, y: threshY))
                    }
                    context.stroke(threshPath, with: .color(.red.opacity(0.4)), style: StrokeStyle(lineWidth: 1, dash: [3, 3]))
                }
            }
            .frame(height: 140)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .padding(.horizontal)

            // Legend
            HStack(spacing: 12) {
                HStack(spacing: 4) {
                    Circle().fill(.green).frame(width: 6, height: 6)
                    Text("Accel").font(.caption2)
                }
                HStack(spacing: 4) {
                    Circle().fill(.cyan).frame(width: 6, height: 6)
                    Text("Rotation").font(.caption2)
                }
                HStack(spacing: 4) {
                    RoundedRectangle(cornerRadius: 1).fill(.red.opacity(0.5)).frame(width: 12, height: 2)
                    Text("Rot limit").font(.caption2)
                }
                HStack(spacing: 4) {
                    RoundedRectangle(cornerRadius: 1).fill(.orange.opacity(0.6)).frame(width: 12, height: 2)
                    Text("Noise").font(.caption2)
                }
            }
            .foregroundStyle(.secondary)
            .padding(.horizontal)
            .padding(.bottom, 8)
        }
    }

    // MARK: - Stats Bar

    private var statsBar: some View {
        HStack(spacing: 0) {
            statItem("Apogees", value: "\(appState.captureEngine.apogeeCount)")
            Divider().frame(height: 24)
            statItem("Frames", value: "\(appState.captureEngine.totalFramesCaptured)")
            Divider().frame(height: 24)
            statItem("Cap/min", value: "\(appState.captureEngine.capturesPerMinute)")
            Divider().frame(height: 24)
            statItem("Latency", value: String(format: "%.0fms", appState.inferenceEngine.lastInferenceTime * 1000))
        }
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }

    private func statItem(_ title: String, value: String) -> some View {
        VStack(spacing: 2) {
            Text(value)
                .font(.subheadline.monospacedDigit().bold())
            Text(title)
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
    }

    // MARK: - Frame Gallery

    private var frameGallery: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Saved Frames (\(frames.count))")
                    .font(.caption.bold())
                Spacer()
                if !frames.isEmpty {
                    Button("Clear All", role: .destructive) {
                        DebugFrameSaver.shared.clearAll()
                        frames = []
                    }
                    .font(.caption)
                }
            }
            .padding(.horizontal)
            .padding(.top, 12)

            if frames.isEmpty {
                Text("No frames saved yet. Enable \"Save All Frames\" in Debug settings and start a patrol.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal)
                    .padding(.vertical, 24)
            } else {
                LazyVStack(spacing: 8) {
                    ForEach(frames) { frame in
                        frameCard(frame)
                            .onTapGesture { selectedFrame = frame }
                    }
                }
                .padding(.horizontal)
            }
        }
        .padding(.bottom, 24)
    }

    private func frameCard(_ frame: DebugFrameSaver.FrameInfo) -> some View {
        HStack(spacing: 12) {
            // Lazy-load image from disk
            AsyncFrameImage(url: frame.url)
                .frame(width: 80, height: 60)
                .clipShape(RoundedRectangle(cornerRadius: 6))

            VStack(alignment: .leading, spacing: 2) {
                Text(frame.classification.replacingOccurrences(of: "_", with: " ").capitalized)
                    .font(.subheadline.bold())
                    .foregroundStyle(
                        InferenceEngine.toxicLabels.contains(frame.classification) ? .red : .primary
                    )
                HStack(spacing: 8) {
                    Text("#\(frame.index)")
                    blurBadge(frame.blurScore)
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            }

            Spacer()

            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundStyle(.tertiary)
        }
        .padding(8)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 10))
    }

    @ViewBuilder
    private func blurBadge(_ score: Double) -> some View {
        let (label, color): (String, Color) = {
            if score < 100 { return ("Blurry", .red) }
            if score < 500 { return ("OK", .orange) }
            return ("Sharp", .green)
        }()
        Text("\(label) (\(Int(score)))")
            .font(.caption2.bold())
            .foregroundStyle(color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(color.opacity(0.15))
            .clipShape(Capsule())
    }

    /// Lists frames off the main thread and publishes the result.
    private func loadFrames() {
        Task {
            let list = await Task.detached(priority: .utility) {
                DebugFrameSaver.shared.listFrames()
            }.value
            frames = list
        }
    }
}

// MARK: - Async Frame Image Loader

/// Loads a JPEG from disk on a background thread to avoid blocking the UI.
private struct AsyncFrameImage: View {
    let url: URL
    @State private var image: UIImage?

    var body: some View {
        Group {
            if let image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fill)
            } else {
                Rectangle()
                    .fill(.gray.opacity(0.2))
                    .overlay {
                        ProgressView()
                            .scaleEffect(0.6)
                    }
            }
        }
        .task {
            image = await loadImage()
        }
    }

    private func loadImage() async -> UIImage? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .utility).async {
                let img = UIImage(contentsOfFile: url.path)
                continuation.resume(returning: img)
            }
        }
    }
}

// MARK: - Fullscreen Frame Viewer

/// Displays a captured frame fullscreen with pinch-to-zoom and blur metric overlay.
private struct FrameFullscreenView: View {
    let frame: DebugFrameSaver.FrameInfo
    @Environment(\.dismiss) private var dismiss
    @State private var image: UIImage?
    @State private var scale: CGFloat = 1.0
    @State private var lastScale: CGFloat = 1.0

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if let image {
                Image(uiImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .scaleEffect(scale)
                    .gesture(
                        MagnifyGesture()
                            .onChanged { value in
                                scale = lastScale * value.magnification
                            }
                            .onEnded { _ in
                                lastScale = scale
                            }
                    )
                    .onTapGesture(count: 2) {
                        withAnimation {
                            scale = scale > 1.0 ? 1.0 : 3.0
                            lastScale = scale
                        }
                    }
            } else {
                ProgressView()
                    .tint(.white)
            }

            // Info overlay
            VStack {
                HStack {
                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title2)
                            .foregroundStyle(.white.opacity(0.8))
                    }
                    Spacer()
                }
                .padding()

                Spacer()

                HStack(spacing: 16) {
                    Text("#\(frame.index)")
                        .monospacedDigit()
                    Text(frame.classification.replacingOccurrences(of: "_", with: " ").capitalized)
                        .bold()

                    Spacer()

                    let (label, color): (String, Color) = {
                        if frame.blurScore < 100 { return ("Blurry", .red) }
                        if frame.blurScore < 500 { return ("OK", .orange) }
                        return ("Sharp", .green)
                    }()
                    HStack(spacing: 4) {
                        Circle().fill(color).frame(width: 8, height: 8)
                        Text("\(label) — \(Int(frame.blurScore))")
                            .monospacedDigit()
                    }
                }
                .font(.subheadline)
                .foregroundStyle(.white)
                .padding()
                .background(.black.opacity(0.6))
            }
        }
        .task {
            image = await loadImage()
        }
    }

    private func loadImage() async -> UIImage? {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let img = UIImage(contentsOfFile: frame.url.path)
                continuation.resume(returning: img)
            }
        }
    }
}

#Preview {
    NavigationStack {
        DebugCaptureReviewView()
            .environmentObject(AppState())
    }
}
