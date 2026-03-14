import SwiftUI

/// Live patrol screen. Dims the display and runs continuous detection.
struct PatrolView: View {
    @EnvironmentObject private var appState: AppState
    @State private var isPatrolling = false
    @State private var lastDetection: DetectionResult?
    @State private var showAROverlay = false
    @State private var screenDimOpacity: Double = 0.7

    var body: some View {
        ZStack {
            // Dimmed background during patrol
            Color.black.opacity(isPatrolling ? screenDimOpacity : 0)
                .ignoresSafeArea()

            VStack(spacing: 24) {
                if isPatrolling {
                    Spacer()

                    Image(systemName: "eye.fill")
                        .font(.system(size: 48))
                        .foregroundStyle(.green)

                    Text("Patrolling…")
                        .font(.title2)
                        .foregroundStyle(.white)

                    if let detection = lastDetection {
                        detectionBanner(detection)
                    }

                    Spacer()

                    Button("Stop Patrol") {
                        stopPatrol()
                    }
                    .font(.headline)
                    .padding()
                    .background(.red.opacity(0.8))
                    .foregroundStyle(.white)
                    .clipShape(Capsule())
                    .padding(.bottom, 48)
                } else {
                    Spacer()

                    Text("Ready to patrol")
                        .font(.title2)

                    Button("Start Patrol") {
                        startPatrol()
                    }
                    .font(.headline)
                    .padding()
                    .background(.green)
                    .foregroundStyle(.white)
                    .clipShape(Capsule())

                    Spacer()
                }
            }
        }
        .navigationTitle("Patrol")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showAROverlay) {
            if let detection = lastDetection {
                AROverlayView(detectionResult: detection)
            }
        }
    }

    // MARK: - Patrol Lifecycle

    private func startPatrol() {
        isPatrolling = true
        // TODO: Wire CaptureEngine + InferenceEngine + AlertEngine
    }

    private func stopPatrol() {
        isPatrolling = false
        // TODO: Stop engines
    }

    // MARK: - Detection Banner

    @ViewBuilder
    private func detectionBanner(_ detection: DetectionResult) -> some View {
        VStack(spacing: 8) {
            Text("⚠ \(detection.plantType.replacingOccurrences(of: "_", with: " ").capitalized)")
                .font(.headline)
                .foregroundStyle(.yellow)

            Text("Confidence: \(Int(detection.confidence * 100))%")
                .font(.subheadline)
                .foregroundStyle(.white.opacity(0.8))

            Button("View in AR") {
                showAROverlay = true
            }
            .font(.caption.bold())
            .padding(.horizontal, 16)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial)
            .clipShape(Capsule())
        }
        .padding()
        .background(.black.opacity(0.6))
        .clipShape(RoundedRectangle(cornerRadius: 12))

        disclaimerText
    }

    private var disclaimerText: some View {
        Text("This app assists identification — always verify visually before touching any plant.")
            .font(.caption2)
            .foregroundStyle(.white.opacity(0.5))
            .multilineTextAlignment(.center)
            .padding(.horizontal)
    }
}

#Preview {
    NavigationStack {
        PatrolView()
            .environmentObject(AppState())
    }
}
