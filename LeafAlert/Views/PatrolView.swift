import SwiftUI

/// Live patrol screen. Dims the display and runs continuous detection.
struct PatrolView: View {
    @EnvironmentObject private var appState: AppState
    @State private var showAROverlay = false
    @State private var savedBrightness: CGFloat?
    @State private var showingCorrection = false
    @State private var selectedCorrection: String?

    /// All possible plant labels for the correction picker.
    private static let allLabels: [(key: String, display: String)] = [
        ("poison_ivy", "Poison Ivy"),
        ("poison_oak", "Poison Oak"),
        ("poison_sumac", "Poison Sumac"),
        ("safe_plants", "Safe Plant"),
        ("not_a_plant", "Not a Plant"),
    ]

    var body: some View {
        ZStack {
            if appState.isPatrolling {
                // Live camera preview
                CameraPreviewView(session: appState.captureEngine.session)
                    .ignoresSafeArea()

                // Bounding box overlay when detection has a valid region
                if let detection = appState.lastDetection,
                   detection.boundingBox != .zero {
                    BoundingBoxOverlay(
                        boundingBox: detection.boundingBox,
                        label: DetectionFormatting.plantDisplayName(detection.plantType),
                        confidence: detection.confidence
                    )
                    .ignoresSafeArea()
                }

                // Top status bar
                VStack {
                    HStack(spacing: 6) {
                        Circle()
                            .fill(appState.captureEngine.pipelineActive ? .green : .red)
                            .frame(width: 8, height: 8)
                        Text(appState.captureEngine.pipelineActive ? "Patrolling" : "Camera paused")
                            .font(.caption2.bold())
                            .foregroundStyle(.white)
                    }
                    .padding(.horizontal, 12)
                    .padding(.vertical, 6)
                    .background(.black.opacity(0.5))
                    .clipShape(Capsule())
                    .padding(.top, 4)

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
                }
            } else {
                Color.black.ignoresSafeArea()

                VStack(spacing: 24) {
                    Spacer()

                    Text("Ready to patrol")
                        .font(.title2)
                        .foregroundStyle(.white)

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

            // Feedback card pinned to bottom
            if let detection = appState.lastDetection {
                VStack {
                    Spacer()
                    feedbackCard(detection)
                        .padding(.horizontal, 16)
                        .padding(.bottom, 100)
                        .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                .animation(.easeInOut(duration: 0.3), value: appState.lastDetection?.id)
            }
        }
        .navigationTitle("Patrol")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showAROverlay, onDismiss: {
            // Resume the capture session after AR releases the camera.
            if appState.isPatrolling {
                appState.captureEngine.start()
            }
        }) {
            if let detection = appState.lastDetection {
                AROverlayView(detectionResult: detection)
            }
        }
        .onChange(of: showAROverlay) { _, isShowing in
            if isShowing {
                // Pause the capture session so ARKit can claim the camera.
                appState.captureEngine.stop()
            }
        }
        .onDisappear {
            stopPatrol()
        }
    }

    // MARK: - Patrol Lifecycle

    private func startPatrol() {
        savedBrightness = UIScreen.main.brightness
        UIScreen.main.brightness = CGFloat(1.0 - appState.screenDimLevel)
        appState.startPatrol()
    }

    private func stopPatrol() {
        guard appState.isPatrolling else { return }
        if let brightness = savedBrightness {
            UIScreen.main.brightness = brightness
            savedBrightness = nil
        }
        appState.stopPatrol()
    }

    // MARK: - Feedback Card

    @ViewBuilder
    private func feedbackCard(_ detection: DetectionResult) -> some View {
        VStack(spacing: 12) {
            // Header row: plant name + dismiss button
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text("\u{26A0} \(DetectionFormatting.plantDisplayName(detection.plantType))")
                        .font(.headline)
                        .foregroundStyle(.yellow)
                    Text("Confidence: \(Int(detection.confidence * 100))%")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.8))
                }

                Spacer()

                // Dismiss button (top-right corner X)
                Button {
                    dismissCard()
                } label: {
                    Image(systemName: "xmark")
                        .font(.caption.bold())
                        .foregroundStyle(.white.opacity(0.7))
                        .padding(6)
                        .background(Circle().fill(.white.opacity(0.15)))
                }
            }

            // Action buttons row
            HStack(spacing: 16) {
                // Correct button
                Button {
                    confirmDetection(detection)
                } label: {
                    Label("Correct", systemImage: "checkmark.circle.fill")
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(.green)
                        .clipShape(Capsule())
                }

                // Wrong button
                Button {
                    withAnimation(.easeInOut(duration: 0.25)) {
                        showingCorrection.toggle()
                        if !showingCorrection {
                            selectedCorrection = nil
                        }
                    }
                } label: {
                    Label("Wrong", systemImage: "xmark.circle.fill")
                        .font(.subheadline.bold())
                        .foregroundStyle(.white)
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(.red)
                        .clipShape(Capsule())
                }

                Spacer()

                // View in AR
                Button {
                    showAROverlay = true
                } label: {
                    Image(systemName: "arkit")
                        .font(.subheadline)
                        .foregroundStyle(.white.opacity(0.8))
                        .padding(10)
                        .background(.white.opacity(0.15))
                        .clipShape(Circle())
                }
            }

            // Correction picker (expands when "Wrong" is tapped)
            if showingCorrection {
                correctionPicker(detection)
                    .transition(.opacity.combined(with: .scale(scale: 0.95, anchor: .top)))
            }

            disclaimerText
        }
        .padding(16)
        .background(.ultraThinMaterial)
        .clipShape(RoundedRectangle(cornerRadius: 16))
    }

    // MARK: - Correction Picker

    @ViewBuilder
    private func correctionPicker(_ detection: DetectionResult) -> some View {
        let options = Self.allLabels.filter { $0.key != detection.plantType }

        VStack(alignment: .leading, spacing: 8) {
            Text("What is it?")
                .font(.caption.bold())
                .foregroundStyle(.white.opacity(0.7))

            ForEach(options, id: \.key) { option in
                Button {
                    selectedCorrection = option.key
                } label: {
                    HStack(spacing: 10) {
                        Image(systemName: selectedCorrection == option.key
                              ? "largecircle.fill.circle"
                              : "circle")
                            .foregroundStyle(selectedCorrection == option.key ? .green : .white.opacity(0.5))
                            .font(.body)

                        Text(option.display)
                            .font(.subheadline)
                            .foregroundStyle(.white)

                        Spacer()
                    }
                    .padding(.vertical, 4)
                }
            }

            // Submit correction button
            Button {
                submitCorrection(detection)
            } label: {
                Text("Submit")
                    .font(.subheadline.bold())
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 10)
                    .background(selectedCorrection != nil ? .green : .gray)
                    .clipShape(Capsule())
            }
            .disabled(selectedCorrection == nil)
        }
        .padding(.top, 4)
    }

    // MARK: - Actions

    private func confirmDetection(_ detection: DetectionResult) {
        appState.detectionLogStore.submitFeedback(
            logID: detection.id,
            status: "confirmed",
            correctedLabel: nil
        )
        resetFeedbackState()
    }

    private func submitCorrection(_ detection: DetectionResult) {
        guard let corrected = selectedCorrection else { return }
        appState.detectionLogStore.submitFeedback(
            logID: detection.id,
            status: "corrected",
            correctedLabel: corrected
        )
        resetFeedbackState()
    }

    private func dismissCard() {
        resetFeedbackState()
    }

    private func resetFeedbackState() {
        withAnimation(.easeInOut(duration: 0.25)) {
            appState.lastDetection = nil
        }
        showingCorrection = false
        selectedCorrection = nil
    }

    // MARK: - Disclaimer

    private var disclaimerText: some View {
        Text(DetectionFormatting.safetyDisclaimer)
            .font(.caption2)
            .foregroundStyle(.white.opacity(0.5))
            .multilineTextAlignment(.center)
    }
}

#Preview {
    NavigationStack {
        PatrolView()
            .environmentObject(AppState())
    }
}
