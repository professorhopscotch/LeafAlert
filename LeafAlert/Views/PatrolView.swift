import SwiftUI

/// Live patrol screen. Dims the display and runs continuous detection.
struct PatrolView: View {
    @EnvironmentObject private var appState: AppState
    @Environment(\.scenePhase) private var scenePhase
    @State private var showAROverlay = false
    /// Snapshot of the detection the AR sheet was opened for. `lastDetection`
    /// expires after a couple of seconds (its box is only valid briefly), which
    /// would otherwise blank the AR sheet while the user is still looking at it.
    @State private var arDetection: DetectionResult?
    @State private var savedBrightness: CGFloat?
    @State private var showingCorrection = false
    @State private var selectedCorrection: String?
    @AppStorage("livePreviewEnabled") private var livePreviewEnabled = true
    @State private var showPreviewToggleConfirm = false
    @State private var showCaptureFlash = false
    /// Shares the preview layer with BoundingBoxOverlay so it can map Vision
    /// coordinates through AVFoundation's gravity/orientation math.
    @StateObject private var previewLayerBox = PreviewLayerBox()

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
                if livePreviewEnabled {
                    if appState.captureEngine.cameraAvailable {
                        // Live camera preview
                        CameraPreviewView(session: appState.captureEngine.session,
                                          layerBox: previewLayerBox)
                            .ignoresSafeArea()
                    } else {
                        // No camera came up (in use by another app, a hardware
                        // fault, or the simulator). Say so instead of showing a
                        // dead preview — detection is NOT running.
                        Color.black.ignoresSafeArea()
                        VStack(spacing: 12) {
                            Image(systemName: "camera.slash")
                                .font(.system(size: 48))
                                .foregroundStyle(.orange)
                            Text("Camera unavailable")
                                .font(.title3.bold())
                                .foregroundStyle(.white)
                            Text("Detection is paused. Close other apps using the camera, then stop and restart the patrol.")
                                .font(.subheadline)
                                .foregroundStyle(.white.opacity(0.8))
                                .multilineTextAlignment(.center)
                                .padding(.horizontal, 32)
                        }
                    }

                    // Bounding box overlay when detection has a valid region
                    if let detection = appState.boxDetection,
                       detection.boundingBox != .zero {
                        BoundingBoxOverlay(
                            boundingBox: detection.boundingBox,
                            label: DetectionFormatting.plantDisplayName(detection.plantType),
                            confidence: detection.confidence,
                            layerBox: previewLayerBox
                        )
                        .ignoresSafeArea()
                    }
                } else {
                    // Dimmed background when live preview is off
                    Color.black.opacity(appState.screenDimLevel)
                        .ignoresSafeArea()

                    VStack {
                        Spacer()
                        Image(systemName: "eye.fill")
                            .font(.system(size: 48))
                            .foregroundStyle(.green)
                        Text("Patrolling\u{2026}")
                            .font(.title2)
                            .foregroundStyle(.white)
                        Spacer()
                    }
                }

                // Top status bar
                VStack {
                    HStack(spacing: 8) {
                        HStack(spacing: 6) {
                            let engine = appState.captureEngine
                            let healthy = engine.pipelineActive && engine.cameraAvailable
                            Circle()
                                .fill(healthy ? .green : .red)
                                .frame(width: 8, height: 8)
                            Text(!engine.cameraAvailable ? "No camera"
                                 : (engine.pipelineActive ? "Patrolling" : "Camera paused"))
                                .font(.caption2.bold())
                                .foregroundStyle(.white)
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 4)
                        .background(.black.opacity(0.5))
                        .clipShape(Capsule())

                        if DataRecorder.shared.isRecording {
                            HStack(spacing: 4) {
                                Circle().fill(.red).frame(width: 8, height: 8)
                                Text("REC")
                                    .font(.caption2.bold())
                                    .foregroundStyle(.white)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 4)
                            .background(.black.opacity(0.5))
                            .clipShape(Capsule())
                        }

                        // Long-press to toggle live preview
                        Image(systemName: livePreviewEnabled ? "camera.fill" : "camera")
                            .font(.caption)
                            .foregroundStyle(.white.opacity(0.7))
                            .padding(8)
                            .background(.black.opacity(0.5))
                            .clipShape(Circle())
                            .onLongPressGesture(minimumDuration: 0.8) {
                                livePreviewEnabled.toggle()
                                showPreviewToggleConfirm = true
                                let generator = UIImpactFeedbackGenerator(style: .medium)
                                generator.impactOccurred()
                                DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                                    withAnimation { showPreviewToggleConfirm = false }
                                }
                            }
                    }
                    .padding(.top, 4)

                    if showPreviewToggleConfirm {
                        Text(livePreviewEnabled ? "Live preview on" : "Live preview off")
                            .font(.caption2.bold())
                            .foregroundStyle(.white)
                            .padding(.horizontal, 12)
                            .padding(.vertical, 4)
                            .background(.black.opacity(0.6))
                            .clipShape(Capsule())
                            .transition(.opacity)
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

            // Green edge flash on capture (only when debug frame saving is on)
            if showCaptureFlash && appState.debugSaveFrames {
                RoundedRectangle(cornerRadius: 0)
                    .stroke(.green, lineWidth: 6)
                    .ignoresSafeArea()
                    .allowsHitTesting(false)
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
        // The correction picker belongs to ONE detection. If a newer detection
        // replaces the card (or it expires) while the picker is open, a Submit
        // would file the correction against the wrong detection.
        .onChange(of: appState.lastDetection?.id) { _, _ in
            showingCorrection = false
            selectedCorrection = nil
        }
        .sheet(isPresented: $showAROverlay, onDismiss: {
            // Resume the capture session after AR releases the camera.
            if appState.isPatrolling {
                appState.captureEngine.start()
            }
        }) {
            if let detection = arDetection {
                // The sheet is its own presentation context, outside Home's stack,
                // so it needs a stack of its own for "Learn about this plant" to push.
                NavigationStack {
                    AROverlayView(detectionResult: detection)
                }
            }
        }
        .onChange(of: showAROverlay) { _, isShowing in
            if isShowing {
                // Pause the capture session so ARKit can claim the camera.
                appState.captureEngine.stop()
            }
        }
        .onChange(of: appState.captureEngine.totalFramesCaptured) { _, _ in
            guard appState.debugSaveFrames else { return }
            withAnimation(.easeIn(duration: 0.1)) { showCaptureFlash = true }
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.3) {
                withAnimation(.easeOut(duration: 0.25)) { showCaptureFlash = false }
            }
        }
        .onChange(of: scenePhase) { _, phase in
            // Keep the dim + keep-awake only while on screen. When the app leaves
            // the foreground, restore the user's real brightness and let the
            // device sleep normally; re-apply on return so a backgrounded patrol
            // doesn't leak a dimmed screen or drain the battery locked-on.
            guard appState.isPatrolling else { return }
            switch phase {
            case .active:
                if savedBrightness != nil {
                    UIScreen.main.brightness = CGFloat(1.0 - appState.screenDimLevel)
                }
                UIApplication.shared.isIdleTimerDisabled = true
            case .inactive, .background:
                if let brightness = savedBrightness {
                    UIScreen.main.brightness = brightness
                }
                UIApplication.shared.isIdleTimerDisabled = false
                // A recording cannot continue in the background (the camera is
                // interrupted), so finalize it now rather than leave an
                // unplayable file. Only on .background — .inactive also fires
                // for transient overlays like a notification banner.
                if phase == .background, DataRecorder.shared.isRecording {
                    DataRecorder.shared.stop()
                }
            @unknown default:
                break
            }
        }
        #if DEBUG
        // Test hook: `-autoStartPatrol 1` starts the patrol on appear so the
        // simulator smoke test can exercise the live patrol UI (including the
        // no-camera degradation path) without a tap.
        .onAppear {
            // Defer past the current SwiftUI update: starting the patrol publishes
            // engine/app state, and publishing from inside onAppear's update pass
            // is undefined behavior (it rendered a blank screen in the simulator).
            guard UserDefaults.standard.bool(forKey: "autoStartPatrol"), !appState.isPatrolling else { return }
            DispatchQueue.main.async { startPatrol() }
            #if DEBUG
            // `-injectDetection poison_ivy:0.72` pushes a synthetic detection
            // through the live path a second later, so the card/box/feedback UX
            // can be exercised and UI-tested without a camera.
            if let spec = UserDefaults.standard.string(forKey: "injectDetection") {
                let parts = spec.split(separator: ":")
                let plant = String(parts.first ?? "poison_ivy")
                let conf = parts.count > 1 ? (Float(parts[1]) ?? 0.7) : 0.7
                DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) {
                    appState.injectDebugDetection(plantType: plant, confidence: conf)
                }
            }
            #endif
        }
        #endif
        .onDisappear {
            // Only stop patrol when truly navigating away, not when a sheet is presented.
            // Sheets (like AR overlay) trigger onDisappear but the user hasn't left patrol.
            if !showAROverlay {
                stopPatrol()
            }
        }
    }

    // MARK: - Patrol Lifecycle

    private func startPatrol() {
        savedBrightness = UIScreen.main.brightness
        UIScreen.main.brightness = CGFloat(1.0 - appState.screenDimLevel)
        // Prevent auto-lock from killing an active patrol.
        UIApplication.shared.isIdleTimerDisabled = true
        appState.startPatrol()
    }

    private func stopPatrol() {
        guard appState.isPatrolling else { return }
        if let brightness = savedBrightness {
            UIScreen.main.brightness = brightness
            savedBrightness = nil
        }
        UIApplication.shared.isIdleTimerDisabled = false
        appState.stopPatrol()
    }

    // MARK: - Feedback Card

    @ViewBuilder
    private func feedbackCard(_ detection: DetectionResult) -> some View {
        VStack(spacing: 12) {
            // Header row: plant name + dismiss button
            HStack {
                let severity = ToxicityThresholds.severity(
                    plantType: detection.plantType,
                    confidence: detection.confidence,
                    sensitivity: Float(appState.sensitivityThreshold)
                )
                VStack(alignment: .leading, spacing: 4) {
                    Text("\u{26A0} \(DetectionFormatting.detectionHeadline(detection.plantType, severity: severity))")
                        .font(.headline)
                        .foregroundStyle(severity == .alert ? .orange : .yellow)
                    Text(DetectionFormatting.detectionSubtitle(confidence: detection.confidence, severity: severity))
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
                    arDetection = detection
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
