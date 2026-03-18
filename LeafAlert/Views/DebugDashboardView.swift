import SwiftUI

/// Debug dashboard with live engine diagnostics, ML model output, and tunable controls.
/// Only accessible in DEBUG builds via a tab on the home screen.
struct DebugDashboardView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        NavigationStack {
            List {
                engineDiagnosticsSection
                mlOutputSection
                controlsSection
                actionsSection
            }
            .navigationTitle("Debug")
            .navigationBarTitleDisplayMode(.inline)
        }
    }

    // MARK: - Section 1: Engine Diagnostics

    private var engineDiagnosticsSection: some View {
        Section("Engine Diagnostics") {
            // Pipeline status
            HStack {
                statusDot(appState.captureEngine.pipelineActive)
                Text("Pipeline")
                Spacer()
                Text(appState.captureEngine.pipelineActive ? "Active" : "Idle")
                    .foregroundStyle(.secondary)
            }

            // Camera duty cycle
            HStack {
                statusDot(appState.captureEngine.isCameraActive)
                Text("Camera Sensor")
                Spacer()
                Text(appState.captureEngine.isCameraActive ? "ON" : "OFF")
                    .foregroundStyle(appState.captureEngine.isCameraActive ? .green : .secondary)
                    .fontWeight(.medium)
            }

            // Accelerometer
            HStack {
                Text("Net Accel")
                Spacer()
                accelBar(appState.captureEngine.currentNetAccel)
                Text(String(format: "%+.3fg", appState.captureEngine.currentNetAccel))
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
                    .frame(width: 80, alignment: .trailing)
            }

            // Apogee events
            HStack {
                Text("Apogee Events")
                Spacer()
                Text("\(appState.captureEngine.apogeeCount)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }

            // Frames captured
            HStack {
                Text("Frames Captured")
                Spacer()
                Text("\(appState.captureEngine.totalFramesCaptured)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }

            // Captures per minute
            HStack {
                Text("Captures/min")
                Spacer()
                Text("\(appState.captureEngine.capturesPerMinute)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }

            // Inference latency
            HStack {
                Text("Inference Latency")
                Spacer()
                let ms = appState.inferenceEngine.lastInferenceTime * 1000
                Text(String(format: "%.0f ms", ms))
                    .monospacedDigit()
                    .foregroundStyle(ms > 100 ? .orange : .secondary)
            }

            // Total inferences
            HStack {
                Text("Total Inferences")
                Spacer()
                Text("\(appState.inferenceEngine.totalInferences)")
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }

            // Model ready
            HStack {
                statusDot(appState.inferenceEngine.isReady)
                Text("ML Model")
                Spacer()
                Text(appState.inferenceEngine.isReady ? "Loaded" : "Not Loaded")
                    .foregroundStyle(.secondary)
            }
        }
    }

    // MARK: - Section 2: ML Model Output

    private var mlOutputSection: some View {
        Section("ML Model Output") {
            let confidences = appState.inferenceEngine.lastClassConfidences
            let threshold = Float(appState.sensitivityThreshold)

            if confidences.isEmpty {
                Text("No inference results yet")
                    .foregroundStyle(.secondary)
                    .italic()
            } else {
                // Sort: toxic classes first, then safe
                let sorted = confidences.sorted { a, b in
                    let aIsToxic = InferenceEngine.toxicLabels.contains(a.key)
                    let bIsToxic = InferenceEngine.toxicLabels.contains(b.key)
                    if aIsToxic != bIsToxic { return aIsToxic }
                    return a.value > b.value
                }

                ForEach(sorted, id: \.key) { label, confidence in
                    let isToxic = InferenceEngine.toxicLabels.contains(label)
                    let wouldAlert = isToxic && confidence >= threshold &&
                        confidence > (confidences["safe_plants"] ?? 0)

                    HStack(spacing: 8) {
                        // Alert indicator
                        if wouldAlert {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                                .font(.caption)
                        }

                        Text(DetectionFormatting.plantDisplayName(label))
                            .font(.subheadline)
                            .fontWeight(wouldAlert ? .bold : .regular)

                        Spacer()

                        // Confidence bar
                        confidenceBar(confidence, isToxic: isToxic, wouldAlert: wouldAlert)

                        Text("\(Int(confidence * 100))%")
                            .monospacedDigit()
                            .font(.subheadline)
                            .foregroundStyle(wouldAlert ? .red : .secondary)
                            .frame(width: 40, alignment: .trailing)
                    }
                }

                // Threshold line indicator
                HStack {
                    Image(systemName: "line.horizontal.3.decrease")
                        .foregroundStyle(.orange)
                    Text("Alert threshold")
                    Spacer()
                    Text("\(Int(threshold * 100))%")
                        .monospacedDigit()
                        .foregroundStyle(.orange)
                }
                .font(.caption)
            }
        }
    }

    // MARK: - Section 3: Live Controls

    private var controlsSection: some View {
        Section("Live Controls") {
            // Sensitivity threshold
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Sensitivity Threshold")
                    Spacer()
                    Text("\(Int(appState.sensitivityThreshold * 100))%")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Slider(value: $appState.sensitivityThreshold, in: 0.1...0.95, step: 0.05)
                    .tint(.orange)
            }

            // Capture cooldown
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Capture Cooldown")
                    Spacer()
                    Text(String(format: "%.1fs", appState.captureEngine.minCaptureInterval))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Slider(
                    value: Binding(
                        get: { appState.captureEngine.minCaptureInterval },
                        set: { appState.captureEngine.minCaptureInterval = $0 }
                    ),
                    in: 0.3...5.0, step: 0.1
                )
            }

            // Forced capture interval
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Forced Capture Interval")
                    Spacer()
                    Text(String(format: "%.0fs", appState.captureEngine.forcedCaptureInterval))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Slider(
                    value: Binding(
                        get: { appState.captureEngine.forcedCaptureInterval },
                        set: { appState.captureEngine.forcedCaptureInterval = $0 }
                    ),
                    in: 2.0...15.0, step: 1.0
                )
            }

            // Stillness threshold
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Stillness Threshold")
                    Spacer()
                    Text(String(format: "%.3fg", appState.captureEngine.stillnessThreshold))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Slider(
                    value: Binding(
                        get: { appState.captureEngine.stillnessThreshold },
                        set: { appState.captureEngine.stillnessThreshold = $0 }
                    ),
                    in: 0.01...0.3, step: 0.01
                )
            }

            // Battery saver toggle
            Toggle("Battery Saver", isOn: $appState.batterySaverEnabled)

            // Audio alerts toggle
            Toggle("Audio Alerts", isOn: $appState.audioAlertsEnabled)
        }
    }

    // MARK: - Section 4: Actions

    private var actionsSection: some View {
        Section("Actions") {
            Button {
                if appState.isPatrolling {
                    appState.stopPatrol()
                } else {
                    appState.startPatrol()
                }
            } label: {
                Label(
                    appState.isPatrolling ? "Stop Patrol" : "Start Patrol",
                    systemImage: appState.isPatrolling ? "stop.circle.fill" : "play.circle.fill"
                )
                .foregroundStyle(appState.isPatrolling ? .red : .green)
            }

            Button {
                // Force a capture by setting a very short forced interval temporarily
                let original = appState.captureEngine.forcedCaptureInterval
                appState.captureEngine.forcedCaptureInterval = 0.01
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    appState.captureEngine.forcedCaptureInterval = original
                }
            } label: {
                Label("Force Capture Now", systemImage: "camera.shutter.button.fill")
            }
            .disabled(!appState.isPatrolling)

            Button {
                appState.captureEngine.apogeeCount = 0
                appState.captureEngine.totalFramesCaptured = 0
            } label: {
                Label("Reset Counters", systemImage: "arrow.counterclockwise")
            }
        }
    }

    // MARK: - Helpers

    @ViewBuilder
    private func statusDot(_ isActive: Bool) -> some View {
        Circle()
            .fill(isActive ? .green : .red)
            .frame(width: 8, height: 8)
    }

    @ViewBuilder
    private func accelBar(_ value: Double) -> some View {
        // Mini horizontal bar showing acceleration magnitude
        let clamped = min(max(value, -0.5), 0.5)
        let normalized = (clamped + 0.5)  // 0..1 range, 0.5 = zero
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 2)
                    .fill(.gray.opacity(0.2))
                // Center marker
                Rectangle()
                    .fill(.gray.opacity(0.4))
                    .frame(width: 1)
                    .offset(x: geo.size.width / 2)
                // Value bar
                RoundedRectangle(cornerRadius: 2)
                    .fill(abs(value) < 0.08 ? .green : .orange)
                    .frame(
                        width: abs(CGFloat(clamped)) * geo.size.width,
                        height: geo.size.height
                    )
                    .offset(x: clamped >= 0
                        ? geo.size.width / 2
                        : geo.size.width / 2 - abs(CGFloat(clamped)) * geo.size.width
                    )
            }
        }
        .frame(width: 60, height: 12)
    }

    @ViewBuilder
    private func confidenceBar(_ confidence: Float, isToxic: Bool, wouldAlert: Bool) -> some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 3)
                    .fill(.gray.opacity(0.15))
                RoundedRectangle(cornerRadius: 3)
                    .fill(wouldAlert ? .red : (isToxic ? .orange : .green))
                    .frame(width: geo.size.width * CGFloat(confidence))
            }
        }
        .frame(width: 80, height: 14)
    }
}

#Preview {
    DebugDashboardView()
        .environmentObject(AppState())
}
