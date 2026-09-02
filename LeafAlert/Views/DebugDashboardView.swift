import SwiftUI
import AVFoundation

/// Debug dashboard with live engine diagnostics, ML model output, and tunable controls.
/// Only accessible in DEBUG builds via a tab on the home screen.
struct DebugDashboardView: View {
    @EnvironmentObject private var appState: AppState

    // No NavigationStack of its own: this view is always PUSHED from Home's
    // stack, and a nested stack inside a navigationDestination makes SwiftUI
    // drop the push — routing to Debug silently landed on Home.
    var body: some View {
        List {
            engineDiagnosticsSection
            mlOutputSection
            controlsSection
            frameCaptureSection
            dataRecordingSection
            actionsSection
        }
        .navigationTitle("Debug")
        .navigationBarTitleDisplayMode(.inline)
        // Refresh the cached count here rather than from `body`: reading it is
        // now an O(1) in-memory load, and the directory stat happens once on
        // appear instead of at the ~10 Hz rate `body` re-evaluates.
        .onAppear { DebugFrameSaver.shared.refreshFrameCount() }
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

            // Rotation rate
            HStack {
                Text("Rotation Rate")
                Spacer()
                let degPerSec = appState.captureEngine.currentRotationRate * 180.0 / .pi
                Text(String(format: "%.0f°/s", degPerSec))
                    .monospacedDigit()
                    .foregroundStyle(degPerSec > 86 ? .red : .secondary)
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
                    // Ask the SAME code the app ships with. This panel previously
                    // reimplemented a single global-threshold rule, which stopped
                    // matching reality when per-class thresholds and the uncertainty
                    // band landed — so the one screen you open to ask "why didn't it
                    // warn me?" reported the wrong verdict in both directions.
                    let severity: DetectionSeverity = isToxic
                        ? ToxicityThresholds.severity(plantType: label,
                                                      confidence: confidence,
                                                      sensitivity: threshold)
                        : .ignore
                    let beatsSafe = confidence > (confidences["safe_plants"] ?? 0)
                    let wouldAlert = severity == .alert && beatsSafe
                    let wouldSurface = severity.isActionable && beatsSafe

                    HStack(spacing: 8) {
                        // Alert indicator: filled for a full alert, hollow for the
                        // "verify visually" band, so both are visible here.
                        if wouldAlert {
                            Image(systemName: "exclamationmark.triangle.fill")
                                .foregroundStyle(.red)
                                .font(.caption)
                        } else if wouldSurface {
                            Image(systemName: "exclamationmark.triangle")
                                .foregroundStyle(.yellow)
                                .font(.caption)
                        }

                        Text(DetectionFormatting.plantDisplayName(label))
                            .font(.subheadline)
                            .fontWeight(wouldAlert ? .bold : .regular)

                        Spacer()

                        // Per-class bar for toxic classes, so the readout matches the
                        // per-class gate rather than one global number.
                        if isToxic {
                            Text(String(format: "thr %.2f",
                                        ToxicityThresholds.alertThreshold(for: label, sensitivity: threshold)))
                                .font(.caption2)
                                .monospacedDigit()
                                .foregroundStyle(.secondary)
                        }

                        confidenceBar(confidence, isToxic: isToxic, wouldAlert: wouldAlert)

                        Text("\(Int(confidence * 100))%")
                            .monospacedDigit()
                            .font(.subheadline)
                            .foregroundStyle(wouldAlert ? .red : (wouldSurface ? .yellow : .secondary))
                            .frame(width: 40, alignment: .trailing)
                    }
                }

                // Legend: thresholds are per class, so there is no single line.
                HStack {
                    Image(systemName: "line.horizontal.3.decrease")
                        .foregroundStyle(.orange)
                    Text("Per-class thresholds; hollow = \"verify visually\" band")
                    Spacer()
                    Text("sens \(String(format: "%.2f", threshold))")
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
            // These gates are in-memory and live: a tweak changes the running
            // patrol immediately and persists until relaunch (never to disk).
            // Defaults are shown beside each value; Reset restores all of them.
            HStack {
                Text("Live-only tuning · resets on relaunch")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Button("Reset to defaults") {
                    appState.captureEngine.resetTuningToDefaults()
                }
                .font(.caption)
            }
            // Sensitivity threshold
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Sensitivity Threshold")
                    Spacer()
                    Text("\(Int(appState.sensitivityThreshold * 100))%")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Slider(value: $appState.sensitivityThreshold, in: ToxicityThresholds.sensitivityRange, step: 0.05)
                    .tint(.orange)
            }

            // Capture cooldown
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Capture Cooldown")
                    Text(String(format: "(default %.1fs)", CaptureEngine.TuningDefaults.minCaptureInterval))
                        .font(.caption2).foregroundStyle(.tertiary)
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
                    Text(String(format: "(default %.0fs)", CaptureEngine.TuningDefaults.forcedCaptureInterval))
                        .font(.caption2).foregroundStyle(.tertiary)
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
                    Text(String(format: "(default %.3fg)", CaptureEngine.TuningDefaults.stillnessThreshold))
                        .font(.caption2).foregroundStyle(.tertiary)
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

            // Rotation rate threshold
            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Rotation Rate Limit")
                    Text(String(format: "(default %.0f°/s)", CaptureEngine.TuningDefaults.rotationRateThreshold * 180.0 / .pi))
                        .font(.caption2).foregroundStyle(.tertiary)
                    Spacer()
                    Text(String(format: "%.0f°/s", appState.captureEngine.rotationRateThreshold * 180.0 / .pi))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Slider(
                    value: Binding(
                        get: { appState.captureEngine.rotationRateThreshold },
                        set: { appState.captureEngine.rotationRateThreshold = $0 }
                    ),
                    in: 0.5...5.0, step: 0.1
                )
            }

            // Battery saver toggle
            Toggle("Battery Saver", isOn: $appState.batterySaverEnabled)

            // Audio alerts toggle
            Toggle("Audio Alerts", isOn: $appState.audioAlertsEnabled)
        }
    }

    // MARK: - Section 4: Frame Capture Debug

    private var frameCaptureSection: some View {
        Section("Frame Capture Debug") {
            Toggle("Save All Frames to Disk", isOn: $appState.debugSaveFrames)

            Text("Saves every captured frame as JPEG to Files → LeafAlert → debug_frames/. Useful for reviewing image quality. Disable when not needed — uses significant storage.")
                .font(.caption)
                .foregroundStyle(.secondary)

            if appState.debugSaveFrames {
                HStack {
                    Text("Saved Frames")
                    Spacer()
                    Text("\(DebugFrameSaver.shared.frameCount)")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }

                NavigationLink(destination: DebugCaptureReviewView()) {
                    Label("Open Capture Review", systemImage: "photo.stack")
                }

                Button("Clear Saved Frames", role: .destructive) {
                    DebugFrameSaver.shared.clearAll()
                }
            }
        }
    }

    // MARK: - Section 4b: Data Recording

    @State private var recordingTick = 0
    /// Cached session list. `listSessions()` walks two directory trees (one
    /// possibly on iCloud behind a security scope) and recursively sizes every
    /// session — far too heavy for `body`, which re-evaluates at up to 5 Hz on
    /// patrol. Refreshed on appear and whenever a recording starts/stops/deletes.
    @State private var sessions: [DataRecorder.SessionInfo] = []
    /// Live size of the in-progress recording, polled at 1 Hz off the main thread.
    @State private var liveSessionSize: Int64 = 0

    /// Guards "Force Capture Now" against re-entrancy: while a force is in
    /// flight, repeated taps must not re-read the already-forced (0.01s)
    /// value as the baseline to restore, which would jam the engine in
    /// continuous-capture mode with no in-app recovery.
    @State private var forceCaptureInFlight = false

    private var dataRecordingSection: some View {
        Section("Data Recording") {
            if recorder.isRecording {
                HStack {
                    Circle().fill(.red).frame(width: 10, height: 10)
                    Text("Recording")
                        .font(.subheadline.bold())
                    Spacer()
                    Text(formatDuration(recorder.elapsedSeconds))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }

                HStack {
                    Text("Session size")
                    Spacer()
                    Text(formatBytes(liveSessionSize))
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }

                Button("Stop Recording", role: .destructive) {
                    recorder.stop {
                        DispatchQueue.main.async { recordingTick += 1 }
                    }
                }
            } else {
                Button {
                    _ = recorder.start(videoSettings: DataRecorder.defaultVideoSettings)
                    recordingTick += 1
                } label: {
                    Label("Start Recording", systemImage: "record.circle")
                        .foregroundStyle(.red)
                }
                .disabled(!appState.isPatrolling)
                if !appState.isPatrolling {
                    // Leaving the Patrol screen stops the patrol, so this button
                    // can never be enabled from here. Say so instead of dimming.
                    Text("Needs a running patrol — start the patrol and use the REC button on the Patrol screen.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                let dest = FeedbackExporter.shared.hasSyncFolder
                    ? "iCloud folder (auto-syncs to Mac)"
                    : "Files → LeafAlert → recordings/"
                Text(appState.isPatrolling
                    ? "Captures continuous video + IMU + events to \(dest). Use for offline replay and gating calibration."
                    : "Start a patrol first to enable recording.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            // Sessions list (cached — see `sessions`)
            if !sessions.isEmpty {
                Text("Sessions (\(sessions.count))")
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)

                ForEach(sessions) { session in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            HStack(spacing: 4) {
                                if session.isSynced {
                                    Image(systemName: "icloud.fill")
                                        .font(.caption2)
                                        .foregroundStyle(.blue)
                                }
                                Text(session.id).font(.caption.monospacedDigit())
                            }
                            Text(formatBytes(session.sizeBytes))
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button {
                            recorder.deleteSession(session.id)
                            recordingTick += 1
                        } label: {
                            Image(systemName: "trash")
                                .foregroundStyle(.red)
                        }
                    }
                }
            }
        }
        .onAppear { refreshSessions() }
        .onChange(of: recordingTick) { _, _ in refreshSessions() }
        .task(id: recordingTick) {
            // Poll the live session size at 1 Hz while recording. This is a
            // recursive directory walk that used to run on every body evaluation.
            while !Task.isCancelled, recorder.isRecording {
                liveSessionSize = await Task.detached(priority: .utility) {
                    DataRecorder.shared.sessionFolderSize
                }.value
                try? await Task.sleep(nanoseconds: 1_000_000_000)
            }
        }
    }

    /// Lists sessions off the main thread and publishes the result.
    private func refreshSessions() {
        Task {
            let list = await Task.detached(priority: .utility) {
                DataRecorder.shared.listSessions()
            }.value
            sessions = list
        }
    }

    private var recorder: DataRecorder { DataRecorder.shared }

    private func formatDuration(_ seconds: TimeInterval) -> String {
        let m = Int(seconds) / 60
        let s = Int(seconds) % 60
        return String(format: "%02d:%02d", m, s)
    }

    private func formatBytes(_ bytes: Int64) -> String {
        let f = ByteCountFormatter()
        f.allowedUnits = [.useMB, .useKB]
        f.countStyle = .file
        return f.string(fromByteCount: bytes)
    }

    // MARK: - Section 5: Actions

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
                // Force a capture by setting a very short forced interval temporarily.
                // Guard against re-entrancy: if a force is already in flight, ignore the
                // tap so we never capture the already-forced (0.01s) value as the baseline
                // — doing so would leave the engine stuck in continuous capture forever.
                guard !forceCaptureInFlight else { return }
                forceCaptureInFlight = true
                let original = appState.captureEngine.forcedCaptureInterval
                appState.captureEngine.forcedCaptureInterval = 0.01
                DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
                    appState.captureEngine.forcedCaptureInterval = original
                    forceCaptureInFlight = false
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
    NavigationStack { DebugDashboardView() }
        .environmentObject(AppState())
}
