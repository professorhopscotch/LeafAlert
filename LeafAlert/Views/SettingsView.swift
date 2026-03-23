import SwiftUI

/// User-configurable settings for detection sensitivity, alerts, and battery management.
struct SettingsView: View {
    @AppStorage("sensitivityThreshold") private var sensitivityThreshold: Double = 0.65
    @AppStorage("audioAlertsEnabled") private var audioAlertsEnabled = true
    @AppStorage("screenDimLevel") private var screenDimLevel: Double = 0.7
    @AppStorage("batterySaverEnabled") private var batterySaverEnabled = false
    @AppStorage("livePreviewEnabled") private var livePreviewEnabled = true

    @ObservedObject private var uploader = FeedbackUploader.shared
    @State private var manualHost = ""
    @State private var showClearConfirmation = false

    @EnvironmentObject private var appState: AppState

    var body: some View {
        Form {
            Section("Detection Sensitivity") {
                VStack(alignment: .leading) {
                    Text("Confidence Threshold: \(Int(sensitivityThreshold * 100))%")
                    Slider(value: $sensitivityThreshold, in: 0.3...1, step: 0.05)
                }
                Text("Lower values trigger more alerts but may include false positives.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Alerts") {
                Toggle("Audio Alerts", isOn: $audioAlertsEnabled)
                Text("When enabled, medium and high confidence detections produce an audio chirp in addition to haptic feedback.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Display") {
                Toggle("Live Camera Preview", isOn: $livePreviewEnabled)
                Text("Shows the camera feed during patrol with bounding boxes around detections. Disabling saves battery.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                VStack(alignment: .leading) {
                    Text("Screen Dim Level: \(Int(screenDimLevel * 100))%")
                    Slider(value: $screenDimLevel, in: 0.3...1.0, step: 0.05)
                }
                Text("Controls how much the screen dims during active patrol to save battery. Only applies when live preview is off.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Battery") {
                Toggle("Battery Saver Mode", isOn: $batterySaverEnabled)
                Text("Reduces capture frequency to once every 3 seconds.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Feedback Sync") {
                // Server status
                HStack {
                    Image(systemName: uploader.isServerAvailable ? "wifi" : "wifi.slash")
                        .foregroundStyle(uploader.isServerAvailable ? .green : .secondary)
                    if let address = uploader.serverAddress {
                        Text(address)
                            .font(.subheadline.monospaced())
                    } else {
                        Text("Searching for server\u{2026}")
                            .foregroundStyle(.secondary)
                    }
                }

                // Manual server entry
                HStack {
                    TextField("Server IP (e.g. 192.168.1.42)", text: $manualHost)
                        .textContentType(.URL)
                        .autocorrectionDisabled()
                        .textInputAutocapitalization(.never)
                        .font(.subheadline.monospaced())
                    Button("Connect") {
                        let host = manualHost.trimmingCharacters(in: .whitespaces)
                        guard !host.isEmpty else { return }
                        uploader.setManualServer(host: host)
                    }
                    .disabled(manualHost.trimmingCharacters(in: .whitespaces).isEmpty)
                }

                // Sync button + counts
                HStack {
                    Button {
                        uploader.syncAll()
                    } label: {
                        HStack(spacing: 6) {
                            if uploader.isSyncing {
                                ProgressView()
                                    .controlSize(.small)
                            }
                            Text(uploader.isSyncing ? "Syncing\u{2026}" : "Sync Now")
                        }
                    }
                    .disabled(!uploader.isServerAvailable || uploader.isSyncing)

                    Spacer()

                    Text("\(uploader.syncedCount) synced")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let error = uploader.lastError {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                }

                if let lastSync = uploader.lastSyncDate {
                    Text("Last sync: \(lastSync.formatted(.relative(presentation: .named)))")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Text("Run `python3 scripts/feedback_server.py` on your Mac. The app will discover it automatically via Bonjour, or enter the IP manually.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Data") {
                LabeledContent("Feedback entries", value: "\(FeedbackExporter.shared.feedbackCount)")

                Button("Clear Detection Log", role: .destructive) {
                    showClearConfirmation = true
                }
            }

            Section("About") {
                LabeledContent("Version", value: "1.0.0")
                Text("LeafAlert uses on-device machine learning to help identify toxic plants. No data ever leaves your device.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
        .confirmationDialog(
            "Clear all detection logs?",
            isPresented: $showClearConfirmation,
            titleVisibility: .visible
        ) {
            Button("Clear All Logs", role: .destructive) {
                appState.detectionLogStore.clearAll()
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This will permanently delete all saved detections and photos. This cannot be undone.")
        }
    }
}

#Preview {
    NavigationStack {
        SettingsView()
            .environmentObject(AppState())
    }
}
