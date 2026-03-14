import SwiftUI

/// User-configurable settings for detection sensitivity, alerts, and battery management.
struct SettingsView: View {
    @AppStorage("sensitivityThreshold") private var sensitivityThreshold: Double = 0.65
    @AppStorage("audioAlertsEnabled") private var audioAlertsEnabled = true
    @AppStorage("screenDimLevel") private var screenDimLevel: Double = 0.7
    @AppStorage("batterySaverEnabled") private var batterySaverEnabled = false

    @State private var showClearConfirmation = false

    @EnvironmentObject private var appState: AppState

    var body: some View {
        Form {
            Section("Detection Sensitivity") {
                VStack(alignment: .leading) {
                    Text("Confidence Threshold: \(Int(sensitivityThreshold * 100))%")
                    Slider(value: $sensitivityThreshold, in: 0...1, step: 0.05)
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
                VStack(alignment: .leading) {
                    Text("Screen Dim Level: \(Int(screenDimLevel * 100))%")
                    Slider(value: $screenDimLevel, in: 0.3...1.0, step: 0.05)
                }
                Text("Controls how much the screen dims during active patrol to save battery.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Battery") {
                Toggle("Battery Saver Mode", isOn: $batterySaverEnabled)
                Text("Reduces capture frequency to once every 3 seconds.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Data") {
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
