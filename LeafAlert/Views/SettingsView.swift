import SwiftUI

/// User-configurable settings for detection sensitivity, alerts, and battery management.
struct SettingsView: View {
    @AppStorage("sensitivityThreshold") private var sensitivityThreshold: Double = 0.50
    @AppStorage("audioAlertsEnabled") private var audioAlertsEnabled = true
    @AppStorage("screenDimLevel") private var screenDimLevel: Double = 0.7
    @AppStorage("batterySaverEnabled") private var batterySaverEnabled = false
    @AppStorage("livePreviewEnabled") private var livePreviewEnabled = true

    @ObservedObject private var exporter = FeedbackExporter.shared
    @State private var showFolderPicker = false
    @State private var showClearConfirmation = false

    @EnvironmentObject private var appState: AppState

    /// A qualitative label for the sensitivity slider. A LOWER threshold means the
    /// app alerts more readily (more sensitive), so the wording is inverted from the
    /// raw value on purpose.
    private var sensitivityLabel: String {
        switch sensitivityThreshold {
        case ..<0.45: return "High (more alerts)"
        case 0.45..<0.60: return "Balanced"
        default: return "Low (fewer alerts)"
        }
    }

    /// "1.2.0 (34)" from the bundle, so Settings never drifts from the build.
    private static var versionString: String {
        let info = Bundle.main.infoDictionary
        let short = info?["CFBundleShortVersionString"] as? String ?? "?"
        let build = info?["CFBundleVersion"] as? String
        return build.map { "\(short) (\($0))" } ?? short
    }

    var body: some View {
        Form {
            Section("Detection Sensitivity") {
                VStack(alignment: .leading) {
                    Text("Alert Sensitivity: \(sensitivityLabel)")
                    Slider(value: $sensitivityThreshold, in: ToxicityThresholds.sensitivityRange, step: 0.05)
                }
                Text("More sensitive catches more toxic plants but raises false alarms; less sensitive is quieter but can miss plants. Near-misses are always shown as \u{201C}possible — verify visually.\u{201D} No setting is a substitute for looking.")
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
                Text("Captures at most once every \(Int(CaptureEngine.TuningDefaults.batterySaverInterval)) seconds instead of every \(Int(CaptureEngine.TuningDefaults.minCaptureInterval)), and idles the camera sensor between captures.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Feedback Sync") {
                if exporter.hasSyncFolder {
                    HStack {
                        Image(systemName: "checkmark.icloud.fill")
                            .foregroundStyle(.green)
                        Text(exporter.syncFolderName ?? "iCloud Drive")
                            .font(.subheadline)
                    }

                    Button("Change Folder") {
                        showFolderPicker = true
                    }

                    Button("Remove Sync Folder", role: .destructive) {
                        exporter.clearSyncFolder()
                    }
                } else {
                    Button {
                        showFolderPicker = true
                    } label: {
                        HStack {
                            Image(systemName: "icloud.and.arrow.up")
                            Text("Set iCloud Sync Folder")
                        }
                    }
                }

                Text("Pick a folder in iCloud Drive. Feedback images and metadata sync to your Mac automatically.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Section("Data") {
                LabeledContent("Feedback entries", value: "\(exporter.feedbackCount)")

                Button("Clear Detection Log", role: .destructive) {
                    showClearConfirmation = true
                }
            }

            Section("About") {
                LabeledContent("Version", value: Self.versionString)
                Text("LeafAlert uses on-device machine learning to help identify toxic plants. No data ever leaves your device.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Settings")
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showFolderPicker) {
            FolderPickerView { url in
                exporter.setSyncFolder(url)
            }
        }
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

/// Wraps UIDocumentPickerViewController to let the user choose a folder (e.g., in iCloud Drive).
struct FolderPickerView: UIViewControllerRepresentable {
    let onPick: (URL) -> Void

    func makeUIViewController(context: Context) -> UIDocumentPickerViewController {
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder])
        // Default to the local feedback folder so the user starts near their captures.
        // Falls back gracefully if the directory doesn't exist yet.
        let feedbackDir = FeedbackExporter.shared.feedbackDirectoryURL
        if FileManager.default.fileExists(atPath: feedbackDir.path) {
            picker.directoryURL = feedbackDir
        }
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIDocumentPickerViewController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(onPick: onPick) }

    class Coordinator: NSObject, UIDocumentPickerDelegate {
        let onPick: (URL) -> Void
        init(onPick: @escaping (URL) -> Void) { self.onPick = onPick }

        func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
            guard let url = urls.first else { return }
            onPick(url)
        }
    }
}

#Preview {
    NavigationStack {
        SettingsView()
            .environmentObject(AppState())
    }
}
