import SwiftUI
import SwiftData

@main
struct LeafAlertApp: App {
    @StateObject private var appState = AppState()

    init() {
        #if DEBUG
        // UI tests launch with `-resetSettings 1` so persisted simulator state
        // (a moved sensitivity slider, toggles) cannot change what a test
        // expects. The disclaimer flag is deliberately left alone: the tests
        // handle the sheet either way, and the detection log is store-backed.
        if UserDefaults.standard.bool(forKey: "resetSettings") {
            for key in ["sensitivityThreshold", "audioAlertsEnabled", "screenDimLevel",
                        "batterySaverEnabled", "livePreviewEnabled", "debugSaveFrames"] {
                UserDefaults.standard.removeObject(forKey: key)
            }
        }
        #endif
    }

    var body: some Scene {
        WindowGroup {
            HomeView()
                .environmentObject(appState)
                .modelContainer(appState.detectionLogStore.container)
                .sheet(isPresented: .constant(!appState.hasShownDisclaimer)) {
                    DisclaimerView {
                        appState.markDisclaimerShown()
                    }
                }
        }
    }
}

/// First-launch disclaimer shown once before the user can use the app.
struct DisclaimerView: View {
    let onAccept: () -> Void

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: "exclamationmark.triangle.fill")
                .font(.system(size: 60))
                .foregroundStyle(.yellow)

            Text("Important Disclaimer")
                .font(.title.bold())

            Text("This app assists identification — always verify visually before touching any plant.\n\nLeafAlert uses machine learning to help detect potentially toxic plants, but no detection system is perfect. Always exercise caution and consult expert resources when in doubt.")
                .font(.body)
                .multilineTextAlignment(.center)
                .padding(.horizontal)

            Spacer()

            Button("I Understand") {
                onAccept()
            }
            .font(.headline)
            .frame(maxWidth: .infinity)
            .padding()
            .background(.green)
            .foregroundStyle(.white)
            .clipShape(RoundedRectangle(cornerRadius: 16))
            .padding(.horizontal)
            .padding(.bottom, 32)
        }
        .interactiveDismissDisabled()
    }
}

#Preview {
    HomeView()
        .environmentObject(AppState())
}
