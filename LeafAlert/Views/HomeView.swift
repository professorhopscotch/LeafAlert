import SwiftUI

/// Main landing screen with Start Patrol button and navigation to other sections.
struct HomeView: View {
    @EnvironmentObject private var appState: AppState

    var body: some View {
        NavigationStack {
            VStack(spacing: 32) {
                Spacer()

                Image(systemName: "leaf.fill")
                    .font(.system(size: 80))
                    .foregroundStyle(.green)

                Text("LeafAlert")
                    .font(.largeTitle.bold())

                Text("Detect toxic plants while you hike.\nNo internet required.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)

                Spacer()

                NavigationLink(destination: PatrolView()) {
                    Label("Start Patrol", systemImage: "figure.hiking")
                        .font(.title2.bold())
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(.green)
                        .foregroundStyle(.white)
                        .clipShape(RoundedRectangle(cornerRadius: 16))
                }
                .padding(.horizontal)

                HStack(spacing: 24) {
                    NavigationLink(destination: PatrolMapView()) {
                        Label("Map", systemImage: "map")
                    }
                    NavigationLink(destination: PlantDetailView()) {
                        Label("Plants", systemImage: "leaf")
                    }
                    NavigationLink(destination: SettingsView()) {
                        Label("Settings", systemImage: "gear")
                    }
                }
                .font(.callout)
                .padding(.bottom, 32)
            }
            .navigationBarTitleDisplayMode(.inline)
        }
    }
}

#Preview {
    HomeView()
        .environmentObject(AppState())
}
