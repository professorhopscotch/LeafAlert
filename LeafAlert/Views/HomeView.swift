import SwiftUI
import SwiftData

/// Screens reachable from Home. Doubles as the deep-link vocabulary:
/// `leafalert://patrol`, `leafalert://map`, `leafalert://plants`,
/// `leafalert://settings`, `leafalert://debug` (debug builds only).
enum Route: String, Hashable, CaseIterable {
    case patrol, map, plants, settings, debug

    init?(url: URL) {
        guard url.scheme?.lowercased() == "leafalert" else { return nil }
        let key = (url.host() ?? url.lastPathComponent).lowercased()
        self.init(rawValue: key)
    }
}

/// Main landing screen with Start Patrol button and navigation to other sections.
struct HomeView: View {
    @EnvironmentObject private var appState: AppState
    @State private var path = NavigationPath()
    @State private var didApplyLaunchRoute = false

    @Query(sort: \DetectionLog.timestamp, order: .reverse) private var allDetections: [DetectionLog]

    private var recentDetections: [DetectionLog] {
        Array(allDetections.prefix(3))
    }

    var body: some View {
        NavigationStack(path: $path) {
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

                NavigationLink(value: Route.patrol) {
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
                    NavigationLink(value: Route.map) {
                        Label("Map", systemImage: "map")
                    }
                    NavigationLink(value: Route.plants) {
                        Label("Plants", systemImage: "leaf")
                    }
                    NavigationLink(value: Route.settings) {
                        Label("Settings", systemImage: "gear")
                    }
                    #if DEBUG
                    NavigationLink(value: Route.debug) {
                        Label("Debug", systemImage: "ant.fill")
                    }
                    #endif
                }
                .font(.callout)

                recentDetectionsSection

                Text(DetectionFormatting.safetyDisclaimer)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                    .padding(.bottom, 16)
            }
            .navigationBarTitleDisplayMode(.inline)
            .navigationDestination(for: Route.self) { route in
                destination(for: route)
            }
        }
        // Deep links (also how the simulator smoke test drives the app): always
        // land on the requested screen from Home, wherever the user was.
        .onOpenURL { url in
            guard let route = Route(url: url) else { return }
            path = NavigationPath()
            path.append(route)
        }
        #if DEBUG
        // Test hook: `xcrun simctl launch booted com.leafalert.app -launchRoute patrol`
        // routes straight to a screen. Custom-URL deep links opened from outside
        // the app trigger an "Open in LeafAlert?" system prompt that headless
        // automation cannot dismiss; a launch argument (auto-registered into
        // UserDefaults' argument domain) does not. Applied once per launch.
        .onAppear {
            guard !didApplyLaunchRoute,
                  let key = UserDefaults.standard.string(forKey: "launchRoute"),
                  let route = Route(rawValue: key.lowercased()) else { return }
            didApplyLaunchRoute = true
            path = NavigationPath()
            path.append(route)
        }
        #endif
    }

    @ViewBuilder
    private func destination(for route: Route) -> some View {
        switch route {
        case .patrol: PatrolView()
        case .map: PatrolMapView()
        case .plants: PlantDetailView()
        case .settings: SettingsView()
        case .debug:
            #if DEBUG
            DebugDashboardView()
            #else
            SettingsView()
            #endif
        }
    }

    // MARK: - Recent Detections

    @ViewBuilder
    private var recentDetectionsSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Recent Detections")
                .font(.headline)
                .padding(.horizontal)

            if recentDetections.isEmpty {
                Text("No detections yet")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 8)
            } else {
                ForEach(recentDetections) { log in
                    recentDetectionRow(log)
                }
            }
        }
        .padding(.horizontal)
    }

    private func recentDetectionRow(_ log: DetectionLog) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(DetectionFormatting.plantDisplayName(log.plantType))
                        .font(.subheadline.weight(.medium))
                    if log.isSynthetic {
                        Text("SYNTHETIC")
                            .font(.caption2.bold())
                            .padding(.horizontal, 5).padding(.vertical, 1)
                            .background(.orange.opacity(0.25))
                            .clipShape(Capsule())
                    }
                }
                Text(DetectionFormatting.relativeTimestamp(log.timestamp))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text("\(Int(log.confidence * 100))%")
                .font(.subheadline.monospacedDigit())
                .foregroundStyle(DetectionFormatting.confidenceColor(log.confidence))
        }
        .padding(.vertical, 4)
        .padding(.horizontal)
    }
}

#Preview {
    HomeView()
        .environmentObject(AppState())
}
