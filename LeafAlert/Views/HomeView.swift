import SwiftUI
import SwiftData

/// Main landing screen with Start Patrol button and navigation to other sections.
struct HomeView: View {
    @EnvironmentObject private var appState: AppState

    @Query(sort: \DetectionLog.timestamp, order: .reverse) private var allDetections: [DetectionLog]

    private var recentDetections: [DetectionLog] {
        Array(allDetections.prefix(3))
    }

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

                recentDetectionsSection

                Text(DetectionFormatting.safetyDisclaimer)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)
                    .padding(.bottom, 16)
            }
            .navigationBarTitleDisplayMode(.inline)
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
                Text(DetectionFormatting.plantDisplayName(log.plantType))
                    .font(.subheadline.weight(.medium))
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
