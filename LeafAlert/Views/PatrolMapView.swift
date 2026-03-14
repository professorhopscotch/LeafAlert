import SwiftUI
import MapKit
import SwiftData

/// Displays all logged detections as color-coded pins on a MapKit map.
/// Works fully offline — no network tiles needed for base map.
struct PatrolMapView: View {
    @Query(sort: \DetectionLog.timestamp, order: .reverse) private var logs: [DetectionLog]

    var body: some View {
        Map {
            ForEach(logs) { log in
                Annotation(
                    log.plantType.replacingOccurrences(of: "_", with: " ").capitalized,
                    coordinate: CLLocationCoordinate2D(
                        latitude: log.latitude,
                        longitude: log.longitude
                    )
                ) {
                    pinView(for: log)
                }
            }
        }
        .navigationTitle("Detection Map")
        .navigationBarTitleDisplayMode(.inline)
        .overlay {
            if logs.isEmpty {
                ContentUnavailableView(
                    "No Detections Yet",
                    systemImage: "map",
                    description: Text("Start a patrol to begin logging detections.")
                )
            }
        }
    }

    @ViewBuilder
    private func pinView(for log: DetectionLog) -> some View {
        Circle()
            .fill(pinColor(for: log.confidence))
            .frame(width: 14, height: 14)
            .overlay(
                Circle().stroke(.white, lineWidth: 2)
            )
    }

    private func pinColor(for confidence: Float) -> Color {
        switch confidence {
        case ..<0.75:
            return .yellow
        case 0.75..<0.90:
            return .orange
        default:
            return .red
        }
    }
}

#Preview {
    NavigationStack {
        PatrolMapView()
    }
}
