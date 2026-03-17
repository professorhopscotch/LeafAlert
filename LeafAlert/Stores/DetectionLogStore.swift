import Foundation
import SwiftData
import CoreLocation

/// Manages persistence of detection records via SwiftData.
/// All data stays on-device — no sync, no upload, no network.
@MainActor
final class DetectionLogStore: ObservableObject {

    // MARK: - Properties

    private let modelContainer: ModelContainer
    private let locationManager = CLLocationManager()

    var modelContext: ModelContext {
        modelContainer.mainContext
    }

    // MARK: - Init

    init() {
        do {
            let schema = Schema([DetectionLog.self])
            let config = ModelConfiguration(isStoredInMemoryOnly: false)
            modelContainer = try ModelContainer(for: schema, configurations: [config])
        } catch {
            fatalError("[DetectionLogStore] Failed to create ModelContainer: \(error)")
        }
    }

    /// Returns the SwiftData ModelContainer for use with the .modelContainer modifier.
    var container: ModelContainer {
        modelContainer
    }

    // MARK: - CRUD

    /// Saves a new detection log entry from a detection result.
    func save(result: DetectionResult, imageData: Data?) {
        let location = locationManager.location
        let log = DetectionLog(
            id: result.id,
            timestamp: result.timestamp,
            latitude: location?.coordinate.latitude ?? 0,
            longitude: location?.coordinate.longitude ?? 0,
            confidence: result.confidence,
            plantType: result.plantType,
            imageThumbData: imageData
        )
        modelContext.insert(log)
        try? modelContext.save()
    }

    /// Submits user feedback for a detection log entry.
    func submitFeedback(logID: UUID, status: String, correctedLabel: String? = nil) {
        let predicate = #Predicate<DetectionLog> { $0.id == logID }
        let descriptor = FetchDescriptor(predicate: predicate)
        guard let log = try? modelContext.fetch(descriptor).first else { return }
        log.feedbackStatus = status
        log.correctedLabel = correctedLabel
        try? modelContext.save()
    }

    /// Fetches all detection logs, most recent first.
    func fetchAll() -> [DetectionLog] {
        let descriptor = FetchDescriptor<DetectionLog>(
            sortBy: [SortDescriptor(\.timestamp, order: .reverse)]
        )
        return (try? modelContext.fetch(descriptor)) ?? []
    }

    /// Deletes all detection log entries.
    func clearAll() {
        do {
            try modelContext.delete(model: DetectionLog.self)
            try modelContext.save()
        } catch {
            print("[DetectionLogStore] Failed to clear logs: \(error)")
        }
    }

    // MARK: - Location

    /// Requests location permission for geotagging detections.
    func requestLocationPermission() {
        locationManager.requestWhenInUseAuthorization()
    }
}
