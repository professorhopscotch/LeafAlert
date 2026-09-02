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

    /// True if SwiftData fell back to in-memory storage due to a persistent store error.
    private(set) var isUsingInMemoryFallback = false

    var modelContext: ModelContext {
        modelContainer.mainContext
    }

    // MARK: - Init

    init() {
        let schema = Schema([DetectionLog.self])
        do {
            let config = ModelConfiguration(isStoredInMemoryOnly: false)
            modelContainer = try ModelContainer(for: schema, configurations: [config])
        } catch {
            print("[DetectionLogStore] Persistent store failed: \(error). Falling back to in-memory storage.")
            // Fall back to in-memory so detection still works even if logging can't persist.
            let fallbackConfig = ModelConfiguration(isStoredInMemoryOnly: true)
            do {
                modelContainer = try ModelContainer(for: schema, configurations: [fallbackConfig])
            } catch {
                // In-memory should never fail, but if it does, create a minimal container.
                modelContainer = try! ModelContainer(for: schema, configurations: [ModelConfiguration(isStoredInMemoryOnly: true)])
            }
            isUsingInMemoryFallback = true
        }
    }

    /// Returns the SwiftData ModelContainer for use with the .modelContainer modifier.
    var container: ModelContainer {
        modelContainer
    }

    // MARK: - CRUD

    /// Saves a new detection log entry from a detection result.
    func save(result: DetectionResult, imageData: Data?, synthetic: Bool = false) {
        let location = locationManager.location
        let log = DetectionLog(
            id: result.id,
            timestamp: result.timestamp,
            latitude: location?.coordinate.latitude ?? 0,
            longitude: location?.coordinate.longitude ?? 0,
            hasLocation: location != nil,
            confidence: result.confidence,
            plantType: result.plantType,
            imageThumbData: imageData
        )
        log.isSynthetic = synthetic
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

        // A synthetic (injected) detection is a flat placeholder image with a
        // made-up label. Recording the answer locally keeps the UI flow honest;
        // exporting it would feed fabricated "feedback" into the training pool.
        guard !log.isSynthetic else { return }

        // Export the feedback (image + manifest entry) for the retraining loop.
        // A "confirmed" tap exports the original prediction as the training label;
        // a "corrected" tap exports the user-supplied label.
        FeedbackExporter.shared.exportFeedback(
            imageData: log.imageThumbData,
            originalPrediction: log.plantType,
            correctedLabel: log.correctedLabel ?? log.plantType,
            feedbackStatus: status,
            confidence: log.confidence,
            latitude: log.latitude,
            longitude: log.longitude,
            timestamp: log.timestamp
        )
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

    /// Requests location permission for geotagging detections and begins
    /// receiving location fixes so `locationManager.location` is populated.
    func requestLocationPermission() {
        locationManager.desiredAccuracy = kCLLocationAccuracyNearestTenMeters
        locationManager.requestWhenInUseAuthorization()
        locationManager.startUpdatingLocation()
    }

    /// Stops receiving location fixes (e.g. when patrol ends) to conserve power.
    func stopLocationUpdates() {
        locationManager.stopUpdatingLocation()
    }
}
