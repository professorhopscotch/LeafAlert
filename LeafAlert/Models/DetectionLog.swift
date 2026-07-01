import Foundation
import SwiftData
import CoreLocation

/// Persisted detection record stored via SwiftData. Never leaves the device.
@Model
final class DetectionLog {
    var id: UUID
    var timestamp: Date
    var latitude: Double
    var longitude: Double
    /// True when latitude/longitude reflect a real GPS fix. When false, the
    /// (0,0) coordinate means "no fix" rather than a literal location.
    var hasLocation: Bool = false
    var confidence: Float
    var plantType: String
    var feedbackStatus: String = "none"
    var correctedLabel: String?
    @Attribute(.externalStorage) var imageThumbData: Data?

    var hasUserFeedback: Bool { feedbackStatus != "none" }

    init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        latitude: Double = 0,
        longitude: Double = 0,
        hasLocation: Bool = false,
        confidence: Float = 0,
        plantType: String = "",
        feedbackStatus: String = "none",
        correctedLabel: String? = nil,
        imageThumbData: Data? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.hasLocation = hasLocation
        self.confidence = confidence
        self.plantType = plantType
        self.feedbackStatus = feedbackStatus
        self.correctedLabel = correctedLabel
        self.imageThumbData = imageThumbData
    }
}
