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
    var confidence: Float
    var plantType: String
    var userConfirmed: Bool?
    @Attribute(.externalStorage) var imageThumbData: Data?

    init(
        id: UUID = UUID(),
        timestamp: Date = Date(),
        latitude: Double = 0,
        longitude: Double = 0,
        confidence: Float = 0,
        plantType: String = "",
        userConfirmed: Bool? = nil,
        imageThumbData: Data? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.latitude = latitude
        self.longitude = longitude
        self.confidence = confidence
        self.plantType = plantType
        self.userConfirmed = userConfirmed
        self.imageThumbData = imageThumbData
    }
}
