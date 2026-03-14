import Foundation
import CoreGraphics

/// Result from a single inference pass on a camera frame.
struct DetectionResult: Identifiable, Sendable {
    let id: UUID
    let plantType: String
    let confidence: Float
    let boundingBox: CGRect
    let timestamp: Date

    init(
        id: UUID = UUID(),
        plantType: String,
        confidence: Float,
        boundingBox: CGRect,
        timestamp: Date = Date()
    ) {
        self.id = id
        self.plantType = plantType
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.timestamp = timestamp
    }
}
