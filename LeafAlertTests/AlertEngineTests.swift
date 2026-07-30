import XCTest
import CoreGraphics
@testable import LeafAlert

/// Tests for alert delivery gating. `process` returns the severity it evaluated, so
/// we can assert the decision without observing haptics or audio.
final class AlertEngineTests: XCTestCase {

    private func makeEngine(threshold: Float = ToxicityThresholds.neutralSensitivity) -> AlertEngine {
        let engine = AlertEngine()
        engine.sensitivityThreshold = threshold
        engine.audioAlertsEnabled = false   // keep tests silent
        return engine
    }

    private func detection(_ plant: String, _ confidence: Float) -> DetectionResult {
        DetectionResult(plantType: plant, confidence: confidence, boundingBox: .zero)
    }

    func testSafePlantsNeverAlerts() {
        let engine = makeEngine()
        XCTAssertEqual(engine.process(detection("safe_plants", 0.99)), .ignore)
    }

    func testUnknownClassNeverAlerts() {
        let engine = makeEngine()
        XCTAssertEqual(engine.process(detection("not_a_plant", 0.99)), .ignore)
    }

    func testHighConfidenceToxicAlerts() {
        let engine = makeEngine()
        XCTAssertEqual(engine.process(detection("poison_ivy", 0.95)), .alert)
    }

    func testNearMissIsSurfacedAsUncertainRatherThanSilenced() {
        let engine = makeEngine()
        let thr = ToxicityThresholds.alertThreshold(for: "poison_ivy",
                                                    sensitivity: ToxicityThresholds.neutralSensitivity)
        XCTAssertEqual(engine.process(detection("poison_ivy", thr - 0.05)), .uncertain)
    }

    func testImpossibleConfidenceIsRejected() {
        // A confidence above 1.0 means the model output is not a probability; the
        // thresholds would be meaningless, so refuse rather than alert on garbage.
        let engine = makeEngine()
        XCTAssertEqual(engine.process(detection("poison_ivy", 1.5)), .ignore)
    }

    func testCooldownStillReportsSeverityWhileSuppressingRepeatCues() {
        // Two alert-worthy detections back to back: the second is inside the
        // cooldown, but the caller must still learn it was alert-worthy so the UI
        // keeps showing the warning card.
        let engine = makeEngine()
        XCTAssertEqual(engine.process(detection("poison_ivy", 0.95)), .alert)
        XCTAssertEqual(engine.process(detection("poison_ivy", 0.95)), .alert)
    }

    func testRaisingSensitivityThresholdSuppressesMarginalDetections() {
        let strict = makeEngine(threshold: 0.90)
        // Comfortably alert-worthy at the neutral bar, but not at a strict one.
        let severity = strict.process(detection("poison_ivy", 0.45))
        XCTAssertNotEqual(severity, .alert)
    }

    func testEveryToxicClassCanAlert() {
        for plant in InferenceEngine.toxicLabels {
            let engine = makeEngine()   // fresh engine avoids the cooldown
            XCTAssertEqual(engine.process(detection(plant, 0.99)), .alert,
                           "\(plant) failed to alert at 0.99 confidence")
        }
    }
}
