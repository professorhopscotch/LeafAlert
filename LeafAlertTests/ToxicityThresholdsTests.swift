import XCTest
@testable import LeafAlert

/// Tests for the alert decision — the single most safety-critical piece of logic in
/// the app. A regression here either fires warnings the user learns to ignore, or
/// silently withholds a warning about a toxic plant.
final class ToxicityThresholdsTests: XCTestCase {

    private let neutral = ToxicityThresholds.neutralSensitivity  // 0.50

    // MARK: - Per-class thresholds

    func testPerClassThresholdsMatchDerivedValues() {
        // Re-derived on the frozen held-out set; ivy/oak separate weakly so they
        // alert at a lower bar than sumac. If these drift, re-run
        // scripts/evaluate_model.py rather than editing the expectation.
        XCTAssertEqual(ToxicityThresholds.alertThreshold(for: "poison_ivy", sensitivity: neutral), 0.40, accuracy: 0.0001)
        XCTAssertEqual(ToxicityThresholds.alertThreshold(for: "poison_oak", sensitivity: neutral), 0.40, accuracy: 0.0001)
        XCTAssertEqual(ToxicityThresholds.alertThreshold(for: "poison_sumac", sensitivity: neutral), 0.52, accuracy: 0.0001)
    }

    func testUnknownClassFallsBackToDefaultThreshold() {
        // A label the table doesn't know must still get a sane threshold, never 0
        // (which would alert on everything) and never 1 (which would never alert).
        let t = ToxicityThresholds.alertThreshold(for: "not_a_real_class", sensitivity: neutral)
        XCTAssertGreaterThan(t, 0.0)
        XCTAssertLessThan(t, 1.0)
    }

    func testSensitivityBiasesThresholdsInTheExpectedDirection() {
        // Lower slider value = MORE sensitive = lower bar to alert.
        let sensitive = ToxicityThresholds.alertThreshold(for: "poison_ivy", sensitivity: 0.30)
        let base = ToxicityThresholds.alertThreshold(for: "poison_ivy", sensitivity: neutral)
        let strict = ToxicityThresholds.alertThreshold(for: "poison_ivy", sensitivity: 0.80)
        XCTAssertLessThan(sensitive, base)
        XCTAssertGreaterThan(strict, base)
    }

    func testThresholdIsClampedToUsableRange() {
        // Extreme slider values must not produce a threshold that alerts on
        // everything or on nothing.
        for s in [Float(-5.0), -0.5, 0.0, 1.0, 5.0] {
            let t = ToxicityThresholds.alertThreshold(for: "poison_ivy", sensitivity: s)
            XCTAssertGreaterThanOrEqual(t, 0.15, "threshold \(t) too low at sensitivity \(s)")
            XCTAssertLessThanOrEqual(t, 0.95, "threshold \(t) too high at sensitivity \(s)")
        }
    }

    // MARK: - Severity banding

    func testConfidenceAtOrAboveThresholdAlerts() {
        let thr = ToxicityThresholds.alertThreshold(for: "poison_ivy", sensitivity: neutral)
        XCTAssertEqual(ToxicityThresholds.severity(plantType: "poison_ivy", confidence: thr, sensitivity: neutral), .alert)
        XCTAssertEqual(ToxicityThresholds.severity(plantType: "poison_ivy", confidence: thr + 0.2, sensitivity: neutral), .alert)
        XCTAssertEqual(ToxicityThresholds.severity(plantType: "poison_ivy", confidence: 1.0, sensitivity: neutral), .alert)
    }

    func testNearMissLandsInUncertainBandNotSilence() {
        // The whole point of the uncertain band: a toxic plant just below the alert
        // bar must still reach the user as "verify visually", never silence.
        let thr = ToxicityThresholds.alertThreshold(for: "poison_ivy", sensitivity: neutral)
        let justBelow = thr - 0.01
        XCTAssertEqual(ToxicityThresholds.severity(plantType: "poison_ivy", confidence: justBelow, sensitivity: neutral), .uncertain)

        let bottomOfBand = thr - ToxicityThresholds.uncertaintyMargin
        XCTAssertEqual(ToxicityThresholds.severity(plantType: "poison_ivy", confidence: bottomOfBand, sensitivity: neutral), .uncertain)
    }

    func testWellBelowBandIsIgnored() {
        let thr = ToxicityThresholds.alertThreshold(for: "poison_ivy", sensitivity: neutral)
        let below = thr - ToxicityThresholds.uncertaintyMargin - 0.01
        XCTAssertEqual(ToxicityThresholds.severity(plantType: "poison_ivy", confidence: below, sensitivity: neutral), .ignore)
        XCTAssertEqual(ToxicityThresholds.severity(plantType: "poison_ivy", confidence: 0.0, sensitivity: neutral), .ignore)
    }

    func testSeverityIsMonotonicInConfidence() {
        // Raising confidence must never make the app LESS likely to warn.
        let rank: [DetectionSeverity: Int] = [.ignore: 0, .uncertain: 1, .alert: 2]
        for cls in ["poison_ivy", "poison_oak", "poison_sumac"] {
            var previous = 0
            for step in stride(from: Float(0.0), through: 1.0, by: 0.02) {
                let s = ToxicityThresholds.severity(plantType: cls, confidence: step, sensitivity: neutral)
                let r = rank[s]!
                XCTAssertGreaterThanOrEqual(r, previous, "severity dropped at confidence \(step) for \(cls)")
                previous = r
            }
        }
    }

    func testActionableCoversAlertAndUncertainOnly() {
        XCTAssertTrue(DetectionSeverity.alert.isActionable)
        XCTAssertTrue(DetectionSeverity.uncertain.isActionable)
        XCTAssertFalse(DetectionSeverity.ignore.isActionable)
    }

    // MARK: - Cross-class invariant

    func testEveryToxicLabelHasAThreshold() {
        // A toxic class missing from the table would silently fall back to the
        // default bar rather than its derived one.
        for label in InferenceEngine.toxicLabels {
            XCTAssertNotNil(ToxicityThresholds.baseAlert[label],
                            "toxic label \(label) has no derived threshold")
        }
    }

    func testSafePlantsIsNotTreatedAsToxic() {
        XCTAssertFalse(InferenceEngine.toxicLabels.contains("safe_plants"))
    }
}
