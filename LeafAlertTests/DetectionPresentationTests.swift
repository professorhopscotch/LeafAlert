import XCTest
import CoreGraphics
@testable import LeafAlert

/// Tests for what the app SAYS and SHOWS about a detection.
///
/// For a plant-safety app the copy is part of the safety surface: text that implies
/// certainty ("Poison Ivy", "100% match") invites someone to trust a model that is
/// wrong ~30% of the time. These tests pin the hedging so a future copy edit can't
/// quietly remove it.
final class DetectionPresentationTests: XCTestCase {

    // MARK: - Hedged copy

    func testHeadlineHedgesForEverySeverity() {
        let alert = DetectionFormatting.detectionHeadline("poison_ivy", severity: .alert)
        let uncertain = DetectionFormatting.detectionHeadline("poison_ivy", severity: .uncertain)

        XCTAssertTrue(alert.lowercased().hasPrefix("likely"), "alert headline should hedge: \(alert)")
        XCTAssertTrue(uncertain.lowercased().hasPrefix("possible"), "uncertain headline should hedge harder: \(uncertain)")
        // Both must still name the plant so the user knows what to look for.
        XCTAssertTrue(alert.contains("Poison Ivy"))
        XCTAssertTrue(uncertain.contains("Poison Ivy"))
    }

    func testSubtitleAlwaysDirectsUserToVerifyVisually() {
        for severity in [DetectionSeverity.alert, .uncertain, .ignore] {
            let s = DetectionFormatting.detectionSubtitle(confidence: 0.9, severity: severity)
            XCTAssertTrue(s.lowercased().contains("verify"),
                          "subtitle for \(severity) must tell the user to verify: \(s)")
        }
    }

    func testCopyNeverClaimsSomethingIsSafe() {
        // The app must never produce a confident all-clear: it cannot distinguish
        // "no toxic plant" from "did not recognise the toxic plant".
        for severity in [DetectionSeverity.alert, .uncertain, .ignore] {
            let text = (DetectionFormatting.detectionHeadline("poison_ivy", severity: severity)
                        + " " + DetectionFormatting.detectionSubtitle(confidence: 0.9, severity: severity))
                .lowercased()
            XCTAssertFalse(text.contains("no poison"), "copy implies an all-clear: \(text)")
            XCTAssertFalse(text.contains("is safe"), "copy implies an all-clear: \(text)")
            XCTAssertFalse(text.contains("certain"), "copy implies certainty: \(text)")
        }
    }

    func testUncertainSubtitleDoesNotShowARawPercentage() {
        // A precise-looking number on a low-confidence guess reads as authority the
        // model has not earned.
        let s = DetectionFormatting.detectionSubtitle(confidence: 0.42, severity: .uncertain)
        XCTAssertFalse(s.contains("42%"), "uncertain copy should not present a raw score: \(s)")
    }

    func testPlantDisplayNameHumanisesRawLabels() {
        XCTAssertEqual(DetectionFormatting.plantDisplayName("poison_ivy"), "Poison Ivy")
        XCTAssertEqual(DetectionFormatting.plantDisplayName("safe_plants"), "Safe Plants")
    }

    func testSafetyDisclaimerMentionsVerification() {
        XCTAssertTrue(DetectionFormatting.safetyDisclaimer.lowercased().contains("verify"))
    }

    // MARK: - Bounding-box coordinate mapping

    /// Vision's origin is bottom-left; the view's is top-left. Getting this wrong
    /// mirrors the box vertically and points the user at the wrong patch of ground.
    func testVisionRectFlipsVerticallyIntoViewSpace() {
        let size = CGSize(width: 100, height: 200)
        // A box hugging the BOTTOM in Vision space (y ≈ 0)…
        let visionBottom = CGRect(x: 0.0, y: 0.0, width: 0.5, height: 0.25)
        let view = BoundingBoxOverlay.visionToView(visionBottom, in: size, previewLayer: nil)
        // …must land at the BOTTOM of the view (large y), not the top.
        XCTAssertGreaterThan(view.midY, size.height / 2,
                             "vision y=0 should map to the bottom of the view, got \(view)")
    }

    func testMappedRectIsClampedToViewBounds() {
        let size = CGSize(width: 100, height: 200)
        // Deliberately overflowing rect.
        let huge = CGRect(x: -0.5, y: -0.5, width: 3.0, height: 3.0)
        let view = BoundingBoxOverlay.visionToView(huge, in: size, previewLayer: nil)
        XCTAssertGreaterThanOrEqual(view.minX, 0)
        XCTAssertGreaterThanOrEqual(view.minY, 0)
        XCTAssertLessThanOrEqual(view.maxX, size.width)
        XCTAssertLessThanOrEqual(view.maxY, size.height)
    }

    func testMappedRectScalesWithViewSize() {
        let box = CGRect(x: 0.25, y: 0.25, width: 0.5, height: 0.5)
        let small = BoundingBoxOverlay.visionToView(box, in: CGSize(width: 100, height: 100), previewLayer: nil)
        let large = BoundingBoxOverlay.visionToView(box, in: CGSize(width: 400, height: 400), previewLayer: nil)
        XCTAssertGreaterThan(large.width, small.width)
    }
}
