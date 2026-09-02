import XCTest
import CoreMotion
@testable import LeafAlert

/// Tests for the hardware-free half of CaptureEngine: the stride-apex detector
/// and the duty-cycle decision. These pin the three defects a review found had
/// each individually neutralised "physics-based timing": a stale latch acted on
/// at cooldown expiry, the apex being detected at a zero-crossing (maximum
/// speed) instead of a peak, and no liveness guarantee under a rotation gate.
final class CaptureEngineTests: XCTestCase {

    // MARK: - ApexDetector

    /// Synthetic walking bounce: down = A·sin(2πft). The apex is the PEAK of
    /// `down` (phase 0.25 of each period); the zero-crossings are at phase 0 and
    /// 0.5. The old detector fired at a crossing — the blurriest instant.
    func testApexFiresOncePerStrideAtThePeakNotTheZeroCrossing() {
        var det = ApexDetector(peakFloor: 0.03, refractory: 0.25, smoothing: 1.0) // no EMA lag
        let f = 2.0, amplitude = 0.3, dt = 0.01
        var fires: [Double] = []
        for i in 0..<400 {                       // 4 s ≈ 8 strides at 2 Hz
            let t = Double(i) * dt
            if det.feed(down: amplitude * sin(2 * .pi * f * t), at: t) { fires.append(t) }
        }
        XCTAssertEqual(fires.count, 8, "expected one fire per stride, got \(fires)")
        for t in fires {
            let phase = (t * f).truncatingRemainder(dividingBy: 1.0)
            // Fires one sample after the true peak (it needs to see the fall).
            XCTAssertEqual(phase, 0.25, accuracy: 0.04,
                           "fired at phase \(phase); the peak is 0.25, crossings are 0.0/0.5")
        }
    }

    func testStationaryJitterBelowFloorNeverFires() {
        var det = ApexDetector(peakFloor: 0.03, refractory: 0.25, smoothing: 1.0)
        var fires = 0
        for i in 0..<500 {
            let t = Double(i) * 0.01
            if det.feed(down: 0.01 * sin(2 * .pi * 3 * t), at: t) { fires += 1 }   // 0.01 g < floor
        }
        XCTAssertEqual(fires, 0)
    }

    func testRefractoryPreventsDoubleFireWithinOneStride() {
        var det = ApexDetector(peakFloor: 0.03, refractory: 0.25, smoothing: 1.0)
        // Two bumps 0.08 s apart — a jittery single stride, not two strides.
        let samples: [Double] = [0, 0.1, 0.2, 0.1, 0, 0.1, 0.2, 0.1, 0]
        var fires = 0
        for (i, v) in samples.enumerated() {
            if det.feed(down: v, at: Double(i) * 0.02) { fires += 1 }
        }
        XCTAssertEqual(fires, 1)
    }

    func testResetForgetsHistory() {
        var det = ApexDetector(peakFloor: 0.03, refractory: 10, smoothing: 1.0)
        _ = det.feed(down: 0.0, at: 0); _ = det.feed(down: 0.2, at: 0.01)
        XCTAssertTrue(det.feed(down: 0.0, at: 0.02))          // fires; refractory now armed for 10 s
        det.reset()
        _ = det.feed(down: 0.0, at: 0.03); _ = det.feed(down: 0.2, at: 0.04)
        XCTAssertTrue(det.feed(down: 0.0, at: 0.05), "reset must clear the refractory clock")
    }

    // MARK: - Sensor sign convention

    /// Free fall is the one situation with an unambiguous answer: the phone
    /// accelerates DOWNWARD at 1 g while the accelerometer reads zero, so
    /// CoreMotion reports userAcceleration = −gravity. `down` must come out
    /// positive. A detector fed the un-negated projection fires at heel-strike
    /// (the bottom of the bounce) instead of the apex, and no synthetic-signal
    /// test can see that — only this convention check can.
    func testVerticalDownIsPositiveInFreeFall() {
        let gravity = CMAcceleration(x: 0, y: 0, z: -1)         // face-up: earthward is −z
        let freeFall = CMAcceleration(x: 0, y: 0, z: 1)         // userAcceleration = −gravity
        XCTAssertEqual(CaptureEngine.verticalDown(userAcceleration: freeFall, gravity: gravity), 1.0, accuracy: 1e-9)
    }

    func testVerticalDownIsZeroAtRestAndOrientationInvariant() {
        let rest = CMAcceleration(x: 0, y: 0, z: 0)
        XCTAssertEqual(CaptureEngine.verticalDown(userAcceleration: rest, gravity: CMAcceleration(x: 0, y: 0, z: -1)), 0, accuracy: 1e-9)
        // Same free-fall sample with the phone held upright (earthward is −y).
        XCTAssertEqual(CaptureEngine.verticalDown(userAcceleration: CMAcceleration(x: 0, y: 1, z: 0),
                                                  gravity: CMAcceleration(x: 0, y: -1, z: 0)), 1.0, accuracy: 1e-9)
    }

    // MARK: - StillnessFilter

    func testStillnessRequiresSustainedQuietNotOneSample() {
        var f = StillnessFilter(smoothing: 0.033, seed: 1.0)
        // A brisk walk with one quiet sample (a zero-crossing) must NOT read as still.
        for _ in 0..<50 { _ = f.feed(0.30) }
        XCTAssertGreaterThan(f.feed(0.0), 0.08, "a single quiet sample must not count as stillness")
        // Sustained quiet does.
        var last = 1.0
        for _ in 0..<120 { last = f.feed(0.0) }       // 1.2 s
        XCTAssertLessThan(last, 0.08)
    }

    func testStillnessFilterSeedsHighSoAMovingStartIsNotStill() {
        var f = StillnessFilter()
        XCTAssertGreaterThan(f.feed(0.0), 0.08, "first sample alone must not read as still")
    }

    // MARK: - DutyCycle decision

    private func input(windowOpen: Bool = false, windowAge: TimeInterval = 0,
                       sinceCapture: TimeInterval, latched: Bool,
                       trigger: CaptureEngine.CaptureTrigger = .apogee,
                       processing: Bool = false) -> DutyCycle.Input {
        DutyCycle.Input(windowOpen: windowOpen, windowAge: windowAge, maxWindow: 1.5,
                        timeSinceCapture: sinceCapture, cooldown: 1.0, forcedInterval: 4.0,
                        apexLatched: latched, latchedTrigger: trigger, processing: processing)
    }

    func testLatchSetDuringCooldownIsConsumedNotActedOn() {
        let d = DutyCycle.decide(input(sinceCapture: 0.3, latched: true))
        XCTAssertEqual(d.action, .none)
        XCTAssertTrue(d.consumeLatch, "a latch inside the cooldown is stale and must be dropped")
    }

    func testFreshApexAfterCooldownOpensWindow() {
        let d = DutyCycle.decide(input(sinceCapture: 1.2, latched: true))
        XCTAssertEqual(d.action, .openWindow(.apogee))
        XCTAssertTrue(d.consumeLatch)
    }

    func testStillnessLatchOpensWithItsOwnTrigger() {
        let d = DutyCycle.decide(input(sinceCapture: 1.2, latched: true, trigger: .stillness))
        XCTAssertEqual(d.action, .openWindow(.stillness))
    }

    func testNoApexBeforeForcedIntervalStaysClosed() {
        let d = DutyCycle.decide(input(sinceCapture: 2.0, latched: false))
        XCTAssertEqual(d.action, .none)
        XCTAssertFalse(d.consumeLatch)
    }

    func testForcedWindowGuaranteesLiveness() {
        // No apex for longer than forcedInterval → capture anyway, so a user who
        // is standing still and panning can never starve detection.
        let d = DutyCycle.decide(input(sinceCapture: 4.5, latched: false))
        XCTAssertEqual(d.action, .openWindow(.forced))
    }

    func testProcessingBlocksOpeningAndDropsTheLatch() {
        let d = DutyCycle.decide(input(sinceCapture: 5, latched: true, processing: true))
        XCTAssertEqual(d.action, .none)
        XCTAssertTrue(d.consumeLatch)
    }

    func testExpiredWindowClosesAndDropsStaleLatch() {
        let d = DutyCycle.decide(input(windowOpen: true, windowAge: 1.6, sinceCapture: 9, latched: true))
        XCTAssertEqual(d.action, .closeWindow)
        XCTAssertTrue(d.consumeLatch)
    }

    func testOpenWindowWithinDurationIsLeftAlone() {
        let d = DutyCycle.decide(input(windowOpen: true, windowAge: 0.2, sinceCapture: 9, latched: false))
        XCTAssertEqual(d.action, .none)
        XCTAssertFalse(d.consumeLatch)
    }

    /// The original bug as a scenario. A latch set at t=0.3 s inside a 1.0 s
    /// cooldown must NOT open the window the moment the cooldown ends — that
    /// would time the capture by the cooldown clock at an arbitrary gait phase.
    func testStaleLatchCannotOpenWindowAtCooldownExpiry() {
        var latched = true
        var d = DutyCycle.decide(input(sinceCapture: 0.3, latched: latched))
        if d.consumeLatch { latched = false }
        d = DutyCycle.decide(input(sinceCapture: 1.05, latched: latched))   // no new apex since
        XCTAssertEqual(d.action, .none, "only a FRESH apex after the cooldown may open the window")
    }
}
