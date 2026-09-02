import XCTest

/// Launch-route smoke tests.
///
/// Each test launches the app directly on a screen via the DEBUG-only
/// `-launchRoute` argument and checks that the screen actually appeared. This is
/// the class of regression unit tests cannot see: a nested NavigationStack
/// silently dropping a push (the Debug and Plants routes used to land on Home),
/// or the patrol screen rendering blank on a host with no camera.
final class LaunchRouteSmokeTests: XCTestCase {

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// Launches on `route` and clears the first-run disclaimer if it is up.
    private func launch(route: String? = nil, extra: [String] = []) -> XCUIApplication {
        let app = XCUIApplication()
        var args = extra
        if let route { args += ["-launchRoute", route] }
        app.launchArguments = args
        // A fresh simulator (CI) shows the camera / location permission alerts
        // on the first patrol; accept them so no test depends on test order.
        addUIInterruptionMonitor(withDescription: "System permission alert") { alert in
            for title in ["Allow", "Allow While Using App", "OK"] where alert.buttons[title].exists {
                alert.buttons[title].tap()
                return true
            }
            return false
        }
        app.launch()
        let understand = app.buttons["I Understand"]
        if understand.waitForExistence(timeout: 3) { understand.tap() }
        // Interruption monitors only run on the next interaction; a tap on the
        // status-bar area is harmless.
        app.coordinate(withNormalizedOffset: CGVector(dx: 0.5, dy: 0.01)).tap()
        return app
    }

    func testHomeShowsThePrimaryActions() {
        let app = launch()
        XCTAssertTrue(app.buttons["Start Patrol"].waitForExistence(timeout: 10))
        XCTAssertTrue(app.buttons["Plants"].exists)
        XCTAssertTrue(app.buttons["Settings"].exists)
    }

    func testDebugRouteLandsOnTheDebugDashboard() {
        let app = launch(route: "debug")
        XCTAssertTrue(app.navigationBars["Debug"].waitForExistence(timeout: 10),
                      "the Debug push was dropped — landed somewhere else")
        XCTAssertTrue(app.staticTexts["Pipeline"].exists)
    }

    func testPlantsRouteShowsThePlantGuide() {
        let app = launch(route: "plants")
        XCTAssertTrue(app.navigationBars["Plant Guide"].waitForExistence(timeout: 10))
    }

    func testSettingsRouteShowsSettings() {
        let app = launch(route: "settings")
        XCTAssertTrue(app.navigationBars["Settings"].waitForExistence(timeout: 10))
    }

    func testPatrolWithoutACameraShowsTheUnavailableStateNotABlankScreen() {
        let app = launch(route: "patrol", extra: ["-autoStartPatrol", "1"])
        XCTAssertTrue(app.navigationBars["Patrol"].waitForExistence(timeout: 10))
        // The simulator has no camera: the screen must say so, and still offer Stop.
        XCTAssertTrue(app.staticTexts["Camera unavailable"].waitForExistence(timeout: 10),
                      "patrol rendered without the no-camera state (blank screen regression)")
        XCTAssertTrue(app.buttons["Stop Patrol"].exists)
    }

    // MARK: - Detection card (synthetic detection; the simulator has no camera)

    func testInjectedAlertShowsTheFeedbackCardAndKeepsItPastTheBoxExpiry() {
        let app = launch(route: "patrol", extra: ["-autoStartPatrol", "1", "-injectDetection", "poison_ivy:0.72"])
        let headline = app.staticTexts.containing(NSPredicate(format: "label CONTAINS[c] %@", "Likely Poison Ivy")).firstMatch
        XCTAssertTrue(headline.waitForExistence(timeout: 10))
        XCTAssertTrue(app.buttons["Correct"].exists)
        XCTAssertTrue(app.buttons["Wrong"].exists)
        // The box expires after 2.5 s; the card must NOT go with it — it used to.
        sleep(4)
        XCTAssertTrue(app.buttons["Correct"].exists, "the feedback card vanished with the box expiry")
    }

    func testCorrectionFlowSubmitsAndDismissesTheCard() {
        let app = launch(route: "patrol", extra: ["-autoStartPatrol", "1", "-injectDetection", "poison_oak:0.66"])
        XCTAssertTrue(app.buttons["Wrong"].waitForExistence(timeout: 10))
        app.buttons["Wrong"].tap()
        XCTAssertTrue(app.staticTexts["What is it?"].waitForExistence(timeout: 3))
        app.buttons["Safe Plant"].tap()
        app.buttons["Submit"].tap()
        XCTAssertTrue(app.buttons["Submit"].waitForNonExistence(timeout: 3))
        XCTAssertFalse(app.buttons["Correct"].exists)
    }

    func testUncertainDetectionUsesHedgedCopy() {
        // 0.30 is below poison_ivy's 0.40 alert bar but inside the 0.20 uncertainty margin.
        let app = launch(route: "patrol", extra: ["-autoStartPatrol", "1", "-injectDetection", "poison_ivy:0.30"])
        let headline = app.staticTexts.containing(NSPredicate(format: "label CONTAINS[c] %@", "Possible Poison Ivy")).firstMatch
        XCTAssertTrue(headline.waitForExistence(timeout: 10))
        // The card's footer always says "verify visually"; assert the SUBTITLE the
        // uncertain severity produces, so this cannot pass by accident.
        XCTAssertTrue(app.staticTexts["Low confidence · verify visually before touching"].exists)
    }

    func testMapRouteShowsTheDetectionMap() {
        let app = launch(route: "map")
        XCTAssertTrue(app.navigationBars["Detection Map"].waitForExistence(timeout: 10))
    }

    func testInjectedDetectionPersistsIntoRecentDetectionsAcrossRelaunch() {
        // Inject on patrol, then relaunch cold onto Home: the detection log store
        // must have persisted it and Home must list it.
        let first = launch(route: "patrol", extra: ["-autoStartPatrol", "1", "-injectDetection", "poison_sumac:0.80"])
        XCTAssertTrue(first.buttons["Correct"].waitForExistence(timeout: 10))
        first.terminate()
        let app = launch()
        XCTAssertTrue(app.staticTexts["Recent Detections"].waitForExistence(timeout: 10))
        let row = app.staticTexts.containing(NSPredicate(format: "label CONTAINS[c] %@", "Poison Sumac")).firstMatch
        XCTAssertTrue(row.waitForExistence(timeout: 5), "the injected detection was not persisted to the log / Home list")
    }
}
