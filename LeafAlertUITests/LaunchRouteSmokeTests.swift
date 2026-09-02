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
        app.launch()
        let understand = app.buttons["I Understand"]
        if understand.waitForExistence(timeout: 3) { understand.tap() }
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
}
