## LeafAlert Project Status — Handoff Document

### Branch
`claude/leafalert-project-scaffold-6LIkj` on `professorhopscotch/LeafAlert`

### What Was Built
A complete iOS 16+ SwiftUI app for detecting toxic plants on hikes, fully offline-capable. **18 Swift source files** across 5 modules:

**App Layer**
- `LeafAlertApp.swift` — Entry point, SwiftData container setup
- `AppState.swift` — Combine-based state management, live settings sync via UserDefaults KVO

**Engines (AVFoundation / CoreML / ARKit)**
- `CaptureEngine.swift` — Camera frame capture with configurable intervals, frame-processing guard, capturesPerMinute diagnostics
- `InferenceEngine.swift` — CoreML inference with overlapping-call guard, performance timing, bounding box upgrade path documented
- `AlertEngine.swift` — Haptic/audio alerts via AudioToolbox

**Stores**
- `DetectionLogStore.swift` — SwiftData persistence for detection logs

**Models**
- `DetectionResult.swift` — Inference output struct
- `DetectionLog.swift` — SwiftData `@Model` with lat/lon, confidence, userConfirmed
- `PlantInfo.swift` — Static plant reference data

**Views**
- `HomeView.swift` — Landing screen, recent detections (last 3 via `@Query`), safety disclaimer
- `PatrolView.swift` — Camera patrol wired to all engines, screen brightness management
- `AROverlayView.swift` — Camera-relative AR anchor placement, pulsing animation, general warning fallback
- `PatrolMapView.swift` — Map pins with tap-to-select detail popup, confirm/dismiss, segmented filter (All/Confirmed/Unconfirmed)
- `PlantDetailView.swift` — Plant reference information
- `SettingsView.swift` — User preferences
- `DetectionFormatting.swift` — Shared helpers (plantDisplayName, confidenceColor, relativeTimestamp, safetyDisclaimer)

**Resources & Config**
- `LaunchScreen.storyboard` — Centered leaf icon + "LeafAlert" label
- `Assets.xcassets` — AccentColor (forest green) + AppIcon placeholder
- `Info.plist` — Camera, location, motion permissions
- `project.pbxproj` — All 9 frameworks linked (AVFoundation, Vision, CoreML, ARKit, RealityKit, MapKit, CoreLocation, CoreMotion, AudioToolbox)

### Build & Test Status
- **Cannot build/test in current environment** — Linux host, no Xcode/Swift toolchain
- **No test target exists** — zero XCTest files, no test target in pbxproj
- **No CI/CD** — no GitHub Actions, Fastfile, or Makefile

### What Needs to Happen Next (on macOS with Xcode)
1. `git clone` and open `LeafAlert.xcodeproj` in Xcode
2. Build with `xcodebuild -scheme LeafAlert -sdk iphonesimulator build` — fix any compile errors (likely minor: missing CoreML model file, signing config)
3. Add a CoreML model file (`.mlmodelc`) for plant detection — the `InferenceEngine` expects one but none is bundled yet
4. Create a test target and write unit tests for models, engines, and stores
5. Set up a CI pipeline (GitHub Actions recommended)

### Known Gaps
| Area | Detail |
|------|--------|
| ML Model | No `.mlmodel` file included — needs training or a pre-trained model |
| Tests | Zero test coverage — needs XCTest target + unit tests |
| CI/CD | No pipeline configured |
| Code Signing | Set to "Automatic" — needs team/profile on real device builds |
| App Icon | Placeholder only — needs actual artwork |
| Offline Map Tiles | MapKit uses default Apple tiles (requires network for first load) |
