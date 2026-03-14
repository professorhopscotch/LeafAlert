# LeafAlert Build Plan

## Research Summary

The scaffold already has 15 Swift files with mostly real implementations. Key gaps:

1. **PatrolView.swift** — `startPatrol()`/`stopPatrol()` have TODO stubs; don't call `appState.startPatrol()`/`appState.stopPatrol()`
2. **AROverlayView.swift** — `updateUIView` has TODO for AR anchor placement; no bounding box overlay rendering
3. **project.pbxproj** — Frameworks build phase is empty (AVFoundation, Vision, CoreML, ARKit, MapKit, CoreLocation, CoreMotion not linked)
4. **Assets.xcassets** — Missing entirely (needed for app icon, accent color)
5. **.gitignore** — Missing (Xcode project needs one)
6. **Battery optimization** — Build order step 11 not done: queue QoS review, background capture limits, camera preset validation
7. **InferenceEngine** — bounding box always `.zero`; should attempt to extract from Vision observations

All models, stores, 4/6 views, all 3 engines, and the app entry point have working implementations.

---

## Work Units

### Unit 1: PatrolView Engine Wiring
**Files:** `LeafAlert/Views/PatrolView.swift`
**Change:** Replace TODO stubs in `startPatrol()` and `stopPatrol()` with actual calls to `appState.startPatrol()` and `appState.stopPatrol()`. Sync the local `isPatrolling` state with `appState.isPatrolling`. Add screen brightness management (dim on start, restore on stop). Add `onDisappear` cleanup. Ensure the detection banner reads from `appState.lastDetection`.

### Unit 2: AROverlayView Completion
**Files:** `LeafAlert/Views/AROverlayView.swift`
**Change:** Implement AR anchor placement in `updateUIView`. Given a bounding box from DetectionResult, create a RealityKit entity (colored semi-transparent box or highlight plane) positioned ~1-2m in front of the camera at the approximate direction derived from the bounding box. Add a pulsing animation to the overlay entity. Handle the case where bounding box is `.zero` by placing a general warning indicator.

### Unit 3: Project Configuration & Build Fixes
**Files:** `LeafAlert.xcodeproj/project.pbxproj`, new `LeafAlert/Assets.xcassets/`, new `.gitignore`
**Change:** Add all required system frameworks to the pbxproj frameworks build phase (AVFoundation, Vision, CoreML, ARKit, RealityKit, MapKit, CoreLocation, CoreMotion, AudioToolbox). Create Assets.xcassets with AccentColor and AppIcon stubs. Add a comprehensive .gitignore for Xcode/Swift projects. Add Assets.xcassets to the pbxproj resources phase.

### Unit 4: Battery Optimization & Inference Enhancement
**Files:** `LeafAlert/Engines/CaptureEngine.swift`, `LeafAlert/Engines/InferenceEngine.swift`, `LeafAlert/App/AppState.swift`
**Change:** Review and harden battery optimization: ensure capture queue uses `.userInitiated` QoS (already done — verify), add frame skip counter that drops frames if inference is still processing the previous one, add `minCaptureInterval` property to CaptureEngine for explicit throttling. In InferenceEngine, extract actual bounding box from VNClassificationObservation (or VNRecognizedObjectObservation if available) instead of returning `.zero`. In AppState, add `@AppStorage` observation to live-sync settings changes to engines during patrol.

### Unit 5: Launch Polish & Integration
**Files:** `LeafAlert/App/LeafAlertApp.swift`, `LeafAlert/Views/HomeView.swift`, `LeafAlert/Views/PatrolView.swift`, `LeafAlert/Views/PatrolMapView.swift`, new `LeafAlert/Resources/LaunchScreen.storyboard`
**Change:** Add the required disclaimer text ("This app assists identification — always verify visually before touching any plant") to detection banners in PatrolView and PatrolMapView detail popups. Create a minimal LaunchScreen.storyboard with the app name and leaf icon. In HomeView, add a "Recent Detections" section showing the last 3 detections from the log. Add proper `.onAppear`/`.onDisappear` lifecycle handling to views that need cleanup.

---

## E2E Test Recipe

**Skip e2e** — This is a Swift/iOS project targeting iOS 16 with ARKit/CoreML dependencies. There is no Xcode or iOS simulator available in this Linux environment. Verification is limited to:
1. Ensure all Swift files have valid syntax (no obvious typos, balanced braces)
2. Ensure all cross-file type references resolve (types used in one file are defined in another)
3. Run `swift -frontend -typecheck` if available (likely not on this system)

Workers should focus on code correctness review via the simplify skill.

---

## Worker Instructions Template

See the prompt for each agent below. Each agent gets:
- The overall goal
- Its specific unit task
- Codebase conventions (SwiftUI, @EnvironmentObject for AppState, @AppStorage for settings, ObservableObject engines)
- The e2e recipe (skip — no Xcode available)
- The standard post-implementation checklist
