import AVFoundation
import CoreMotion
import Combine

/// Motion state at the moment a frame was captured.
struct CaptureContext {
    let netAcceleration: Double
    let rotationRate: Double
    let trigger: CaptureEngine.CaptureTrigger
    let timestamp: Date
}

// MARK: - Pure duty-cycle logic (hardware-free, unit-tested)

/// Detects the stride APEX from the signed vertical acceleration stream.
///
/// Let `down` = −(userAcceleration · gravity). CoreMotion's `gravity` points
/// toward the earth, but `userAcceleration` is the accelerometer RESIDUAL, which
/// is the NEGATIVE of kinematic acceleration: in free fall the accelerometer
/// reads zero, so userAcceleration = −gravity while the phone accelerates
/// downward at 1 g. Hence the negation — `down` is positive while the phone
/// accelerates DOWNWARD. See `CaptureEngine.verticalDown`, pinned by a test.
///
/// Over one walking bounce the body's vertical velocity is zero at the two
/// turnarounds — the apex (top) and heel-strike (bottom) — and maximal in
/// between. A zero-crossing of the ACCELERATION therefore marks maximum vertical
/// speed: the blurriest instant, and exactly where the previous detector fired.
/// The apex is where `down` PEAKS (support force is lowest at the top of the
/// bounce); heel-strike is where it TROUGHS (impact). We want the apex: it is a
/// turnaround (zero velocity) and, unlike heel-strike, it is smooth.
///
/// So this is a local-MAXIMUM detector on a lightly smoothed `down`, with a
/// magnitude floor to ignore sensor jitter while stationary and a refractory
/// period so one stride cannot fire twice.
///
/// Physics-derived; validate in the field against the DataRecorder IMU log.
struct ApexDetector {
    /// Peaks below this (g) are treated as jitter, not a stride.
    var peakFloor: Double
    /// Minimum time between fires (s). Brisk walking is ~2 Hz; 4 Hz is a hard ceiling.
    var refractory: TimeInterval
    /// EMA weight for the new sample; ~2-3 samples of lag at 100 Hz.
    var smoothing: Double

    private var smoothed: Double = 0
    private var prev: Double = 0
    private var prevDelta: Double = 0
    private var lastFire: TimeInterval = -.infinity
    private var samples = 0

    init(peakFloor: Double = 0.03, refractory: TimeInterval = 0.25, smoothing: Double = 0.35) {
        self.peakFloor = peakFloor
        self.refractory = refractory
        self.smoothing = smoothing
    }

    /// Feed one sample. Returns `true` exactly once per detected apex.
    mutating func feed(down: Double, at t: TimeInterval) -> Bool {
        smoothed = samples == 0 ? down : smoothing * down + (1 - smoothing) * smoothed
        samples += 1
        let delta = smoothed - prev
        // Rising then falling: the previous sample was a local maximum.
        let isPeak = prevDelta > 0 && delta <= 0 && prev >= peakFloor
        prev = smoothed
        prevDelta = delta
        guard isPeak, t - lastFire >= refractory else { return false }
        lastFire = t
        return true
    }

    mutating func reset() {
        smoothed = 0; prev = 0; prevDelta = 0; lastFire = -.infinity; samples = 0
    }
}

/// Smooths |userAcceleration| so "still" means sustained quiet, not one sample.
///
/// During slow walking the vertical acceleration passes through zero twice per
/// stride, so a single-sample test (|ua| < threshold) fired on those crossings
/// and opened the window at an arbitrary gait phase — labelled "stillness". An
/// EMA with a ~0.3 s time constant (weight 0.033 at 100 Hz) only drops below the
/// threshold when the phone has actually been quiet for a while. It is seeded
/// high so a moving start cannot be mistaken for still.
struct StillnessFilter {
    var smoothing: Double
    private var ema: Double
    private let seed: Double

    init(smoothing: Double = 0.033, seed: Double = 1.0) {
        self.smoothing = smoothing
        self.seed = seed
        self.ema = seed
    }

    /// Feed one |userAcceleration| sample; returns the smoothed magnitude.
    mutating func feed(_ magnitude: Double) -> Double {
        ema = smoothing * magnitude + (1 - smoothing) * ema
        return ema
    }

    mutating func reset() { ema = seed }
}

/// The duty-cycle decision, factored out of the engine so it can be tested
/// without a camera or motion sensors.
enum DutyCycle {
    enum Action: Equatable {
        case none
        case closeWindow
        case openWindow(CaptureEngine.CaptureTrigger)
    }

    struct Input {
        var windowOpen: Bool
        var windowAge: TimeInterval
        var maxWindow: TimeInterval
        var timeSinceCapture: TimeInterval
        var cooldown: TimeInterval
        var forcedInterval: TimeInterval
        var apexLatched: Bool
        var latchedTrigger: CaptureEngine.CaptureTrigger
        var processing: Bool
    }

    struct Decision: Equatable {
        var action: Action
        /// Whether the apex latch should be cleared. A latch set during the
        /// cooldown is STALE by the time the cooldown ends — acting on it would
        /// open the window at an arbitrary gait phase, on the cooldown clock.
        /// Consuming it means only a FRESH apex after the cooldown can trigger.
        var consumeLatch: Bool
    }

    static func decide(_ i: Input) -> Decision {
        if i.windowOpen {
            // A window that expires without a usable frame closes; any latch that
            // accrued meanwhile is stale for the same reason as below.
            return i.windowAge >= i.maxWindow
                ? Decision(action: .closeWindow, consumeLatch: true)
                : Decision(action: .none, consumeLatch: false)
        }
        if i.timeSinceCapture < i.cooldown || i.processing {
            return Decision(action: .none, consumeLatch: true)
        }
        if i.apexLatched {
            return Decision(action: .openWindow(i.latchedTrigger), consumeLatch: true)
        }
        if i.timeSinceCapture >= i.forcedInterval {
            return Decision(action: .openWindow(.forced), consumeLatch: false)
        }
        return Decision(action: .none, consumeLatch: false)
    }
}

/// Manages camera capture with motion-timed inference and duty-cycled power use.
///
/// # What is actually gated
///
/// The AVCaptureSession runs continuously while patrolling — the live preview
/// needs it, and start/stop latency (hundreds of ms) would defeat apex timing.
/// What the duty cycle gates is INFERENCE: the expensive per-frame work (CoreML,
/// JPEG encode, saliency, disk). Between capture windows, frames flow only to the
/// preview and to DataRecorder. Battery Saver additionally throttles the sensor
/// frame rate between windows.
///
/// # Timing
///
///   1. Window closed — accelerometer (~1 mW) runs at 100 Hz; frames are not inferred.
///   2. Stride apex (or stillness) detected after the cooldown → window opens.
///   3. First frame that passes the rotation gate is inferred → window closes,
///      cooldown begins. A window that gets no usable frame within
///      `maxCaptureWindowDuration` closes on its own.
///   4. If no apex arrives within `forcedCaptureInterval`, a forced window opens
///      so detection never silently stalls (e.g. standing still while panning).
///
/// # Apex detection
///
/// See `ApexDetector`. The apex is the top of the walking bounce — a velocity
/// turnaround with the least blur. It is a local maximum of the downward
/// acceleration, NOT a zero-crossing (which is the point of maximum speed).
///
/// At 100 Hz motion + 30 fps video, worst-case latency from apex to inferred
/// frame is ~50 ms (duty-cycle poll) + ~33 ms (next frame).
final class CaptureEngine: NSObject, ObservableObject {

    enum CaptureTrigger: String { case apogee, stillness, forced }

    /// Camera authorization state, for the UI to surface an "access needed" message.
    enum CameraPermission { case unknown, authorized, denied }

    /// Downward kinematic acceleration in g, from a CoreMotion sample.
    ///
    /// CoreMotion's `userAcceleration` is the accelerometer residual — total
    /// minus gravity — and an accelerometer measures PROPER acceleration, which is
    /// the negative of kinematic acceleration along gravity: in free fall it reads
    /// zero, so userAcceleration = −gravity while the phone accelerates downward
    /// at 1 g. Projecting onto the (earthward, unit-length) gravity vector and
    /// negating therefore yields "positive = accelerating toward the earth".
    static func verticalDown(userAcceleration ua: CMAcceleration, gravity g: CMAcceleration) -> Double {
        -(ua.x * g.x + ua.y * g.y + ua.z * g.z)
    }

    // MARK: - Published State

    @Published private(set) var isRunning = false

    /// Current camera authorization state. When `.denied`, the UI should prompt the
    /// user to enable camera access in Settings — the capture pipeline cannot run.
    @Published private(set) var cameraPermission: CameraPermission = .unknown

    /// Whether a camera input actually came up when the patrol started. False
    /// when the device has no usable back camera (in use by another app, a
    /// hardware fault, or the simulator). The UI must say so rather than show a
    /// dead preview behind a green "Patrolling" pill.
    @Published private(set) var cameraAvailable = true

    /// Number of frames captured in the current 60-second window, for UI diagnostics.
    @Published private(set) var capturesPerMinute: Int = 0

    /// Whether the capture pipeline is active (for UI heartbeat indicator).
    @Published private(set) var pipelineActive = false

    /// Current net acceleration (|a| - 1g), updated at 100 Hz. For debug display.
    @Published private(set) var currentNetAccel: Double = 0.0

    /// Whether the inference window is open (duty cycle). For debug display.
    @Published private(set) var isCameraActive = false

    /// Total apogee events detected since patrol started.
    @Published var apogeeCount: Int = 0

    /// Total frames captured since patrol started.
    @Published var totalFramesCaptured: Int = 0

    /// Current rotation rate magnitude (rad/s). For debug display.
    @Published private(set) var currentRotationRate: Double = 0.0

    // MARK: - Configuration (thread-safe via configLock)

    private let configLock = NSLock()

    /// Documented defaults for the live-tunable gates. The debug dashboard shows
    /// these beside each slider and can restore them. They live in memory only —
    /// never persisted — so every app launch starts from here.
    enum TuningDefaults {
        static let minCaptureInterval: TimeInterval = 1.0
        static let batterySaverInterval: TimeInterval = 4.0
        static let maxCaptureWindowDuration: TimeInterval = 1.5
        static let forcedCaptureInterval: TimeInterval = 4.0
        static let stillnessThreshold: Double = 0.08
        static let rotationRateThreshold: Double = 1.5   // rad/s ≈ 86°/s
    }

    /// Restores every live-tunable gate to its documented default, so a debug
    /// tweak cannot linger into a real patrol without an obvious way back.
    func resetTuningToDefaults() {
        minCaptureInterval = TuningDefaults.minCaptureInterval
        batterySaverInterval = TuningDefaults.batterySaverInterval
        maxCaptureWindowDuration = TuningDefaults.maxCaptureWindowDuration
        forcedCaptureInterval = TuningDefaults.forcedCaptureInterval
        stillnessThreshold = TuningDefaults.stillnessThreshold
        rotationRateThreshold = TuningDefaults.rotationRateThreshold
    }

    private var _minCaptureInterval: TimeInterval = TuningDefaults.minCaptureInterval
    private var _batterySaverInterval: TimeInterval = TuningDefaults.batterySaverInterval
    private var _maxCaptureWindowDuration: TimeInterval = TuningDefaults.maxCaptureWindowDuration
    private var _forcedCaptureInterval: TimeInterval = TuningDefaults.forcedCaptureInterval
    private var _stillnessThreshold: Double = TuningDefaults.stillnessThreshold
    private var _rotationRateThreshold: Double = TuningDefaults.rotationRateThreshold
    private var _isBatterySaverEnabled = false

    /// Minimum interval between captures (cooldown). Camera stays off during this period.
    var minCaptureInterval: TimeInterval {
        get { configLock.lock(); defer { configLock.unlock() }; return _minCaptureInterval }
        set { configLock.lock(); defer { configLock.unlock() }; _minCaptureInterval = newValue }
    }

    /// Minimum interval between captures when in battery-saver mode.
    var batterySaverInterval: TimeInterval {
        get { configLock.lock(); defer { configLock.unlock() }; return _batterySaverInterval }
        set { configLock.lock(); defer { configLock.unlock() }; _batterySaverInterval = newValue }
    }

    /// Maximum time the camera stays powered on waiting for a good frame.
    var maxCaptureWindowDuration: TimeInterval {
        get { configLock.lock(); defer { configLock.unlock() }; return _maxCaptureWindowDuration }
        set { configLock.lock(); defer { configLock.unlock() }; _maxCaptureWindowDuration = newValue }
    }

    /// Maximum time between forced capture attempts regardless of apogee detection.
    var forcedCaptureInterval: TimeInterval {
        get { configLock.lock(); defer { configLock.unlock() }; return _forcedCaptureInterval }
        set { configLock.lock(); defer { configLock.unlock() }; _forcedCaptureInterval = newValue }
    }

    /// Net acceleration (g) below which the device is considered "still enough" to
    /// trigger a capture window even without an apogee zero-crossing.
    var stillnessThreshold: Double {
        get { configLock.lock(); defer { configLock.unlock() }; return _stillnessThreshold }
        set { configLock.lock(); defer { configLock.unlock() }; _stillnessThreshold = newValue }
    }

    /// Maximum rotation rate (rad/s) allowed for a capture. If the device is rotating
    /// faster than this, apogee/stillness events are suppressed to avoid motion blur.
    /// ~1.5 rad/s ≈ 86°/s — a moderate pan speed.
    var rotationRateThreshold: Double {
        get { configLock.lock(); defer { configLock.unlock() }; return _rotationRateThreshold }
        set { configLock.lock(); defer { configLock.unlock() }; _rotationRateThreshold = newValue }
    }

    /// Whether battery saver mode is active.
    var isBatterySaverEnabled: Bool {
        get { configLock.lock(); defer { configLock.unlock() }; return _isBatterySaverEnabled }
        set { configLock.lock(); defer { configLock.unlock() }; _isBatterySaverEnabled = newValue }
    }

    // MARK: - Callbacks

    /// Called on a background queue when a frame is ready for inference.
    var onFrameCaptured: ((CVPixelBuffer, CaptureContext) -> Void)?

    /// Exposed for CameraPreviewView to attach a preview layer.
    var session: AVCaptureSession { captureSession }

    // MARK: - Private Properties

    private let captureSession = AVCaptureSession()
    private let motionManager = CMMotionManager()

    /// Serial delivery queue for CoreMotion. A bare `OperationQueue()` is
    /// CONCURRENT, so under load consecutive samples could be handled out of
    /// order and the apex detector (a sequential peak finder) would see a
    /// scrambled series. One lane, high QoS, so timing stays faithful.
    private let motionQueue: OperationQueue = {
        let q = OperationQueue()
        q.name = "com.leafalert.motion"
        q.maxConcurrentOperationCount = 1
        q.qualityOfService = .userInteractive
        return q
    }()
    private let captureQueue = DispatchQueue(label: "com.leafalert.capture", qos: .userInitiated)
    private var lastCaptureTime: Date = .distantPast
    private var videoOutput: AVCaptureVideoDataOutput?

    // -- Apogee detection state (motion update queue only) --

    /// Previous net acceleration sample for zero-crossing detection.
    /// Written on the motion queue, read on captureQueue — guarded by motionLock.
    private var _previousNetAccel: Double = 0.0

    private var previousNetAccel: Double {
        get { motionLock.lock(); defer { motionLock.unlock() }; return _previousNetAccel }
        set { motionLock.lock(); defer { motionLock.unlock() }; _previousNetAccel = newValue }
    }

    /// Stride-apex detector. Fed on the motion queue; guarded by motionLock so
    /// `stop()` can reset it from another thread.
    private var _apexDetector = ApexDetector()

    /// Smoothed |userAcceleration| for the stillness test (see StillnessFilter).
    /// Motion queue only; guarded by motionLock alongside the apex detector.
    private var _stillness = StillnessFilter()

    /// Current rotation rate magnitude, updated on the motion queue.
    /// Read by duty cycle via motionLock.
    private var _currentRotRate: Double = 0.0

    private var currentRotRate: Double {
        get { motionLock.lock(); defer { motionLock.unlock() }; return _currentRotRate }
        set { motionLock.lock(); defer { motionLock.unlock() }; _currentRotRate = newValue }
    }

    // -- Duty cycle state --

    /// Thread-safe flag: motion updates set true at apogee/stillness, captureQueue reads & resets.
    private let motionLock = NSLock()
    /// The motion latch: "a capture-worthy instant happened; open a window".
    /// Written by the motion handler, consumed by the duty cycle. All three
    /// fields change together under ONE lock acquisition and are read as one
    /// snapshot, so a consumer can never pair a fresh flag with a stale trigger
    /// or apex timestamp (which mislabelled captures and logged the previous
    /// stride's apex_ts).
    private struct MotionLatch {
        var armed = false
        var trigger: CaptureTrigger = .forced
        /// CoreMotion timestamp (device uptime clock) of the apex that armed the
        /// latch; logged with the frame's presentation timestamp so the
        /// apex→shutter phase can be measured offline (scripts/gait_check.py).
        var apexTimestamp: TimeInterval = 0
    }
    private var _latch = MotionLatch()

    private var latchSnapshot: MotionLatch {
        motionLock.lock(); defer { motionLock.unlock() }
        return _latch
    }

    private func arm(_ trigger: CaptureTrigger, apexTimestamp: TimeInterval? = nil) {
        motionLock.lock(); defer { motionLock.unlock() }
        _latch.armed = true
        _latch.trigger = trigger
        if let apexTimestamp { _latch.apexTimestamp = apexTimestamp }
    }

    /// Arms only if nothing is latched — an apex is never relabelled as stillness.
    private func armIfIdle(_ trigger: CaptureTrigger) {
        motionLock.lock(); defer { motionLock.unlock() }
        guard !_latch.armed else { return }
        _latch.armed = true
        _latch.trigger = trigger
    }

    private func disarmLatch() {
        motionLock.lock(); defer { motionLock.unlock() }
        _latch.armed = false
    }

    /// Whether the inference window is open (ready to capture a frame for ML).
    /// Accessed only on `captureQueue`.
    private var isInferenceWindowOpen = false

    /// When the current inference window was opened. Used to enforce maxCaptureWindowDuration.
    /// Accessed only on `captureQueue`.
    private var captureWindowOpenedAt: Date = .distantPast

    /// The apex that opened the current window (nil for stillness/forced).
    /// Accessed only on `captureQueue`.
    private var windowApexTimestamp: TimeInterval?

    /// Guards against sending a new frame while the previous one is still being processed.
    /// Accessed only on `captureQueue`.
    private var isProcessingFrame = false

    /// What triggered the current capture window. Set in manageDutyCycle, read in captureOutput.
    /// Accessed only on `captureQueue`.
    private var lastTrigger: CaptureTrigger = .forced

    /// Counter for the current 60-second window (accessed on captureQueue).
    private var frameCountInWindow: Int = 0

    /// Publishes `capturesPerMinute` once a minute. Dispatch timer on captureQueue.
    private var diagnosticsTimer: DispatchSourceTimer?

    /// Drives the duty cycle at 20 Hz. Dispatch timer on captureQueue (see
    /// `startDutyCycleTimer` for why it is not a run-loop Timer).
    private var dutyCycleTimer: DispatchSourceTimer?

    // MARK: - Lifecycle

    /// Whether the capture session has been configured (inputs/outputs added).
    private var isConfigured = false

    func start() {
        guard !isRunning else { return }

        // Gate startup on camera authorization. We never silently fail: a denial is
        // published so the UI can prompt the user to enable access in Settings.
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            beginRunning()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                guard let self else { return }
                DispatchQueue.main.async {
                    if granted {
                        self.cameraPermission = .authorized
                        self.beginRunning()
                    } else {
                        self.cameraPermission = .denied
                    }
                }
            }
        case .denied, .restricted:
            DispatchQueue.main.async { self.cameraPermission = .denied }
        @unknown default:
            DispatchQueue.main.async { self.cameraPermission = .denied }
        }
    }

    /// Performs the actual session/motion startup once camera access is confirmed.
    /// Must be called on the main thread (callers dispatch as needed).
    private func beginRunning() {
        guard !isRunning else { return }
        cameraPermission = .authorized

        if !isConfigured {
            // Only latch as configured if the session actually accepted the camera
            // input/output. A failure here leaves isConfigured false so a later
            // start() (e.g. after the user grants access) can reconfigure.
            isConfigured = configureCaptureSession()
        }

        startMotionUpdates()
        startDiagnosticsTimer()
        startDutyCycleTimer()
        observeSessionInterruptions()

        // Mark as running before dispatching to prevent stop() racing ahead.
        isRunning = true
        DispatchQueue.main.async { self.pipelineActive = true }

        // Publish whether a camera actually came up so the UI can say so.
        let available = isConfigured
        DispatchQueue.main.async { self.cameraAvailable = available }

        // Start the session (keeps it warm) but camera output starts disabled.
        // Never run an input-less session: there is nothing for it to capture,
        // and an unconfigured session behaves unpredictably across hosts.
        guard isConfigured else { return }
        captureQueue.async { [weak self] in
            guard let self else { return }
            // If stop() was called before this block ran, bail out.
            guard self.isRunning else { return }
            self.captureSession.startRunning()
        }
    }

    func stop() {
        guard isRunning else { return }
        // Set isRunning false immediately so the pending startRunning() block will bail out.
        isRunning = false
        motionManager.stopDeviceMotionUpdates()
        diagnosticsTimer?.cancel()
        diagnosticsTimer = nil
        dutyCycleTimer?.cancel()
        dutyCycleTimer = nil
        // Drop any latch and detector history so the next patrol cannot open a
        // window on a stale apex from this one.
        disarmLatch()
        motionLock.lock(); _apexDetector.reset(); _stillness.reset(); motionLock.unlock()
        removeSessionObservers()

        captureQueue.async { [weak self] in
            guard let self else { return }
            self.setInferenceWindow(open: false)
            self.captureSession.stopRunning()
        }
        DispatchQueue.main.async {
            self.pipelineActive = false
            self.isCameraActive = false
            self.currentNetAccel = 0
        }
    }

    // MARK: - Camera Power (Duty Cycle)

    /// Opens or closes the inference window. When open, the next frame from the
    /// delegate will be forwarded to `onFrameCaptured` for ML processing.
    /// The video connection stays enabled at all times so the preview layer always
    /// receives frames.
    /// Must be called on `captureQueue`.
    private func setInferenceWindow(open: Bool) {
        if open && !isInferenceWindowOpen {
            isInferenceWindowOpen = true
            captureWindowOpenedAt = Date()
            updateFrameRateForWindow(open: true)
            DispatchQueue.main.async { self.isCameraActive = true }
        } else if !open && isInferenceWindowOpen {
            isInferenceWindowOpen = false
            updateFrameRateForWindow(open: false)
            DispatchQueue.main.async { self.isCameraActive = false }
        }
    }

    // MARK: - Sensor frame rate (Battery Saver)

    /// Sensor frame rate between capture windows when Battery Saver is on. The
    /// preview gets choppier, which is the trade the user opted into; inside a
    /// cooldown ends the sensor returns to full rate (see manageDutyCycle) —
    /// before any window can open — so the apex frame is taken at full cadence.
    private static let batterySaverIdleFPS: Double = 10

    /// captureQueue-only. Whether WE lowered the sensor rate. Tracked explicitly
    /// so the rate is always restored on the next window even if Battery Saver
    /// was switched off in between — otherwise a throttled sensor stayed at
    /// 10 fps for the life of the process.
    private var isThrottled = false
    private var savedMinFrameDuration = CMTime.invalid
    private var savedMaxFrameDuration = CMTime.invalid

    private var captureDevice: AVCaptureDevice? {
        captureSession.inputs.compactMap { ($0 as? AVCaptureDeviceInput)?.device }.first
    }

    /// The only real power lever this engine has: the session must keep running
    /// for the preview, but the sensor does not have to run at 30 fps while we
    /// are not going to infer anything. Called on `captureQueue`.
    private func updateFrameRateForWindow(open: Bool) {
        if open {
            // Stay "throttled" until the restore actually succeeds: clearing the
            // flag after a failed restore made the next throttle save the 10 fps
            // durations as the baseline and pinned the sensor there for good.
            if isThrottled, restoreFrameRate() {
                isThrottled = false
            }
        } else if isBatterySaverEnabled && !isThrottled {
            isThrottled = throttleFrameRate(to: Self.batterySaverIdleFPS)
        }
    }

    /// Lowers the sensor rate, remembering the device's current durations so
    /// `restoreFrameRate` puts back exactly what was there. Returns false if the
    /// active format cannot run at `fps` or the device could not be locked; any
    /// failure leaves the rate unchanged — this is an optimisation only.
    private func throttleFrameRate(to fps: Double) -> Bool {
        guard let device = captureDevice,
              device.activeFormat.videoSupportedFrameRateRanges
                .contains(where: { $0.minFrameRate <= fps && fps <= $0.maxFrameRate })
        else { return false }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            // The baseline is sampled once and kept until a restore succeeds.
            if !savedMinFrameDuration.isValid || !savedMaxFrameDuration.isValid {
                savedMinFrameDuration = device.activeVideoMinFrameDuration
                savedMaxFrameDuration = device.activeVideoMaxFrameDuration
            }
            let duration = CMTime(value: 1, timescale: CMTimeScale(fps.rounded()))
            device.activeVideoMinFrameDuration = duration
            device.activeVideoMaxFrameDuration = duration
            return true
        } catch {
            return false
        }
    }

    /// Returns true only if the saved baseline was put back on the device.
    private func restoreFrameRate() -> Bool {
        guard let device = captureDevice,
              savedMinFrameDuration.isValid, savedMaxFrameDuration.isValid else { return false }
        do {
            try device.lockForConfiguration()
            defer { device.unlockForConfiguration() }
            device.activeVideoMinFrameDuration = savedMinFrameDuration
            device.activeVideoMaxFrameDuration = savedMaxFrameDuration
            savedMinFrameDuration = .invalid
            savedMaxFrameDuration = .invalid
            return true
        } catch {
            return false   // keep the flag and the baseline; retry on the next window
        }
    }

    // MARK: - Private Setup

    /// Configures the capture session's inputs/outputs.
    /// - Returns: `true` if the camera input was added successfully; `false` otherwise
    ///   so the caller does not latch `isConfigured` on a dead session.
    @discardableResult
    private func configureCaptureSession() -> Bool {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480

        // Prefer the ultra-wide camera for maximum field of view; fall back to wide angle.
        let camera: AVCaptureDevice? =
            AVCaptureDevice.default(.builtInUltraWideCamera, for: .video, position: .back)
            ?? AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)

        guard
            let camera,
            let input = try? AVCaptureDeviceInput(device: camera),
            captureSession.canAddInput(input)
        else {
            captureSession.commitConfiguration()
            return false
        }
        captureSession.addInput(input)

        let output = AVCaptureVideoDataOutput()
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: captureQueue)
        if captureSession.canAddOutput(output) {
            captureSession.addOutput(output)
        }
        videoOutput = output

        captureSession.commitConfiguration()
        return true
    }

    /// Polls device motion at 100 Hz for stride-apex and rotation-rate detection.
    ///
    /// `down` = −(userAcceleration · gravity), positive = accelerating toward the
    /// earth — see `verticalDown` for why the negation is required. The stride
    /// apex is a local MAXIMUM of `down` — see `ApexDetector` for why a
    /// zero-crossing is the wrong event. The unsigned magnitude of
    /// `userAcceleration`, smoothed by `StillnessFilter`, drives the stillness
    /// test; the raw magnitude is recorded as telemetry.
    ///
    /// The rotation-rate check here is only a cheap pre-filter on the latch; the
    /// gate that actually rejects blurry frames runs at capture time in
    /// `captureOutput`, against the rotation rate at that instant.
    private func startMotionUpdates() {
        guard motionManager.isDeviceMotionAvailable else {
            // No motion sensors (simulator) — keep inference window always open.
            captureQueue.async { [weak self] in self?.setInferenceWindow(open: true) }
            return
        }
        motionManager.deviceMotionUpdateInterval = 0.01  // 100 Hz
        motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] motion, _ in
            guard let self, let motion else { return }

            // Forward every IMU sample to the recorder (no-op when not recording)
            DataRecorder.shared.appendIMUSample(motion)

            let ua = motion.userAcceleration
            // Unsigned motion intensity — stillness test + telemetry.
            let netAccel = sqrt(ua.x * ua.x + ua.y * ua.y + ua.z * ua.z)
            // Signed downward kinematic acceleration (see `verticalDown` for the
            // sign convention — userAcceleration is the accelerometer residual).
            let down = Self.verticalDown(userAcceleration: ua, gravity: motion.gravity)

            let rr = motion.rotationRate
            let rotRate = sqrt(rr.x * rr.x + rr.y * rr.y + rr.z * rr.z)
            self.currentRotRate = rotRate
            let rotationTooFast = rotRate > self.rotationRateThreshold

            let (isApex, smoothedMagnitude): (Bool, Double) = {
                self.motionLock.lock(); defer { self.motionLock.unlock() }
                return (self._apexDetector.feed(down: down, at: motion.timestamp),
                        self._stillness.feed(netAccel))
            }()
            // Sustained quiet, not a single quiet sample (see StillnessFilter).
            let isStill = smoothedMagnitude < self.stillnessThreshold

            // Latch a capture trigger. The capture-time gate makes the final call.
            // An apex takes precedence: a stillness sample must not relabel a
            // latched apex, or the capture log misattributes the trigger.
            if !rotationTooFast {
                if isApex {
                    self.arm(.apogee, apexTimestamp: motion.timestamp)
                    DispatchQueue.main.async { self.apogeeCount += 1 }
                } else if isStill {
                    self.armIfIdle(.stillness)
                }
            }

            // Throttle UI updates to ~10 Hz (every 10th sample at 100 Hz)
            if Int.random(in: 0..<10) == 0 {
                DispatchQueue.main.async {
                    self.currentNetAccel = netAccel
                    self.currentRotationRate = rotRate
                }
            }

            self.previousNetAccel = netAccel
        }
    }

    /// Runs the duty cycle at 20 Hz on `captureQueue` via a DispatchSourceTimer.
    ///
    /// Deliberately NOT a run-loop `Timer`: those fire only in `.default` mode, so
    /// they stall for as long as the user is scrolling any list (the run loop sits
    /// in `.tracking`) — which silently halted capture on the very debug screens
    /// used to inspect it. A dispatch timer on our own queue has no such
    /// dependency and cannot be delayed by main-thread work.
    private func startDutyCycleTimer() {
        dutyCycleTimer?.cancel()
        let timer = DispatchSource.makeTimerSource(queue: captureQueue)
        timer.schedule(deadline: .now() + 0.05, repeating: 0.05, leeway: .milliseconds(5))
        timer.setEventHandler { [weak self] in
            guard let self, self.isRunning else { return }
            self.manageDutyCycle()
        }
        timer.resume()
        dutyCycleTimer = timer
    }

    /// Core duty cycle logic. Called on `captureQueue`.
    private func manageDutyCycle() {
        let now = Date()
        let cooldown = isBatterySaverEnabled ? batterySaverInterval : minCaptureInterval

        let timeSinceCapture = now.timeIntervalSince(lastCaptureTime)

        // Battery Saver: put the sensor back to full rate as soon as a window
        // COULD open (cooldown over), not when it does — otherwise the apex
        // frame inherits the 10 fps cadence and arrives up to ~100 ms late.
        if !isInferenceWindowOpen, timeSinceCapture >= cooldown, isThrottled, restoreFrameRate() {
            isThrottled = false
        }

        let latch = latchSnapshot
        let decision = DutyCycle.decide(DutyCycle.Input(
            windowOpen: isInferenceWindowOpen,
            windowAge: now.timeIntervalSince(captureWindowOpenedAt),
            maxWindow: maxCaptureWindowDuration,
            timeSinceCapture: timeSinceCapture,
            cooldown: cooldown,
            forcedInterval: forcedCaptureInterval,
            apexLatched: latch.armed,
            latchedTrigger: latch.trigger,
            processing: isProcessingFrame
        ))

        if decision.consumeLatch { disarmLatch() }

        switch decision.action {
        case .none:
            break
        case .closeWindow:
            setInferenceWindow(open: false)
        case .openWindow(let trigger):
            lastTrigger = trigger
            windowApexTimestamp = trigger == .apogee ? latch.apexTimestamp : nil
            setInferenceWindow(open: true)
        }
    }

    private func startDiagnosticsTimer() {
        diagnosticsTimer?.cancel()
        let timer = DispatchSource.makeTimerSource(queue: captureQueue)
        timer.schedule(deadline: .now() + 60, repeating: 60)
        timer.setEventHandler { [weak self] in
            guard let self else { return }
            let count = self.frameCountInWindow
            self.frameCountInWindow = 0
            DispatchQueue.main.async { self.capturesPerMinute = count }
        }
        timer.resume()
        diagnosticsTimer = timer
    }

    // MARK: - Session Interruption Handling

    private func observeSessionInterruptions() {
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(sessionWasInterrupted),
            name: .AVCaptureSessionWasInterrupted,
            object: captureSession
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(sessionInterruptionEnded),
            name: .AVCaptureSessionInterruptionEnded,
            object: captureSession
        )
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(sessionRuntimeError),
            name: .AVCaptureSessionRuntimeError,
            object: captureSession
        )
    }

    private func removeSessionObservers() {
        NotificationCenter.default.removeObserver(self, name: .AVCaptureSessionWasInterrupted, object: captureSession)
        NotificationCenter.default.removeObserver(self, name: .AVCaptureSessionInterruptionEnded, object: captureSession)
        NotificationCenter.default.removeObserver(self, name: .AVCaptureSessionRuntimeError, object: captureSession)
    }

    @objc private func sessionWasInterrupted(_ notification: Notification) {
        print("[CaptureEngine] Session interrupted")
        DispatchQueue.main.async { self.pipelineActive = false }
    }

    @objc private func sessionInterruptionEnded(_ notification: Notification) {
        guard isConfigured else { return }
        print("[CaptureEngine] Interruption ended — restarting session")
        captureQueue.async { [weak self] in
            guard let self else { return }
            if !self.captureSession.isRunning {
                self.captureSession.startRunning()
            }
        }
        DispatchQueue.main.async { self.pipelineActive = true }
    }

    @objc private func sessionRuntimeError(_ notification: Notification) {
        // Only restart a session that actually has a camera input. Restarting an
        // input-less session just posts another runtime error — a tight loop that
        // burned CPU on hosts with no camera, and whose notification traffic
        // surfaced as periodic "Publishing changes from within view updates"
        // runtime issues (36 per launch with the loop; zero without it).
        guard isConfigured else {
            print("[CaptureEngine] Runtime error on an unconfigured session — not restarting")
            return
        }
        print("[CaptureEngine] Runtime error — attempting restart")
        captureQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
    }

    // MARK: - Public

    /// Called by the consumer to signal that frame processing is complete.
    /// Safe to call from any queue — the flag is reset on `captureQueue`.
    func markFrameProcessingComplete() {
        captureQueue.async { [weak self] in
            self?.isProcessingFrame = false
        }
    }
}

// MARK: - AVCaptureVideoDataOutputSampleBufferDelegate

extension CaptureEngine: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        // Forward every frame to the recorder (no-op when not recording).
        DataRecorder.shared.appendVideoFrame(sampleBuffer)

        // Only forward frames for inference when the duty cycle window is open.
        guard isInferenceWindowOpen else { return }
        guard !isProcessingFrame else { return }

        // Quality gate at the instant of capture. The latch-time rotation check
        // can be stale by the time a frame arrives, so re-check now and, if the
        // phone is turning fast enough to smear the frame, skip it and wait for
        // the next one in this window. Forced windows are exempt: they exist to
        // guarantee liveness, so continuous panning cannot starve detection.
        if lastTrigger != .forced && currentRotRate > rotationRateThreshold {
            return
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        // Got a frame — capture it and close the inference window.
        isProcessingFrame = true
        lastCaptureTime = Date()
        frameCountInWindow += 1
        DispatchQueue.main.async { self.totalFramesCaptured += 1 }
        setInferenceWindow(open: false)

        let context = CaptureContext(
            netAcceleration: previousNetAccel,
            rotationRate: currentRotRate,
            trigger: lastTrigger,
            timestamp: Date()
        )
        // `pts` and `apex_ts` are both on the device uptime clock, so their
        // difference is the apex→shutter phase, free of delivery latency.
        let pts = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        var details = "trigger=\(lastTrigger.rawValue) accel=\(String(format: "%.3f", previousNetAccel)) rot=\(String(format: "%.3f", currentRotRate))"
        details += String(format: " pts=%.4f window_ms=%.0f", pts,
                          Date().timeIntervalSince(captureWindowOpenedAt) * 1000)
        if let apex = windowApexTimestamp {
            details += String(format: " apex_ts=%.4f", apex)
        }
        DataRecorder.shared.logEvent("capture", details: details)
        onFrameCaptured?(pixelBuffer, context)
    }
}
