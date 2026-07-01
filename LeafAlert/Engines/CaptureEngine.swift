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

/// Manages camera capture with physics-based timing and duty-cycled power management.
///
/// # Power Architecture
///
/// The accelerometer (MEMS, ~1 mW) runs continuously at 100 Hz as a low-power trigger.
/// The camera sensor (~200-400 mW) runs in a duty cycle:
///
///   1. Camera OFF (default) — only accelerometer running
///   2. Apogee detected → camera ON, capture window opens
///   3. First good frame captured → camera OFF, cooldown begins
///   4. After cooldown interval → return to step 1
///
/// This yields ~80-90% power savings vs. continuous 30 fps recording, while still
/// capturing optimally-timed frames at stride apogee.
///
/// # Apogee Detection
///
/// During walking gait, net vertical acceleration (|a| - 1g) oscillates.
/// The zero-crossing from positive to negative corresponds to the peak of each
/// stride bounce — the moment of minimum phone motion and sharpest frames.
///
/// At 100 Hz accelerometer + 30 fps camera, worst-case latency from apogee
/// detection to captured frame is ~43 ms.
final class CaptureEngine: NSObject, ObservableObject {

    enum CaptureTrigger: String { case apogee, stillness, forced }

    /// Camera authorization state, for the UI to surface an "access needed" message.
    enum CameraPermission { case unknown, authorized, denied }

    // MARK: - Published State

    @Published private(set) var isRunning = false

    /// Current camera authorization state. When `.denied`, the UI should prompt the
    /// user to enable camera access in Settings — the capture pipeline cannot run.
    @Published private(set) var cameraPermission: CameraPermission = .unknown

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

    private var _minCaptureInterval: TimeInterval = 1.0
    private var _batterySaverInterval: TimeInterval = 4.0
    private var _maxCaptureWindowDuration: TimeInterval = 1.5
    private var _forcedCaptureInterval: TimeInterval = 4.0
    private var _stillnessThreshold: Double = 0.08
    private var _rotationRateThreshold: Double = 1.5  // rad/s — reject captures above this
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

    /// Previous SIGNED vertical acceleration (userAcceleration projected onto
    /// gravity) for the apogee zero-crossing. Only touched on the motion queue,
    /// but guarded for consistency with the other motion state.
    private var _previousVertical: Double = 0.0

    private var previousVertical: Double {
        get { motionLock.lock(); defer { motionLock.unlock() }; return _previousVertical }
        set { motionLock.lock(); defer { motionLock.unlock() }; _previousVertical = newValue }
    }

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
    private var _apogeeDetected = false
    private var _lastMotionTrigger: CaptureTrigger = .forced

    private var apogeeDetected: Bool {
        get { motionLock.lock(); defer { motionLock.unlock() }; return _apogeeDetected }
        set { motionLock.lock(); defer { motionLock.unlock() }; _apogeeDetected = newValue }
    }

    /// The type of motion event that last set apogeeDetected.
    private var lastMotionTrigger: CaptureTrigger {
        get { motionLock.lock(); defer { motionLock.unlock() }; return _lastMotionTrigger }
        set { motionLock.lock(); defer { motionLock.unlock() }; _lastMotionTrigger = newValue }
    }

    /// Whether the inference window is open (ready to capture a frame for ML).
    /// Accessed only on `captureQueue`.
    private var isInferenceWindowOpen = false

    /// When the current inference window was opened. Used to enforce maxCaptureWindowDuration.
    /// Accessed only on `captureQueue`.
    private var captureWindowOpenedAt: Date = .distantPast

    /// Guards against sending a new frame while the previous one is still being processed.
    /// Accessed only on `captureQueue`.
    private var isProcessingFrame = false

    /// What triggered the current capture window. Set in manageDutyCycle, read in captureOutput.
    /// Accessed only on `captureQueue`.
    private var lastTrigger: CaptureTrigger = .forced

    /// Counter for the current 60-second window (accessed on captureQueue).
    private var frameCountInWindow: Int = 0

    /// Timer that publishes `capturesPerMinute` and manages the duty cycle.
    private var diagnosticsTimer: Timer?

    /// Timer that polls for apogee events and manages camera power.
    private var dutyCycleTimer: Timer?

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

        // Start the session (keeps it warm) but camera output starts disabled.
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
        diagnosticsTimer?.invalidate()
        diagnosticsTimer = nil
        dutyCycleTimer?.invalidate()
        dutyCycleTimer = nil
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
            DispatchQueue.main.async { self.isCameraActive = true }
        } else if !open && isInferenceWindowOpen {
            isInferenceWindowOpen = false
            DispatchQueue.main.async { self.isCameraActive = false }
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

    /// Polls device motion at 100 Hz for apogee zero-crossing and rotation rate detection.
    ///
    /// Apogee is detected on the SIGNED vertical acceleration — `userAcceleration`
    /// projected onto the gravity direction:
    ///   - Positive while accelerating upward (push-off phase of stride)
    ///   - Negative while accelerating downward (falling toward the ground)
    ///   - Crosses zero (positive → negative) at the stride apex — the optimal
    ///     capture moment. A plain magnitude can never go negative, so it cannot
    ///     express this crossing; the signed projection can.
    ///
    /// The unsigned magnitude of `userAcceleration` is still used for the
    /// stillness test and the recorded `netAcceleration` telemetry field.
    ///
    /// Rotation rate is used as a gate: if the device is rotating too fast,
    /// captures are suppressed to avoid motion blur.
    private func startMotionUpdates() {
        guard motionManager.isDeviceMotionAvailable else {
            // No motion sensors (simulator) — keep inference window always open.
            captureQueue.async { [weak self] in self?.setInferenceWindow(open: true) }
            return
        }
        motionManager.deviceMotionUpdateInterval = 0.01  // 100 Hz
        motionManager.startDeviceMotionUpdates(to: .init()) { [weak self] motion, _ in
            guard let self, let motion else { return }

            // Forward every IMU sample to the recorder (no-op when not recording)
            DataRecorder.shared.appendIMUSample(motion)

            // Gravity-subtracted user acceleration.
            let ua = motion.userAcceleration
            // Unsigned motion intensity — used for stillness + telemetry.
            let netAccel = sqrt(ua.x * ua.x + ua.y * ua.y + ua.z * ua.z)
            // Signed vertical component: project user acceleration onto the
            // (unit-length) gravity vector. Positive = accelerating upward,
            // negative = downward; crosses zero at the stride apex.
            let g = motion.gravity
            let vertical = ua.x * g.x + ua.y * g.y + ua.z * g.z

            // Rotation rate magnitude (rad/s)
            let rr = motion.rotationRate
            let rotRate = sqrt(rr.x * rr.x + rr.y * rr.y + rr.z * rr.z)
            self.currentRotRate = rotRate

            let rotationTooFast = rotRate > self.rotationRateThreshold

            // Apogee: signed zero-crossing from positive (pushing up) to negative
            // (falling). Require the positive side to exceed a noise floor so
            // sensor jitter while stationary isn't counted as an apogee.
            let noiseFloor = 0.03
            let isApogee = self.previousVertical >= noiseFloor && vertical < noiseFloor

            // Near-stillness: phone barely moving (on a table, held very steady).
            let isStill = netAccel < self.stillnessThreshold

            // Only trigger if rotation rate is below threshold
            if !rotationTooFast {
                if isApogee {
                    self.apogeeDetected = true
                    self.lastMotionTrigger = .apogee
                    DispatchQueue.main.async { self.apogeeCount += 1 }
                } else if isStill {
                    self.apogeeDetected = true
                    self.lastMotionTrigger = .stillness
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
            self.previousVertical = vertical
        }
    }

    /// Runs at 20 Hz to check for apogee events and manage camera power.
    /// This is deliberately on the main run loop (lightweight — just checks a bool).
    private func startDutyCycleTimer() {
        dutyCycleTimer?.invalidate()
        dutyCycleTimer = Timer.scheduledTimer(withTimeInterval: 0.05, repeats: true) { [weak self] _ in
            guard let self, self.isRunning else { return }
            self.captureQueue.async {
                self.manageDutyCycle()
            }
        }
    }

    /// Core duty cycle logic. Called on `captureQueue`.
    private func manageDutyCycle() {
        let now = Date()
        let timeSinceLastCapture = now.timeIntervalSince(lastCaptureTime)
        let cooldown = isBatterySaverEnabled ? batterySaverInterval : minCaptureInterval

        if isInferenceWindowOpen {
            // Inference window is open — check if it should close.
            let windowDuration = now.timeIntervalSince(captureWindowOpenedAt)
            if windowDuration >= maxCaptureWindowDuration {
                // Window expired without capturing — close and try again next cycle.
                setInferenceWindow(open: false)
            }
        } else {
            // Inference window is closed — should we open it?
            guard timeSinceLastCapture >= cooldown else { return }
            guard !isProcessingFrame else { return }

            if apogeeDetected {
                lastTrigger = lastMotionTrigger
                apogeeDetected = false
                setInferenceWindow(open: true)
            } else if timeSinceLastCapture >= forcedCaptureInterval {
                lastTrigger = .forced
                setInferenceWindow(open: true)
            }
        }
    }

    private func startDiagnosticsTimer() {
        diagnosticsTimer?.invalidate()
        diagnosticsTimer = Timer.scheduledTimer(withTimeInterval: 60.0, repeats: true) { [weak self] _ in
            guard let self else { return }
            self.captureQueue.async {
                let count = self.frameCountInWindow
                self.frameCountInWindow = 0
                DispatchQueue.main.async {
                    self.capturesPerMinute = count
                }
            }
        }
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
        DataRecorder.shared.logEvent(
            "capture",
            details: "trigger=\(lastTrigger.rawValue) accel=\(String(format: "%.3f", previousNetAccel)) rot=\(String(format: "%.3f", currentRotRate))"
        )
        onFrameCaptured?(pixelBuffer, context)
    }
}
