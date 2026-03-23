import AVFoundation
import CoreMotion
import Combine

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

    // MARK: - Published State

    @Published private(set) var isRunning = false

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

    // MARK: - Configuration (thread-safe via configLock)

    private let configLock = NSLock()

    private var _minCaptureInterval: TimeInterval = 1.0
    private var _batterySaverInterval: TimeInterval = 4.0
    private var _maxCaptureWindowDuration: TimeInterval = 1.5
    private var _forcedCaptureInterval: TimeInterval = 4.0
    private var _stillnessThreshold: Double = 0.08
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

    /// Whether battery saver mode is active.
    var isBatterySaverEnabled: Bool {
        get { configLock.lock(); defer { configLock.unlock() }; return _isBatterySaverEnabled }
        set { configLock.lock(); defer { configLock.unlock() }; _isBatterySaverEnabled = newValue }
    }

    // MARK: - Callbacks

    /// Called on a background queue when a frame is ready for inference.
    var onFrameCaptured: ((CVPixelBuffer) -> Void)?

    /// Exposed for CameraPreviewView to attach a preview layer.
    var session: AVCaptureSession { captureSession }

    // MARK: - Private Properties

    private let captureSession = AVCaptureSession()
    private let motionManager = CMMotionManager()
    private let captureQueue = DispatchQueue(label: "com.leafalert.capture", qos: .userInitiated)
    private var lastCaptureTime: Date = .distantPast
    private var videoOutput: AVCaptureVideoDataOutput?

    // -- Apogee detection state (accelerometer queue only) --

    /// Previous net acceleration sample for zero-crossing detection.
    private var previousNetAccel: Double = 0.0

    // -- Duty cycle state --

    /// Thread-safe flag: accelerometer sets true at apogee, captureQueue reads & resets.
    private let motionLock = NSLock()
    private var _apogeeDetected = false

    private var apogeeDetected: Bool {
        get { motionLock.lock(); defer { motionLock.unlock() }; return _apogeeDetected }
        set { motionLock.lock(); defer { motionLock.unlock() }; _apogeeDetected = newValue }
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
        if !isConfigured {
            configureCaptureSession()
            isConfigured = true
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
        motionManager.stopAccelerometerUpdates()
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

    private func configureCaptureSession() {
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
            return
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
    }

    /// Polls accelerometer at 100 Hz for apogee zero-crossing detection.
    ///
    /// Net acceleration = |a| - 1g:
    ///   - Positive during upward acceleration (push-off phase of stride)
    ///   - Negative during downward acceleration (swing-through / falling)
    ///   - Zero at apogee — optimal capture moment
    private func startMotionUpdates() {
        guard motionManager.isAccelerometerAvailable else {
            // No accelerometer (simulator) — keep inference window always open.
            captureQueue.async { [weak self] in self?.setInferenceWindow(open: true) }
            return
        }
        motionManager.accelerometerUpdateInterval = 0.01  // 100 Hz
        motionManager.startAccelerometerUpdates(to: .init()) { [weak self] data, _ in
            guard let self, let data else { return }
            let magnitude = sqrt(
                pow(data.acceleration.x, 2) +
                pow(data.acceleration.y, 2) +
                pow(data.acceleration.z, 2)
            )
            let netAccel = magnitude - 1.0

            // Apogee: zero-crossing from positive (pushing up) to negative (falling).
            let isApogee = self.previousNetAccel >= 0 && netAccel < 0

            // Near-stillness: phone barely moving (on a table, held very steady).
            let isStill = abs(netAccel) < self.stillnessThreshold

            if isApogee || isStill {
                self.apogeeDetected = true
                if isApogee {
                    DispatchQueue.main.async { self.apogeeCount += 1 }
                }
            }

            // Throttle UI updates to ~10 Hz (every 10th sample at 100 Hz)
            if Int.random(in: 0..<10) == 0 {
                DispatchQueue.main.async { self.currentNetAccel = netAccel }
            }

            self.previousNetAccel = netAccel
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

            let shouldCapture = apogeeDetected || timeSinceLastCapture >= forcedCaptureInterval

            if shouldCapture {
                apogeeDetected = false
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

        onFrameCaptured?(pixelBuffer)
    }
}
