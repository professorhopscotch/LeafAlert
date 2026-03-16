import AVFoundation
import CoreMotion
import Combine

/// Manages camera capture sessions with physics-based frame timing.
///
/// Instead of requiring the device to be "still" (which rarely happens while hiking),
/// we detect the **apogee** of each stride — the instant where vertical acceleration
/// crosses zero going from upward to downward. At that moment the phone is momentarily
/// in near-freefall relative to the hand, producing the sharpest possible frame.
///
/// Think of it like a ball tossed in the air: at the peak, velocity is changing direction
/// but position is momentarily stationary. Same physics applies to a phone in a hiker's hand.
///
/// Accelerometer runs at 100 Hz (10 ms resolution). Camera runs at 30 fps (33 ms frames).
/// Combined worst-case latency from apogee detection to frame capture: ~43 ms.
final class CaptureEngine: NSObject, ObservableObject {

    // MARK: - Published State

    @Published private(set) var isRunning = false

    /// Number of frames captured in the current 60-second window, for UI diagnostics.
    @Published private(set) var capturesPerMinute: Int = 0

    /// Whether frames are currently being captured (for UI heartbeat indicator).
    @Published private(set) var pipelineActive = false

    // MARK: - Configuration

    /// Minimum interval between frame emissions regardless of motion state.
    var minCaptureInterval: TimeInterval = 0.5

    /// Minimum interval between frame emissions when in battery-saver mode.
    var batterySaverInterval: TimeInterval = 3.0

    /// Maximum time (seconds) between forced captures regardless of motion state.
    /// Prevents the pipeline from appearing frozen if apogee detection doesn't trigger
    /// (e.g., phone mounted on a tripod, sitting on a table).
    var forcedCaptureInterval: TimeInterval = 4.0

    /// Net acceleration (g) below which the device is considered "still enough" to capture
    /// even without an apogee zero-crossing. Handles the stationary-phone case.
    var stillnessThreshold: Double = 0.08

    /// Whether battery saver mode is active.
    var isBatterySaverEnabled = false

    // MARK: - Callbacks

    /// Called on a background queue when a frame is ready for inference.
    var onFrameCaptured: ((CVPixelBuffer) -> Void)?

    // MARK: - Private Properties

    private let captureSession = AVCaptureSession()
    private let motionManager = CMMotionManager()
    private let captureQueue = DispatchQueue(label: "com.leafalert.capture", qos: .userInitiated)
    private var lastCaptureTime: Date = .distantPast

    /// Thread-safe apogee flag. Set by the accelerometer queue, read by captureQueue.
    private let motionLock = NSLock()

    /// True when the accelerometer has detected a zero-crossing (apogee) or stillness.
    /// Reset to false after a frame is captured.
    private var _captureReady = false

    private var captureReady: Bool {
        get { motionLock.lock(); defer { motionLock.unlock() }; return _captureReady }
        set { motionLock.lock(); defer { motionLock.unlock() }; _captureReady = newValue }
    }

    /// Previous net acceleration sample, used for zero-crossing detection.
    /// "Net" means gravity subtracted: magnitude(accel) - 1g.
    /// Positive = accelerating upward (or decelerating downward), negative = vice versa.
    private var previousNetAccel: Double = 0.0

    /// Guards against sending a new frame while the previous one is still being processed.
    /// Accessed only on `captureQueue` so no additional synchronization is needed.
    private var isProcessingFrame = false

    /// Counter for the current 60-second window (accessed on captureQueue).
    private var frameCountInWindow: Int = 0

    /// Timer that publishes `capturesPerMinute` and resets the window counter every 60 seconds.
    private var diagnosticsTimer: Timer?

    // MARK: - Lifecycle

    /// Configures and starts the camera session and motion monitoring.
    func start() {
        guard !isRunning else { return }
        configureCaptureSession()
        startMotionUpdates()
        startDiagnosticsTimer()
        observeSessionInterruptions()
        captureQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
        isRunning = true
        DispatchQueue.main.async { self.pipelineActive = true }
    }

    /// Stops camera and motion monitoring.
    func stop() {
        guard isRunning else { return }
        captureQueue.async { [weak self] in
            self?.captureSession.stopRunning()
        }
        motionManager.stopAccelerometerUpdates()
        diagnosticsTimer?.invalidate()
        diagnosticsTimer = nil
        removeSessionObservers()
        isRunning = false
        DispatchQueue.main.async { self.pipelineActive = false }
    }

    // MARK: - Private Setup

    private func configureCaptureSession() {
        captureSession.beginConfiguration()
        captureSession.sessionPreset = .vga640x480

        guard
            let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
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

        captureSession.commitConfiguration()
    }

    /// Polls accelerometer at 100 Hz and detects apogee zero-crossings.
    ///
    /// The net acceleration (|a| - 1g) oscillates during walking:
    ///   - Positive when the hand/phone accelerates upward (push-off phase)
    ///   - Negative when it decelerates / falls (swing-through phase)
    ///   - Zero at the peak (apogee) — the optimal capture moment
    ///
    /// We detect the zero-crossing from positive → negative (or near-zero → negative),
    /// which corresponds to the top of each stride bounce.
    private func startMotionUpdates() {
        guard motionManager.isAccelerometerAvailable else {
            // No accelerometer — always allow captures (e.g., simulator)
            captureReady = true
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
            // Net acceleration: how far from pure gravity (1g).
            // Signed: positive means total accel > 1g (pushing up), negative means < 1g (falling).
            let netAccel = magnitude - 1.0

            // Apogee detection: previous sample was positive (or near-zero), current is negative.
            // This is the zero-crossing at the top of the bounce.
            let isApogee = self.previousNetAccel >= 0 && netAccel < 0

            // Also detect near-stillness (phone barely moving).
            let isStill = abs(netAccel) < self.stillnessThreshold

            if isApogee || isStill {
                self.captureReady = true
            }

            self.previousNetAccel = netAccel
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
            guard let self else { return }
            self.captureSession.startRunning()
        }
    }

    // MARK: - Helpers

    private var effectiveCaptureInterval: TimeInterval {
        isBatterySaverEnabled ? batterySaverInterval : minCaptureInterval
    }

    /// Whether enough time has passed since the last capture to force one regardless of motion.
    private var shouldForceCapture: Bool {
        Date().timeIntervalSince(lastCaptureTime) >= forcedCaptureInterval
    }

    /// Called by the consumer (via `onFrameCaptured`) to signal that frame processing is complete.
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
        // Capture if: apogee/stillness detected, OR forced interval elapsed.
        guard captureReady || shouldForceCapture else { return }
        guard !isProcessingFrame else { return }

        let now = Date()
        if !shouldForceCapture && now.timeIntervalSince(lastCaptureTime) < effectiveCaptureInterval {
            return
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        isProcessingFrame = true
        lastCaptureTime = now
        captureReady = false  // Reset — wait for next apogee
        frameCountInWindow += 1
        onFrameCaptured?(pixelBuffer)
    }
}
