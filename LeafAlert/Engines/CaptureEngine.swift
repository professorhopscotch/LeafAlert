import AVFoundation
import CoreMotion
import Combine

/// Manages camera capture sessions gated by accelerometer-based motion detection.
/// Captures frames when the device is relatively still, and also forces periodic
/// captures during motion so the detection pipeline never stalls completely.
final class CaptureEngine: NSObject, ObservableObject {

    // MARK: - Published State

    @Published private(set) var isRunning = false

    /// Number of frames captured in the current 60-second window, for UI diagnostics.
    @Published private(set) var capturesPerMinute: Int = 0

    /// Whether frames are currently being captured (for UI heartbeat indicator).
    @Published private(set) var pipelineActive = false

    // MARK: - Configuration

    /// Acceleration magnitude threshold (in g) below which a frame is captured.
    var motionThreshold: Double = 0.6  // Relaxed from 0.3 — hiking generates ~0.3-0.5g

    /// Duration (in seconds) that motion must stay below threshold before capture.
    var stillnessWindow: TimeInterval = 0.1  // Relaxed from 0.2 for quicker response

    /// Minimum interval between frame emissions regardless of motion state.
    /// In normal mode this defaults to 0.5 seconds; battery saver mode uses `batterySaverInterval`.
    var minCaptureInterval: TimeInterval = 0.5

    /// Minimum interval between frame emissions when in battery-saver mode.
    var batterySaverInterval: TimeInterval = 3.0

    /// Maximum time (seconds) between forced captures regardless of motion state.
    /// Prevents the pipeline from appearing frozen during continuous walking.
    var forcedCaptureInterval: TimeInterval = 5.0

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

    /// Thread-safe storage for stillSince. Accessed from accelerometer queue (write)
    /// and captureQueue (read), so we use a lock to prevent data races.
    private let stillnessLock = NSLock()
    private var _stillSince: Date?

    private var stillSince: Date? {
        get { stillnessLock.lock(); defer { stillnessLock.unlock() }; return _stillSince }
        set { stillnessLock.lock(); defer { stillnessLock.unlock() }; _stillSince = newValue }
    }

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

    private func startMotionUpdates() {
        guard motionManager.isAccelerometerAvailable else { return }
        motionManager.accelerometerUpdateInterval = 0.05
        motionManager.startAccelerometerUpdates(to: .init()) { [weak self] data, _ in
            guard let self, let data else { return }
            let magnitude = sqrt(
                pow(data.acceleration.x, 2) +
                pow(data.acceleration.y, 2) +
                pow(data.acceleration.z, 2)
            )
            // Subtract ~1g for gravity
            let netAcceleration = abs(magnitude - 1.0)
            if netAcceleration < self.motionThreshold {
                if self.stillSince == nil {
                    self.stillSince = Date()
                }
            } else {
                self.stillSince = nil
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
            guard let self else { return }
            self.captureSession.startRunning()
        }
    }

    // MARK: - Helpers

    private var effectiveCaptureInterval: TimeInterval {
        isBatterySaverEnabled ? batterySaverInterval : minCaptureInterval
    }

    private var isDeviceStill: Bool {
        guard let stillSince else { return false }
        return Date().timeIntervalSince(stillSince) >= stillnessWindow
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
        // Allow capture if device is still OR if we've gone too long without a capture.
        guard isDeviceStill || shouldForceCapture else { return }
        guard !isProcessingFrame else { return }

        let now = Date()
        if !shouldForceCapture && now.timeIntervalSince(lastCaptureTime) < effectiveCaptureInterval {
            return
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        isProcessingFrame = true
        lastCaptureTime = now
        frameCountInWindow += 1
        onFrameCaptured?(pixelBuffer)
    }
}
