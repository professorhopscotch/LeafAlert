import AVFoundation
import CoreMotion
import Combine

/// Manages camera capture sessions gated by accelerometer-based motion detection.
/// Only emits frames when the device is relatively still (below the motion threshold).
final class CaptureEngine: NSObject, ObservableObject {

    // MARK: - Published State

    @Published private(set) var isRunning = false

    /// Number of frames captured in the current 60-second window, for UI diagnostics.
    @Published private(set) var capturesPerMinute: Int = 0

    // MARK: - Configuration

    /// Acceleration magnitude threshold (in g) below which a frame is captured.
    var motionThreshold: Double = 0.3

    /// Duration (in seconds) that motion must stay below threshold before capture.
    var stillnessWindow: TimeInterval = 0.2

    /// Minimum interval between frame emissions regardless of motion state.
    /// In normal mode this defaults to 0.5 seconds; battery saver mode uses `batterySaverInterval`.
    var minCaptureInterval: TimeInterval = 0.5

    /// Minimum interval between frame emissions when in battery-saver mode.
    var batterySaverInterval: TimeInterval = 3.0

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
    private var stillSince: Date?

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
        captureQueue.async { [weak self] in
            self?.captureSession.startRunning()
        }
        isRunning = true
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
        isRunning = false
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

    private var effectiveCaptureInterval: TimeInterval {
        isBatterySaverEnabled ? batterySaverInterval : minCaptureInterval
    }

    private var isDeviceStill: Bool {
        guard let stillSince else { return false }
        return Date().timeIntervalSince(stillSince) >= stillnessWindow
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
        guard isDeviceStill else { return }
        guard !isProcessingFrame else { return }

        let now = Date()
        if now.timeIntervalSince(lastCaptureTime) < effectiveCaptureInterval {
            return
        }

        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        isProcessingFrame = true
        lastCaptureTime = now
        frameCountInWindow += 1
        onFrameCaptured?(pixelBuffer)
    }
}
