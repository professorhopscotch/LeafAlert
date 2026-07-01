import Vision
import CoreML
import CoreGraphics
import CoreVideo

/// Runs on-device plant detection inference using a Core ML model via the Vision framework.
/// The model is swappable — drop a new .mlmodelc into Resources/MLModels with no code changes.
final class InferenceEngine: ObservableObject {

    // MARK: - Published State

    @Published private(set) var isReady = false

    /// Duration (in seconds) of the most recent inference call, for performance diagnostics.
    @Published private(set) var lastInferenceTime: TimeInterval = 0

    /// Per-class confidence values from the most recent inference. For debug display.
    @Published private(set) var lastClassConfidences: [String: Float] = [:]

    /// Total number of inferences run since model was loaded.
    @Published private(set) var totalInferences: Int = 0

    // MARK: - Private Properties

    private var vnModel: VNCoreMLModel?
    private let inferenceQueue = DispatchQueue(
        label: "com.leafalert.inference",
        qos: .userInitiated
    )

    /// Guards against overlapping inference calls. When true, new `classify` requests
    /// are dropped (not queued) to prevent inference backlog.
    /// Accessed only on `inferenceQueue`.
    private var isProcessing = false

    /// Class labels the model can output that represent a toxic plant detection.
    /// "safe_plants" is intentionally excluded — we only surface positive identifications.
    static let toxicLabels: Set<String> = ["poison_ivy", "poison_oak", "poison_sumac"]

    /// All labels the model knows (used for validation).
    static let allLabels: Set<String> = ["poison_ivy", "poison_oak", "poison_sumac", "safe_plants"]

    // MARK: - Setup

    /// Loads the Core ML model from the app bundle.
    func loadModel() {
        inferenceQueue.async { [weak self] in
            guard let self else { return }
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .cpuAndNeuralEngine

                // Attempt to load a model named "PlantDetector" from the bundle.
                // Falls back gracefully if the model isn't yet present.
                guard let modelURL = Bundle.main.url(
                    forResource: "PlantDetector",
                    withExtension: "mlmodelc"
                ) else {
                    print("[InferenceEngine] PlantDetector model not found in bundle.")
                    return
                }

                let mlModel = try MLModel(contentsOf: modelURL, configuration: config)
                self.vnModel = try VNCoreMLModel(for: mlModel)
                DispatchQueue.main.async {
                    self.isReady = true
                }
            } catch {
                print("[InferenceEngine] Failed to load model: \(error)")
            }
        }
    }

    // MARK: - Inference

    /// Runs inference on a pixel buffer with test-time augmentation (TTA).
    /// Averages predictions from the original and horizontally flipped image
    /// to smooth orientation-dependent errors.
    /// - Parameters:
    ///   - pixelBuffer: A CVPixelBuffer from the camera capture pipeline.
    ///   - completion: Called on a background queue with the result, or nil on failure.
    func classify(
        pixelBuffer: CVPixelBuffer,
        completion: @escaping (DetectionResult?) -> Void
    ) {
        guard let vnModel else {
            completion(nil)
            return
        }

        inferenceQueue.async { [weak self] in
            guard let self else {
                completion(nil)
                return
            }

            // Drop this request if a previous inference is still in progress.
            guard !self.isProcessing else {
                completion(nil)
                return
            }
            self.isProcessing = true

            let startTime = CFAbsoluteTimeGetCurrent()

            let finishInference = {
                let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                DispatchQueue.main.async {
                    self.lastInferenceTime = elapsed
                }
                self.inferenceQueue.async {
                    self.isProcessing = false
                }
            }

            // --- Pass 1: Original orientation ---
            // Train/serve crop parity: `.scaleFill` squashes the full frame to the
            // model's 224x224 input with no crop, matching the Python val transform
            // (Resize((224,224))). `.centerCrop` would discard the long-edge margins
            // and crop out toxic plants near the frame edge, lowering recall — the
            // dangerous failure mode for a safety app.
            let originalHandler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer, options: [:]
            )
            let originalRequest = VNCoreMLRequest(model: vnModel)
            originalRequest.imageCropAndScaleOption = .scaleFill

            // --- Pass 2: Horizontally flipped (zero-copy via orientation) ---
            let flippedHandler = VNImageRequestHandler(
                cvPixelBuffer: pixelBuffer,
                orientation: .upMirrored,
                options: [:]
            )
            let flippedRequest = VNCoreMLRequest(model: vnModel)
            // Same parity contract as Pass 1 — squash, don't crop.
            flippedRequest.imageCropAndScaleOption = .scaleFill

            do {
                try originalHandler.perform([originalRequest])
                try flippedHandler.perform([flippedRequest])
            } catch {
                print("[InferenceEngine] TTA inference failed: \(error)")
                finishInference()
                completion(nil)
                return
            }

            // Collect observations from both passes
            guard let originalObs = originalRequest.results as? [VNClassificationObservation],
                  let flippedObs = flippedRequest.results as? [VNClassificationObservation],
                  !originalObs.isEmpty
            else {
                finishInference()
                completion(nil)
                return
            }

            // Average confidence values by class identifier
            let averaged = Self.averageObservations(originalObs, flippedObs)

            // Publish per-class confidences for debug UI
            DispatchQueue.main.async {
                self.lastClassConfidences = averaged
                self.totalInferences += 1
            }

            finishInference()

            // Find the highest-confidence *toxic* plant class.
            guard let topToxic = averaged
                .filter({ Self.toxicLabels.contains($0.key) })
                .max(by: { $0.value < $1.value })
            else {
                completion(nil)
                return
            }

            // Only report if the toxic class beats safe_plants.
            let safeConfidence = averaged["safe_plants"] ?? 0.0
            guard topToxic.value > safeConfidence else {
                completion(nil)
                return
            }

            let clampedConfidence = min(max(topToxic.value, 0.0), 1.0)

            // Use attention-based saliency to approximate bounding box.
            let bbox = Self.saliencyBoundingBox(pixelBuffer: pixelBuffer)

            let result = DetectionResult(
                plantType: topToxic.key,
                confidence: clampedConfidence,
                boundingBox: bbox
            )
            completion(result)
        }
    }

    // MARK: - TTA Helpers

    /// Runs attention-based saliency to find the most prominent object region.
    /// Returns a normalised CGRect (Vision coordinates: origin bottom-left) or .zero on failure.
    private static func saliencyBoundingBox(pixelBuffer: CVPixelBuffer) -> CGRect {
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
        let request = VNGenerateAttentionBasedSaliencyImageRequest()
        // Saliency runs on the full frame (this request type has no
        // imageCropAndScaleOption); its normalised box already lives in the
        // same full-frame space the `.scaleFill` classifier path consumes.
        do {
            try handler.perform([request])
        } catch {
            return .zero
        }
        guard let observation = request.results?.first,
              let salientObject = observation.salientObjects?.first else {
            return .zero
        }
        return salientObject.boundingBox
    }

    /// Averages confidence values from two sets of classification observations.
    private static func averageObservations(
        _ a: [VNClassificationObservation],
        _ b: [VNClassificationObservation]
    ) -> [String: Float] {
        var sums: [String: Float] = [:]
        var counts: [String: Int] = [:]

        for obs in a {
            sums[obs.identifier, default: 0] += obs.confidence
            counts[obs.identifier, default: 0] += 1
        }
        for obs in b {
            sums[obs.identifier, default: 0] += obs.confidence
            counts[obs.identifier, default: 0] += 1
        }

        var result: [String: Float] = [:]
        for (key, sum) in sums {
            result[key] = sum / Float(counts[key] ?? 1)
        }
        return result
    }
}
