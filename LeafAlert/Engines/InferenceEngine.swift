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

    /// Runs inference on a pixel buffer and returns the best detection result.
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

            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            let request = VNCoreMLRequest(model: vnModel) { request, error in
                defer { finishInference() }

                guard error == nil,
                      let observations = request.results as? [VNClassificationObservation],
                      !observations.isEmpty
                else {
                    completion(nil)
                    return
                }

                // Find the highest-confidence *toxic* plant class.
                // If "safe_plants" wins, we return nil (no detection).
                guard let topToxic = observations.first(where: {
                    Self.toxicLabels.contains($0.identifier)
                }) else {
                    completion(nil)
                    return
                }

                // Only report if the toxic class actually beats safe_plants.
                let safeConfidence = observations
                    .first(where: { $0.identifier == "safe_plants" })?
                    .confidence ?? 0.0

                guard topToxic.confidence > safeConfidence else {
                    completion(nil)
                    return
                }

                // Clamp confidence to [0, 1] as a safety measure.
                let clampedConfidence = min(max(topToxic.confidence, 0.0), 1.0)

                let result = DetectionResult(
                    plantType: topToxic.identifier,
                    confidence: clampedConfidence,
                    boundingBox: .zero
                )
                completion(result)
            }

            request.imageCropAndScaleOption = .centerCrop

            do {
                try handler.perform([request])
            } catch {
                print("[InferenceEngine] Inference failed: \(error)")
                finishInference()
                completion(nil)
            }
        }
    }
}
