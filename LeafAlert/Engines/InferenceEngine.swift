import Vision
import CoreML
import CoreGraphics
import CoreVideo

/// Runs on-device plant detection inference using a Core ML model via the Vision framework.
/// The model is swappable — drop a new .mlmodelc into Resources/MLModels with no code changes.
final class InferenceEngine: ObservableObject {

    // MARK: - Published State

    @Published private(set) var isReady = false

    // MARK: - Private Properties

    private var vnModel: VNCoreMLModel?
    private let inferenceQueue = DispatchQueue(
        label: "com.leafalert.inference",
        qos: .userInitiated
    )

    /// Known class labels the model can output.
    static let supportedLabels = ["poison_ivy", "poison_oak", "poison_sumac", "negative"]

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

        inferenceQueue.async {
            let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, options: [:])
            let request = VNCoreMLRequest(model: vnModel) { request, error in
                guard error == nil,
                      let observations = request.results as? [VNClassificationObservation],
                      let top = observations.first,
                      Self.supportedLabels.contains(top.identifier)
                else {
                    completion(nil)
                    return
                }

                let result = DetectionResult(
                    plantType: top.identifier,
                    confidence: top.confidence,
                    boundingBox: .zero  // Classification models don't produce bounding boxes
                )
                completion(result)
            }

            request.imageCropAndScaleOption = .centerCrop

            do {
                try handler.perform([request])
            } catch {
                print("[InferenceEngine] Inference failed: \(error)")
                completion(nil)
            }
        }
    }
}
