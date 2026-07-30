import SwiftUI
import ARKit
import RealityKit

/// Augmented reality view that highlights the approximate location of a detected plant.
struct AROverlayView: View {
    let detectionResult: DetectionResult

    @Environment(\.dismiss) private var dismiss

    /// Severity of the detection being shown, used for hedged copy.
    private var severity: DetectionSeverity {
        ToxicityThresholds.severity(
            plantType: detectionResult.plantType,
            confidence: detectionResult.confidence,
            sensitivity: Float(sensitivityThreshold)
        )
    }

    @AppStorage("sensitivityThreshold") private var sensitivityThreshold: Double = 0.50

    var body: some View {
        ZStack {
            // Live AR camera feed.
            ARViewContainer()
                .ignoresSafeArea()

            // Approximate region indicator. Drawn in 2D over the feed rather than as
            // a world-anchored 3D entity: the box comes from saliency on a frame
            // captured a moment ago, so it marks "roughly where this was seen in
            // that shot", not a point in world space. Pretending otherwise would put
            // a confident red marker on the wrong patch of ground.
            if detectionResult.boundingBox != .zero {
                BoundingBoxOverlay(
                    boundingBox: detectionResult.boundingBox,
                    label: DetectionFormatting.plantDisplayName(detectionResult.plantType),
                    confidence: detectionResult.confidence
                )
                .ignoresSafeArea()
            }

            VStack {
                // Detection info overlay
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(DetectionFormatting.detectionHeadline(detectionResult.plantType, severity: severity))
                            .font(.headline)
                        Text(DetectionFormatting.detectionSubtitle(confidence: detectionResult.confidence, severity: severity))
                            .font(.caption)
                        Text(detectionResult.boundingBox == .zero
                             ? "Location not pinpointed — scan the area carefully."
                             : "Box shows the approximate region from the last scan.")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .padding()
                    .background(.ultraThinMaterial)
                    .clipShape(RoundedRectangle(cornerRadius: 12))

                    Spacer()

                    Button {
                        dismiss()
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .font(.title)
                            .foregroundStyle(.white)
                    }
                }
                .padding()

                Spacer()

                // Disclaimer
                Text("This app assists identification — always verify visually before touching any plant.")
                    .font(.caption2)
                    .foregroundStyle(.white.opacity(0.7))
                    .multilineTextAlignment(.center)
                    .padding()
                    .background(.ultraThinMaterial)

                NavigationLink("Learn about this plant", destination:
                    PlantDetailView(selectedPlantID: detectionResult.plantType)
                )
                .font(.callout.bold())
                .padding()
            }
        }
    }
}

/// UIViewRepresentable wrapper for RealityKit's ARView — a plain live camera feed.
///
/// This deliberately places NO 3D entity in the scene. It previously anchored a
/// warning plane to `AnchorEntity(.camera)`, which is camera-relative: the marker
/// followed the user's gaze instead of staying on the plant, so it appeared to
/// point at whatever they happened to be looking at — including after turning away
/// from the plant entirely. Localising for real needs a detection/segmentation
/// model plus a raycast to a world surface; until then the honest presentation is a
/// 2D "approximate region" box drawn by the parent view.
struct ARViewContainer: UIViewRepresentable {

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: UIScreen.main.bounds)
        arView.automaticallyConfigureSession = false
        guard ARWorldTrackingConfiguration.isSupported else {
            return arView
        }
        let config = ARWorldTrackingConfiguration()
        config.isAutoFocusEnabled = true
        arView.session.run(config)
        return arView
    }

    static func dismantleUIView(_ uiView: ARView, coordinator: ()) {
        uiView.session.pause()
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        // Nothing to update: the region indicator is drawn in 2D by the parent.
    }
}

#Preview {
    AROverlayView(detectionResult: DetectionResult(
        plantType: "poison_ivy",
        confidence: 0.87,
        boundingBox: .zero
    ))
}
