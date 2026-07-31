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

            // NO box is drawn here, deliberately. The saliency rect is expressed in
            // the capture session's frame, but this screen renders a DIFFERENT
            // camera (ARKit's, with its own field of view), and the rect came from a
            // frame captured moments ago rather than the live feed. There is no
            // correct mapping between the two, so any rectangle drawn here would be
            // a confident label over an arbitrary patch of ground — the exact
            // failure this screen is supposed to help the user avoid.
            // The live-preview box in PatrolView is the honest locator; this screen
            // gives a plain-language warning instead.

            VStack {
                // Detection info overlay
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(DetectionFormatting.detectionHeadline(detectionResult.plantType, severity: severity))
                            .font(.headline)
                        Text(DetectionFormatting.detectionSubtitle(confidence: detectionResult.confidence, severity: severity))
                            .font(.caption)
                        Text("Seen moments ago near here — this view can't pinpoint it. Scan the area carefully.")
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
