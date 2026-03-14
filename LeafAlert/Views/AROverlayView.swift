import SwiftUI
import ARKit
import RealityKit

/// Augmented reality view that highlights the approximate location of a detected plant.
struct AROverlayView: View {
    let detectionResult: DetectionResult

    @Environment(\.dismiss) private var dismiss

    var body: some View {
        ZStack {
            // AR session container
            ARViewContainer(boundingBox: detectionResult.boundingBox)
                .ignoresSafeArea()

            VStack {
                // Detection info overlay
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(detectionResult.plantType.replacingOccurrences(of: "_", with: " ").capitalized)
                            .font(.headline)
                        Text("Confidence: \(Int(detectionResult.confidence * 100))%")
                            .font(.caption)
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

/// UIViewRepresentable wrapper for RealityKit's ARView.
struct ARViewContainer: UIViewRepresentable {
    let boundingBox: CGRect

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        let config = ARWorldTrackingConfiguration()
        arView.session.run(config)
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        // TODO: Add overlay anchor at bounding box location
    }
}

#Preview {
    AROverlayView(detectionResult: DetectionResult(
        plantType: "poison_ivy",
        confidence: 0.87,
        boundingBox: .zero
    ))
}
