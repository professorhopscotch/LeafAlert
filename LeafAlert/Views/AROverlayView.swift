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

    /// Tag name used to identify the overlay anchor so it can be removed on subsequent updates.
    private static let anchorName = "leafalert-overlay"
    /// Distance (in metres) at which the overlay plane is placed in front of the camera.
    private static let overlayDistance: Float = 1.5
    /// Duration of one half-cycle of the pulse animation (seconds).
    private static let pulseDuration: TimeInterval = 0.8
    /// Scale multiplier at the peak of the pulse animation.
    private static let pulseScaleFactor: Float = 1.15

    func makeUIView(context: Context) -> ARView {
        let arView = ARView(frame: .zero)
        let config = ARWorldTrackingConfiguration()
        arView.session.run(config)
        return arView
    }

    func updateUIView(_ uiView: ARView, context: Context) {
        // Remove any previously placed overlay anchor to avoid duplicates.
        uiView.scene.anchors.removeAll { $0.name == Self.anchorName }

        let isGeneralWarning = boundingBox == .zero
        let distance = Self.overlayDistance

        // Normalised coordinates: (0,0) is top-left, (1,1) is bottom-right.
        // When the bounding box is zero, place a default indicator directly ahead.
        let centreX: Float = isGeneralWarning ? 0.5 : Float(boundingBox.midX)
        let centreY: Float = isGeneralWarning ? 0.5 : Float(boundingBox.midY)
        let boxWidth: Float = isGeneralWarning ? 0.3 : Float(boundingBox.width)
        let boxHeight: Float = isGeneralWarning ? 0.3 : Float(boundingBox.height)

        // Approximate horizontal and vertical field-of-view angles (radians) for a typical phone camera.
        let hFOV: Float = Float.pi / 3   // ~60 degrees
        let vFOV: Float = Float.pi / 4   // ~45 degrees

        // Angular offset from centre: positive X = right, positive Y = up.
        let xOffset = distance * tan((centreX - 0.5) * hFOV)
        // Invert Y because normalised image Y increases downward but world Y increases upward.
        let yOffset = distance * tan((0.5 - centreY) * vFOV)

        // Scale the overlay plane to roughly match the bounding-box proportions at that distance.
        let planeWidth = max(distance * tan(boxWidth * hFOV / 2) * 2, 0.15)
        let planeHeight = max(distance * tan(boxHeight * vFOV / 2) * 2, 0.15)

        // Build a semi-transparent warning plane.
        let mesh = MeshResource.generatePlane(width: planeWidth, height: planeHeight)
        let color: UIColor = isGeneralWarning
            ? UIColor.orange.withAlphaComponent(0.4)
            : UIColor.red.withAlphaComponent(0.4)
        var material = UnlitMaterial(color: color)
        material.blending = .transparent(opacity: .init(floatLiteral: 1.0))
        let entity = ModelEntity(mesh: mesh, materials: [material])
        entity.name = "overlay-plane"
        entity.position = SIMD3<Float>(xOffset, yOffset, -distance)

        // Create a camera-relative anchor so the overlay stays in front of the user.
        let anchor = AnchorEntity(.camera)
        anchor.name = Self.anchorName
        anchor.addChild(entity)
        uiView.scene.addAnchor(anchor)

        startPulseAnimation(on: entity)
    }

    /// Adds a repeating pulse animation that smoothly toggles between normal size and
    /// a slightly enlarged size using RealityKit transform animations with a Timer.
    private func startPulseAnimation(on entity: ModelEntity) {
        let originalTransform = entity.transform
        var scaledUpTransform = originalTransform
        scaledUpTransform.scale = originalTransform.scale * Self.pulseScaleFactor
        let halfCycle = Self.pulseDuration
        let threshold = originalTransform.scale.x * (1.0 + Self.pulseScaleFactor) / 2.0

        entity.move(to: scaledUpTransform, relativeTo: entity.parent,
                    duration: halfCycle, timingFunction: .easeInOut)

        // Reverse direction each half-cycle; stop when the entity is removed from the scene.
        Timer.scheduledTimer(withTimeInterval: halfCycle, repeats: true) { timer in
            guard entity.scene != nil else {
                timer.invalidate()
                return
            }
            let target = entity.transform.scale.x > threshold
                ? originalTransform : scaledUpTransform
            entity.move(to: target, relativeTo: entity.parent,
                        duration: halfCycle, timingFunction: .easeInOut)
        }
    }
}

#Preview {
    AROverlayView(detectionResult: DetectionResult(
        plantType: "poison_ivy",
        confidence: 0.87,
        boundingBox: .zero
    ))
}
