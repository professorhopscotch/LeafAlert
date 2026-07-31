import SwiftUI
import AVFoundation

/// Draws a pulsing bounding box over the camera preview to highlight the region a
/// detection came from. Expects Vision-normalised coordinates (origin bottom-left,
/// 0–1 range) and converts them to the view's coordinate space.
///
/// IMPORTANT — what this box does and does not mean. The classifier is whole-image;
/// it does not localise. The rectangle comes from Vision's attention-based saliency,
/// i.e. "the most visually prominent region of the frame", which is usually but NOT
/// always the plant. Treat it as "look around here", never as a precise outline, and
/// keep the copy hedged accordingly.
struct BoundingBoxOverlay: View {
    let boundingBox: CGRect
    let label: String
    let confidence: Float
    /// Preview layer used to convert coordinates. When nil (SwiftUI previews, tests)
    /// a plain linear mapping is used, which is only correct for an unrotated,
    /// uncropped feed.
    var layerBox: PreviewLayerBox?

    /// Pulse animation state.
    @State private var isPulsing = false

    var body: some View {
        GeometryReader { geo in
            let rect = visionToView(boundingBox, in: geo.size)

            ZStack(alignment: .topLeading) {
                // Bounding box rectangle
                RoundedRectangle(cornerRadius: 4)
                    .stroke(boxColor, lineWidth: isPulsing ? 3 : 2)
                    .frame(width: rect.width, height: rect.height)
                    .position(x: rect.midX, y: rect.midY)
                    .shadow(color: boxColor.opacity(0.5), radius: isPulsing ? 8 : 4)

                // Label pill above the box
                Text("\(label) \(Int(confidence * 100))%")
                    .font(.caption2.bold())
                    .foregroundStyle(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(boxColor.opacity(0.85))
                    .clipShape(Capsule())
                    .position(
                        x: max(rect.midX, 50),
                        y: max(rect.minY - 14, 14)
                    )
            }
        }
        .onAppear {
            withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) {
                isPulsing = true
            }
        }
        .allowsHitTesting(false)
    }

    /// Converts a Vision normalised rect (origin bottom-left) to view coordinates.
    ///
    /// The preview layer uses `.resizeAspectFill`, so the landscape sample buffer is
    /// both ROTATED and CROPPED to fill a portrait view. A linear scale ignores both
    /// and lands the box away from what it is pointing at — which for a "where not to
    /// step" indicator is worse than drawing nothing. `layerRectConverted` applies
    /// AVFoundation's own gravity + orientation math, so it stays correct if the
    /// preset, gravity or orientation ever change.
    static func visionToView(_ visionRect: CGRect,
                             in size: CGSize,
                             previewLayer: AVCaptureVideoPreviewLayer?) -> CGRect {
        let raw: CGRect
        if let previewLayer {
            // Vision's origin is bottom-left; metadata-output rects are top-left.
            let metadataRect = CGRect(
                x: visionRect.origin.x,
                y: 1 - visionRect.origin.y - visionRect.height,
                width: visionRect.width,
                height: visionRect.height
            )
            raw = previewLayer.layerRectConverted(fromMetadataOutputRect: metadataRect)
        } else {
            raw = CGRect(
                x: visionRect.origin.x * size.width,
                y: (1 - visionRect.origin.y - visionRect.height) * size.height,
                width: visionRect.width * size.width,
                height: visionRect.height * size.height
            )
        }

        // Clamp to view bounds with some padding. `intersection` returns
        // `CGRect.null` (origin = infinity) when the box lies entirely off-screen;
        // passing that to SwiftUI's `.position` feeds non-finite geometry into
        // CoreAnimation, so collapse it to `.zero` and let the caller skip drawing.
        let clamped = raw
            .insetBy(dx: -4, dy: -4)
            .intersection(CGRect(origin: .zero, size: size))
        return clamped.isNull || clamped.isInfinite ? .zero : clamped
    }

    private func visionToView(_ visionRect: CGRect, in size: CGSize) -> CGRect {
        Self.visionToView(visionRect, in: size, previewLayer: layerBox?.layer)
    }

    private var boxColor: Color {
        confidence > 0.8 ? .red : .orange
    }
}
