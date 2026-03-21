import SwiftUI

/// Draws a pulsing bounding box over the camera preview to highlight a detected plant.
/// Expects Vision-normalised coordinates (origin bottom-left, 0–1 range) and converts
/// them to the view's coordinate space.
struct BoundingBoxOverlay: View {
    let boundingBox: CGRect
    let label: String
    let confidence: Float

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

    /// Converts Vision normalised rect (origin bottom-left) to view coordinates (origin top-left).
    private func visionToView(_ visionRect: CGRect, in size: CGSize) -> CGRect {
        let x = visionRect.origin.x * size.width
        let y = (1 - visionRect.origin.y - visionRect.height) * size.height
        let w = visionRect.width * size.width
        let h = visionRect.height * size.height

        // Clamp to view bounds with some padding
        let padded = CGRect(x: x, y: y, width: w, height: h)
            .insetBy(dx: -4, dy: -4)
            .intersection(CGRect(origin: .zero, size: size))
        return padded
    }

    private var boxColor: Color {
        confidence > 0.8 ? .red : .orange
    }
}
