import SwiftUI
import AVFoundation

/// Shares the live `AVCaptureVideoPreviewLayer` with overlay views.
///
/// Overlays that draw on top of the camera feed (e.g. `BoundingBoxOverlay`) must
/// convert Vision's normalised, buffer-relative coordinates into view coordinates.
/// That conversion depends on the layer's `videoGravity` (this app uses
/// `.resizeAspectFill`, which CROPS the video to fill the screen) and on the
/// rotation between the landscape sample buffer and the portrait view. Doing that
/// by hand is where overlays silently drift off-target, so we hand the layer to the
/// overlay and let AVFoundation convert.
///
/// Deliberately not `@Published`: the layer is assigned while SwiftUI is building
/// the view, and publishing there would re-enter that update pass. Overlays read it
/// lazily inside `GeometryReader`, by which point it is set.
final class PreviewLayerBox: ObservableObject {
    private(set) weak var layer: AVCaptureVideoPreviewLayer?

    func set(_ layer: AVCaptureVideoPreviewLayer) {
        self.layer = layer
    }
}

/// Live camera preview that shares the CaptureEngine's AVCaptureSession.
/// Displays the rear camera feed with video gravity set to fill the view.
struct CameraPreviewView: UIViewRepresentable {
    let session: AVCaptureSession
    /// Optional sink for the preview layer so overlays can convert coordinates.
    var layerBox: PreviewLayerBox?

    func makeUIView(context: Context) -> PreviewUIView {
        let view = PreviewUIView()
        view.previewLayer.session = session
        view.previewLayer.videoGravity = .resizeAspectFill
        layerBox?.set(view.previewLayer)
        return view
    }

    func updateUIView(_ uiView: PreviewUIView, context: Context) {
        uiView.previewLayer.session = session
        layerBox?.set(uiView.previewLayer)
    }

    /// Custom UIView subclass that uses AVCaptureVideoPreviewLayer as its backing layer.
    class PreviewUIView: UIView {
        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        var previewLayer: AVCaptureVideoPreviewLayer {
            layer as! AVCaptureVideoPreviewLayer
        }

        override func layoutSubviews() {
            super.layoutSubviews()
            previewLayer.frame = bounds
        }
    }
}
