//! Where guest surfaces present to.
//!
//! `webgpu.h` says nothing about where a presentable surface comes from, and a
//! WASIX guest has no native window handles to offer. So the guest names a
//! target (the `selector` of `WGPUWasixSurfaceSourceCanvas`) and the embedder
//! decides what that name means through a [`SurfaceProvider`].

use std::sync::{Arc, Mutex};

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// Resolves the selector a guest passes when it creates a surface.
pub trait SurfaceProvider: Send + Sync + 'static {
    /// Returns the target for `selector`, or `None` to make surface creation
    /// fail. The empty selector asks for the default target.
    fn open(&self, selector: &str) -> Option<SurfaceTarget>;
}

/// What a guest surface presents to.
#[derive(Clone)]
pub enum SurfaceTarget {
    /// A native window or view the GPU backend presents to directly.
    Window(Arc<dyn WindowTarget>),
    /// No window: the guest renders into an offscreen texture and each
    /// presented frame is offered to the sink.
    Headless(Arc<dyn FrameSink>),
}

/// A native window a surface can be created on.
pub trait WindowTarget: HasWindowHandle + HasDisplayHandle + Send + Sync + 'static {
    /// Current size in physical pixels.
    fn size(&self) -> (u32, u32);

    /// Runs `work` on the thread that owns the window, and returns once it
    /// has run. Guests create their surfaces on their own threads, and some
    /// platforms (macOS) only let a GPU surface be attached to a window from
    /// the thread that owns it. The default, for windows without such a rule,
    /// runs `work` right here.
    ///
    /// An implementation that can no longer reach its thread drops `work`
    /// without running it; the guest's surface is then not created.
    fn run_on_owner_thread(&self, work: Box<dyn FnOnce() + Send>) {
        work()
    }

    /// Called on the guest's thread right before each frame is handed to the
    /// window system. Some window systems pace a window's redraws by being
    /// told of its presents (winit's `Window::pre_present_notify`). The
    /// default does nothing.
    fn pre_present(&self) {}

    /// Called on the guest's thread when the window had no frame to give
    /// because nothing of it can be seen (it is hidden, covered, or behind a
    /// locked screen). Returns when asking again may be worth it, and after
    /// `timeout` at the latest. A window that knows more than the GPU does
    /// can return early, or tell its guest that it cannot be seen. The
    /// default sleeps for `timeout`.
    fn wait_presentable(&self, timeout: std::time::Duration) {
        std::thread::sleep(timeout);
    }
}

/// Pixel layout of a [`Frame`]: four 8-bit channels per pixel, rows top to
/// bottom, no padding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameFormat {
    Rgba8,
    Bgra8,
}

/// The colour space a frame's values are in: what the guest configured its
/// surface with (`WGPUSurfaceColorManagement`), sRGB unless it said otherwise.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrameColorSpace {
    #[default]
    Srgb,
    DisplayP3,
}

/// How values outside the standard range are meant to be shown.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum FrameToneMapping {
    /// Clamped to the standard dynamic range.
    #[default]
    Standard,
    /// Shown with whatever headroom the display has.
    Extended,
}

/// One frame presented to a headless surface.
pub struct Frame<'a> {
    /// Counts presented frames of this surface, starting at 0.
    pub index: u64,
    pub width: u32,
    pub height: u32,
    pub format: FrameFormat,
    /// How to read the values: nothing is converted on the way here.
    pub color_space: FrameColorSpace,
    pub tone_mapping: FrameToneMapping,
    /// Present only when the sink asked for pixels.
    pub pixels: Option<&'a [u8]>,
}

/// Receives the frames of a headless surface.
pub trait FrameSink: Send + Sync + 'static {
    /// Size the guest is told to configure for, in pixels.
    fn size(&self) -> (u32, u32);

    /// Whether [`FrameSink::present`] wants the pixels. Reading a frame back
    /// from the GPU costs a copy and a wait, so the default is not to.
    fn wants_pixels(&self) -> bool {
        false
    }

    fn present(&self, frame: Frame<'_>);
}

/// The default provider: every selector opens a headless target of a fixed
/// size whose frames are dropped. Surfaces work, nothing is shown.
#[derive(Clone, Debug)]
pub struct HeadlessSurfaces {
    width: u32,
    height: u32,
}

impl HeadlessSurfaces {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

impl Default for HeadlessSurfaces {
    fn default() -> Self {
        Self::new(640, 480)
    }
}

impl SurfaceProvider for HeadlessSurfaces {
    fn open(&self, _selector: &str) -> Option<SurfaceTarget> {
        Some(SurfaceTarget::Headless(Arc::new(DiscardFrames {
            width: self.width,
            height: self.height,
        })))
    }
}

struct DiscardFrames {
    width: u32,
    height: u32,
}

impl FrameSink for DiscardFrames {
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn present(&self, _frame: Frame<'_>) {}
}

/// A captured frame, as kept by [`CaptureFrames`].
#[derive(Clone, Debug)]
pub struct CapturedFrame {
    pub index: u64,
    pub width: u32,
    pub height: u32,
    pub format: FrameFormat,
    pub color_space: FrameColorSpace,
    pub tone_mapping: FrameToneMapping,
    pub pixels: Vec<u8>,
}

/// A headless target that keeps the most recent frame's pixels, for tests,
/// screenshots and server-side rendering.
pub struct CaptureFrames {
    width: u32,
    height: u32,
    last: Mutex<Option<CapturedFrame>>,
}

impl CaptureFrames {
    pub fn new(width: u32, height: u32) -> Arc<Self> {
        Arc::new(Self {
            width,
            height,
            last: Mutex::new(None),
        })
    }

    /// The most recently presented frame, if any.
    pub fn last_frame(&self) -> Option<CapturedFrame> {
        self.last
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone()
    }
}

impl FrameSink for CaptureFrames {
    fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    fn wants_pixels(&self) -> bool {
        true
    }

    fn present(&self, frame: Frame<'_>) {
        let captured = CapturedFrame {
            index: frame.index,
            width: frame.width,
            height: frame.height,
            format: frame.format,
            color_space: frame.color_space,
            tone_mapping: frame.tone_mapping,
            pixels: frame.pixels.map(<[u8]>::to_vec).unwrap_or_default(),
        };
        *self
            .last
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(captured);
    }
}

/// Routes every selector to one fixed target.
pub struct SingleSurface(pub SurfaceTarget);

impl SurfaceProvider for SingleSurface {
    fn open(&self, _selector: &str) -> Option<SurfaceTarget> {
        Some(self.0.clone())
    }
}
