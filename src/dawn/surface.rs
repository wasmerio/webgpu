//! `WGPUSurface`.
//!
//! A surface presents to whatever the embedder's [`SurfaceProvider`] opened
//! for the guest's selector: a real window, which Dawn presents to directly,
//! or a headless target, for which the host plays swap chain itself (one
//! offscreen texture per frame, handed to the sink on present).
//!
//! [`SurfaceProvider`]: crate::surface::SurfaceProvider

use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use super::adapter::AdapterState;
use super::ffi::{self, Kind};
use super::lower::Lowering;
use super::lower_gen::raise_surface_texture;
use super::texture::TextureState;
use super::{DeviceState, Extra, Host, Native, lock, log};
use crate::abi::enums::{
    CallbackMode, CompositeAlphaMode, ErrorType, MapAsyncStatus, PredefinedColorSpace, PresentMode,
    Status, SurfaceGetCurrentTextureStatus, TextureFormat, ToneMappingMode, map_mode,
    texture_usage,
};
use crate::abi::structs::{SurfaceConfiguration, SurfaceDescriptor, SurfaceTexture};
use crate::abi::{AbiEnum, Cx, Handle, OutPtr};
use crate::surface::{
    Frame, FrameColorSpace, FrameFormat, FrameSink, FrameToneMapping, SurfaceTarget, WindowTarget,
};

enum Backing {
    Window {
        window: Arc<dyn WindowTarget>,
        /// Dawn's surface; the owning [`Native`] holds the reference.
        raw: ffi::WGPUSurface,
    },
    Headless(Arc<dyn FrameSink>),
}

struct Configured {
    device: Arc<DeviceState>,
    format: TextureFormat,
    usage: u64,
    width: u32,
    height: u32,
    view_formats: Vec<u32>,
    color_space: FrameColorSpace,
    tone_mapping: FrameToneMapping,
}

#[derive(Default)]
struct SurfaceInner {
    configured: Option<Configured>,
    /// The frame handed out and not presented yet.
    frame: Option<Arc<Native>>,
    presented: u64,
    /// Kept between frames: the buffer headless frames are read back through.
    readback: Option<(u64, ffi::WGPUBuffer)>,
}

pub(crate) struct SurfaceState {
    backing: Backing,
    inner: Mutex<SurfaceInner>,
}

// SAFETY: the raw handles are Dawn objects, which may move between threads.
unsafe impl Send for SurfaceState {}
unsafe impl Sync for SurfaceState {}

impl Drop for SurfaceState {
    fn drop(&mut self) {
        let inner = self.inner.get_mut().unwrap_or_else(|p| p.into_inner());
        if let Some((_, buffer)) = inner.readback.take() {
            // SAFETY: the surface's own reference.
            unsafe { ffi::wgpuBufferRelease(buffer) };
        }
    }
}

impl SurfaceState {
    fn size(&self) -> (u32, u32) {
        match &self.backing {
            Backing::Window { window, .. } => window.size(),
            Backing::Headless(sink) => sink.size(),
        }
    }

    fn of(object: &Native) -> Option<&Arc<SurfaceState>> {
        match &object.extra {
            Extra::Surface(state) => Some(state),
            _ => None,
        }
    }
}

/// Formats a headless surface offers, preferred first.
const HEADLESS_FORMATS: [TextureFormat; 5] = [
    TextureFormat::BGRA8Unorm,
    TextureFormat::RGBA8Unorm,
    TextureFormat::BGRA8UnormSrgb,
    TextureFormat::RGBA8UnormSrgb,
    TextureFormat::RGBA16Float,
];

/// `textureUsage` a headless frame gets on top of what the guest asked for:
/// the host reads frames back.
const HOST_FRAME_USAGE: u64 = texture_usage::COPY_SRC;

/// Makes a Dawn surface for a window, on the thread that owns it.
fn window_surface(
    dawn: ffi::WGPUInstance,
    window: &Arc<dyn WindowTarget>,
) -> Result<ffi::WGPUSurface, String> {
    use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
    let window_handle = window
        .window_handle()
        .map_err(|error| error.to_string())?
        .as_raw();
    let display_handle = window
        .display_handle()
        .map_err(|error| error.to_string())?
        .as_raw();
    let create = |source: *const ffi::WGPUChainedStruct| {
        let descriptor = ffi::WGPUSurfaceDescriptor {
            nextInChain: source.cast_mut(),
            ..Default::default()
        };
        // SAFETY: a live instance, and a source that outlives the call.
        let raw = unsafe { ffi::wgpuInstanceCreateSurface(dawn, &descriptor) };
        if raw.is_null() {
            Err("Dawn could not make a surface for the window".to_owned())
        } else {
            Ok(raw)
        }
    };
    use crate::abi::structs as g;
    match (window_handle, display_handle) {
        #[cfg(target_os = "macos")]
        (RawWindowHandle::AppKit(handle), _) => {
            let layer = metal_layer(handle.ns_view.as_ptr());
            let mut source = ffi::WGPUSurfaceSourceMetalLayer {
                layer,
                ..Default::default()
            };
            source.chain.sType = g::SurfaceSourceMetalLayer::S_TYPE;
            create((&source as *const ffi::WGPUSurfaceSourceMetalLayer).cast())
        }
        (RawWindowHandle::Xlib(handle), RawDisplayHandle::Xlib(display)) => {
            let mut source = ffi::WGPUSurfaceSourceXlibWindow {
                display: display
                    .display
                    .map_or(std::ptr::null_mut(), |display| display.as_ptr()),
                // `c_ulong`, which is 32 bits on some targets.
                #[allow(clippy::useless_conversion)]
                window: u64::from(handle.window),
                ..Default::default()
            };
            source.chain.sType = g::SurfaceSourceXlibWindow::S_TYPE;
            create((&source as *const ffi::WGPUSurfaceSourceXlibWindow).cast())
        }
        (RawWindowHandle::Xcb(handle), RawDisplayHandle::Xcb(display)) => {
            let mut source = ffi::WGPUSurfaceSourceXCBWindow {
                connection: display
                    .connection
                    .map_or(std::ptr::null_mut(), |connection| connection.as_ptr()),
                window: handle.window.get(),
                ..Default::default()
            };
            source.chain.sType = g::SurfaceSourceXCBWindow::S_TYPE;
            create((&source as *const ffi::WGPUSurfaceSourceXCBWindow).cast())
        }
        (RawWindowHandle::Wayland(handle), RawDisplayHandle::Wayland(display)) => {
            let mut source = ffi::WGPUSurfaceSourceWaylandSurface {
                display: display.display.as_ptr(),
                surface: handle.surface.as_ptr(),
                ..Default::default()
            };
            source.chain.sType = g::SurfaceSourceWaylandSurface::S_TYPE;
            create((&source as *const ffi::WGPUSurfaceSourceWaylandSurface).cast())
        }
        (RawWindowHandle::Win32(handle), _) => {
            let mut source = ffi::WGPUSurfaceSourceWindowsHWND {
                hwnd: handle.hwnd.get() as *mut c_void,
                hinstance: handle.hinstance.map_or(std::ptr::null_mut(), |instance| {
                    instance.get() as *mut c_void
                }),
                ..Default::default()
            };
            source.chain.sType = g::SurfaceSourceWindowsHWND::S_TYPE;
            create((&source as *const ffi::WGPUSurfaceSourceWindowsHWND).cast())
        }
        (window, _) => Err(format!(
            "windows of this kind are not supported: {window:?}"
        )),
    }
}

/// A `CAMetalLayer` backing `view`, made the view's layer: what Dawn presents
/// to on macOS. Runs on the view's (main) thread.
#[cfg(target_os = "macos")]
fn metal_layer(view: *mut c_void) -> *mut c_void {
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send};
    // SAFETY: AppKit calls on a live NSView, on its thread.
    unsafe {
        let view = view.cast::<AnyObject>();
        let layer: *mut AnyObject = msg_send![class!(CAMetalLayer), new];
        let _: () = msg_send![view, setWantsLayer: true];
        let _: () = msg_send![view, setLayer: layer];
        let window: *mut AnyObject = msg_send![view, window];
        if !window.is_null() {
            let scale: f64 = msg_send![window, backingScaleFactor];
            let _: () = msg_send![layer, setContentsScale: scale];
        }
        layer.cast()
    }
}

impl Host {
    pub(crate) fn manual_instance_create_surface(
        &self,
        _cx: &mut Cx<'_>,
        instance: Handle,
        descriptor: &SurfaceDescriptor,
    ) -> Handle {
        if self.object(instance, Kind::Instance).is_none() {
            return 0;
        }
        // A WASIX guest has no native window handles to offer, so the only
        // surface source that means anything here is the host-provided one.
        let Some(source) = descriptor.chain.wasix_surface_source_canvas() else {
            log(|| {
                "wgpuInstanceCreateSurface: chain WGPUWasixSurfaceSourceCanvas \
                 (webgpu_wasix.h); native window sources are not available to WASIX guests"
                    .to_owned()
            });
            return 0;
        };
        if descriptor.chain.items.len() != 1 {
            log(|| "wgpuInstanceCreateSurface: exactly one surface source is allowed".to_owned());
            return 0;
        }
        let Some(target) = self.ctx.surfaces.open(&source.selector) else {
            log(|| {
                format!(
                    "wgpuInstanceCreateSurface: the embedder provides no target for {:?}",
                    source.selector
                )
            });
            return 0;
        };
        let (backing, raw) = match target {
            SurfaceTarget::Headless(sink) => (Backing::Headless(sink), std::ptr::null_mut()),
            SurfaceTarget::Window(window) => {
                let Some(dawn) = self.dawn() else {
                    return 0;
                };
                // The window says which thread may attach a surface to it.
                let (created, outcome) = std::sync::mpsc::channel();
                let target = Arc::clone(&window);
                let dawn = DawnPtr(dawn);
                window.run_on_owner_thread(Box::new(move || {
                    let dawn = dawn;
                    let _ = created.send(window_surface(dawn.0, &target).map(DawnPtr));
                }));
                match outcome.recv() {
                    Ok(Ok(raw)) => (Backing::Window { window, raw: raw.0 }, raw.0.cast()),
                    Ok(Err(error)) => {
                        log(|| format!("wgpuInstanceCreateSurface: {error}"));
                        return 0;
                    }
                    Err(_) => {
                        log(|| "wgpuInstanceCreateSurface: the window's thread is gone".to_owned());
                        return 0;
                    }
                }
            }
        };
        let state = SurfaceState {
            backing,
            inner: Mutex::new(SurfaceInner::default()),
        };
        self.insert(
            Arc::new(Native {
                kind: Kind::Surface,
                raw,
                device: None,
                extra: Extra::Surface(Arc::new(state)),
            }),
            instance,
        )
    }

    /// Serialized `WGPUSurfaceCapabilities` for the shim:
    ///
    /// ```text
    /// u64 usages; u32 formatCount, presentModeCount, alphaModeCount, reserved;
    /// u32 formats[]; u32 presentModes[]; u32 alphaModes[];
    /// ```
    pub(crate) fn surface_capabilities(&self, surface: Handle, adapter: Handle) -> Option<Vec<u8>> {
        let object = self.object(surface, Kind::Surface)?;
        let state = SurfaceState::of(&object)?;
        let adapter = self.object(adapter, Kind::Adapter)?;
        let adapter = AdapterState::of(&adapter)?;
        let (usages, formats, present_modes, alpha_modes) = match &state.backing {
            Backing::Window { raw, .. } => {
                let mut caps = ffi::WGPUSurfaceCapabilities::default();
                // SAFETY: live surface and adapter, and a struct for Dawn to fill.
                if unsafe { ffi::wgpuSurfaceGetCapabilities(*raw, adapter.raw, &mut caps) }
                    != Status::Success.as_u32()
                {
                    return None;
                }
                let list = |ptr: *const u32, count: usize| -> Vec<u32> {
                    if ptr.is_null() {
                        return Vec::new();
                    }
                    // SAFETY: `count` values Dawn filled in.
                    unsafe { std::slice::from_raw_parts(ptr, count) }.to_vec()
                };
                let formats: Vec<u32> = list(caps.formats, caps.formatCount)
                    .into_iter()
                    .filter(|&format| TextureFormat::from_u32(format).is_some())
                    .collect();
                let present_modes: Vec<u32> = list(caps.presentModes, caps.presentModeCount)
                    .into_iter()
                    .filter(|&mode| PresentMode::from_u32(mode).is_some())
                    .collect();
                let alpha_modes: Vec<u32> = list(caps.alphaModes, caps.alphaModeCount)
                    .into_iter()
                    .filter(|&mode| CompositeAlphaMode::from_u32(mode).is_some())
                    .collect();
                let usages = caps.usages;
                // SAFETY: what Dawn filled in.
                unsafe { ffi::wgpuSurfaceCapabilitiesFreeMembers(caps) };
                (usages, formats, present_modes, alpha_modes)
            }
            Backing::Headless(_) => (
                texture_usage::RENDER_ATTACHMENT
                    | texture_usage::COPY_SRC
                    | texture_usage::COPY_DST
                    | texture_usage::TEXTURE_BINDING,
                HEADLESS_FORMATS
                    .iter()
                    .map(|format| format.as_u32())
                    .collect(),
                vec![PresentMode::Fifo.as_u32(), PresentMode::Immediate.as_u32()],
                vec![
                    CompositeAlphaMode::Opaque.as_u32(),
                    CompositeAlphaMode::Premultiplied.as_u32(),
                ],
            ),
        };
        let mut out = Vec::new();
        out.extend_from_slice(&usages.to_le_bytes());
        for count in [formats.len(), present_modes.len(), alpha_modes.len(), 0] {
            out.extend_from_slice(&(count as u32).to_le_bytes());
        }
        for value in formats.iter().chain(&present_modes).chain(&alpha_modes) {
            out.extend_from_slice(&value.to_le_bytes());
        }
        Some(out)
    }

    pub(crate) fn manual_surface_configure(
        &self,
        _cx: &mut Cx<'_>,
        surface: Handle,
        config: &SurfaceConfiguration,
    ) {
        let Some(object) = self.object(surface, Kind::Surface) else {
            return;
        };
        let Some(state) = SurfaceState::of(&object) else {
            return;
        };
        let Some(device_object) = self.object(config.device, Kind::Device) else {
            return;
        };
        let Some(device) = device_object.device.clone() else {
            return;
        };
        let fail = |message: &str| {
            device.inject(
                ErrorType::Validation,
                &format!("wgpuSurfaceConfigure: {message}"),
            )
        };
        if config.format == TextureFormat::Undefined {
            return fail("a surface configuration needs a format.");
        }
        if config.width == 0 || config.height == 0 {
            return fail("the size must not be zero.");
        }
        if config.usage & texture_usage::TRANSIENT_ATTACHMENT != 0 {
            return fail("a surface's textures are presented, so they cannot be transient.");
        }
        // `WGPUSurfaceColorManagement`: sRGB with standard tone mapping
        // unless the guest chained something else.
        let (color_space, tone_mapping) = match config.chain.surface_color_management() {
            Some(management) => (
                match management.color_space {
                    PredefinedColorSpace::SRGB => FrameColorSpace::Srgb,
                    PredefinedColorSpace::DisplayP3 => FrameColorSpace::DisplayP3,
                },
                match management.tone_mapping_mode {
                    ToneMappingMode::Standard => FrameToneMapping::Standard,
                    ToneMappingMode::Extended => FrameToneMapping::Extended,
                },
            ),
            None => Default::default(),
        };

        let mut inner = lock(&state.inner);
        // Reconfiguring first unconfigures: the current frame is gone.
        inner.frame = None;
        inner.configured = None;
        match &state.backing {
            Backing::Window { raw, .. } => {
                // Saying so beats showing the wrong colours.
                if (color_space, tone_mapping) != Default::default() {
                    drop(inner);
                    return fail("this surface presents in sRGB with standard tone mapping only.");
                }
                let mut l = Lowering::new(self, Some(&device_object));
                let mut plain = config.clone();
                plain.chain = Default::default();
                match l.lower_ptr(&plain) {
                    // SAFETY: a live surface and a configuration owned by `l`.
                    Ok(native) => unsafe { ffi::wgpuSurfaceConfigure(*raw, native) },
                    Err(error) => {
                        drop(inner);
                        return l.fail("wgpuSurfaceConfigure", error);
                    }
                }
            }
            Backing::Headless(_) => {
                if !HEADLESS_FORMATS.contains(&config.format) {
                    drop(inner);
                    return fail("this surface does not support that format.");
                }
            }
        }
        inner.configured = Some(Configured {
            device,
            format: config.format,
            usage: config.usage,
            width: config.width,
            height: config.height,
            view_formats: config
                .view_formats
                .iter()
                .map(|format| format.as_u32())
                .collect(),
            color_space,
            tone_mapping,
        });
    }

    pub(crate) fn manual_surface_get_current_texture(
        &self,
        cx: &mut Cx<'_>,
        surface: Handle,
        surface_texture: OutPtr<SurfaceTexture>,
    ) {
        if surface_texture.is_null() {
            return;
        }
        let mut result = SurfaceTexture {
            chain: Default::default(),
            texture: 0,
            status: SurfaceGetCurrentTextureStatus::Error,
        };
        if let Some(object) = self.object(surface, Kind::Surface)
            && let Some(state) = SurfaceState::of(&object)
        {
            let mut inner = lock(&state.inner);
            let inner = &mut *inner;
            // Asking again before presenting is an error, as is asking an
            // unconfigured surface.
            if let (Some(configured), None) = (&inner.configured, &inner.frame) {
                let instance = self.instance_of(surface);
                let frame = match &state.backing {
                    Backing::Window { raw, .. } => {
                        let mut native = ffi::WGPUSurfaceTexture::default();
                        // SAFETY: a live, configured surface.
                        unsafe { ffi::wgpuSurfaceGetCurrentTexture(*raw, &mut native) };
                        let raised = raise_surface_texture(&native);
                        result.status = raised.status;
                        (!native.texture.is_null()).then_some(native.texture)
                    }
                    Backing::Headless(sink) => {
                        // The size the sink wants may have changed since the
                        // guest configured; tell it to reconfigure.
                        let outdated = sink.size() != (configured.width, configured.height);
                        result.status = if outdated {
                            SurfaceGetCurrentTextureStatus::SuccessSuboptimal
                        } else {
                            SurfaceGetCurrentTextureStatus::SuccessOptimal
                        };
                        let descriptor = ffi::WGPUTextureDescriptor {
                            usage: configured.usage | HOST_FRAME_USAGE,
                            dimension: crate::abi::enums::TextureDimension::D2.as_u32(),
                            size: ffi::WGPUExtent3D {
                                width: configured.width,
                                height: configured.height,
                                depthOrArrayLayers: 1,
                            },
                            format: configured.format.as_u32(),
                            mipLevelCount: 1,
                            sampleCount: 1,
                            viewFormatCount: configured.view_formats.len(),
                            viewFormats: configured.view_formats.as_ptr(),
                            ..Default::default()
                        };
                        // SAFETY: a live device and a descriptor that
                        // outlives the call.
                        let raw = unsafe {
                            ffi::wgpuDeviceCreateTexture(configured.device.raw, &descriptor)
                        };
                        (!raw.is_null()).then_some(raw)
                    }
                };
                match frame {
                    Some(raw) => {
                        let frame = Arc::new(Native {
                            kind: Kind::Texture,
                            raw: raw.cast(),
                            device: Some(Arc::clone(&configured.device)),
                            // The swap chain's storage, not the guest's.
                            extra: Extra::Texture(TextureState::new(0, &self.gpu_memory)),
                        });
                        result.texture = self.insert(Arc::clone(&frame), instance);
                        inner.frame = Some(frame);
                    }
                    None => {
                        if matches!(
                            result.status,
                            SurfaceGetCurrentTextureStatus::SuccessOptimal
                                | SurfaceGetCurrentTextureStatus::SuccessSuboptimal
                        ) {
                            result.status = SurfaceGetCurrentTextureStatus::Error;
                        }
                        log(|| {
                            format!(
                                "wgpuSurfaceGetCurrentTexture: no frame ({:?})",
                                result.status
                            )
                        });
                    }
                }
            }
        }
        let _ = surface_texture.write(&cx.mem, &result);
    }

    pub(crate) fn manual_surface_present(&self, _cx: &mut Cx<'_>, surface: Handle) -> Status {
        let Some(object) = self.object(surface, Kind::Surface) else {
            return Status::Error;
        };
        let Some(state) = SurfaceState::of(&object) else {
            return Status::Error;
        };
        let mut inner = lock(&state.inner);
        let Some(frame) = inner.frame.take() else {
            return Status::Error;
        };
        let index = inner.presented;
        inner.presented += 1;
        match &state.backing {
            Backing::Window { window, raw } => {
                window.pre_present();
                // SAFETY: a live surface with a frame out.
                let status = unsafe { ffi::wgpuSurfacePresent(*raw) };
                Status::from_u32(status).unwrap_or(Status::Error)
            }
            Backing::Headless(sink) => {
                let inner = &mut *inner;
                let Some(configured) = &inner.configured else {
                    return Status::Error;
                };
                let format = match configured.format {
                    TextureFormat::RGBA8Unorm | TextureFormat::RGBA8UnormSrgb => {
                        Some(FrameFormat::Rgba8)
                    }
                    TextureFormat::BGRA8Unorm | TextureFormat::BGRA8UnormSrgb => {
                        Some(FrameFormat::Bgra8)
                    }
                    _ => None,
                };
                let pixels = match format {
                    Some(_) if sink.wants_pixels() => {
                        read_back(self, configured, frame.raw.cast(), &mut inner.readback)
                    }
                    _ => None,
                };
                sink.present(Frame {
                    index,
                    width: configured.width,
                    height: configured.height,
                    format: format.unwrap_or(FrameFormat::Rgba8),
                    color_space: configured.color_space,
                    tone_mapping: configured.tone_mapping,
                    pixels: pixels.as_deref(),
                });
                // Presenting consumes the frame's texture.
                // SAFETY: a live texture.
                unsafe { ffi::wgpuTextureDestroy(frame.raw.cast()) };
                Status::Success
            }
        }
    }

    pub(crate) fn manual_surface_unconfigure(&self, _cx: &mut Cx<'_>, surface: Handle) {
        if let Some(object) = self.object(surface, Kind::Surface)
            && let Some(state) = SurfaceState::of(&object)
        {
            let mut inner = lock(&state.inner);
            inner.frame = None;
            inner.configured = None;
            if let Backing::Window { raw, .. } = &state.backing {
                // SAFETY: a live surface.
                unsafe { ffi::wgpuSurfaceUnconfigure(*raw) };
            }
        }
    }

    pub(crate) fn manual_surface_set_label(&self, _cx: &mut Cx<'_>, surface: Handle, label: &str) {
        if let Some(object) = self.object(surface, Kind::Surface)
            && let Some(state) = SurfaceState::of(&object)
            && let Backing::Window { raw, .. } = &state.backing
        {
            // SAFETY: a live surface; the view outlives the call.
            unsafe { ffi::wgpuSurfaceSetLabel(*raw, super::view(label)) };
        }
    }

    pub(crate) fn manual_surface_wasix_get_size(
        &self,
        cx: &mut Cx<'_>,
        surface: Handle,
        width: u32,
        height: u32,
    ) -> Status {
        let Some(object) = self.object(surface, Kind::Surface) else {
            return Status::Error;
        };
        let Some(state) = SurfaceState::of(&object) else {
            return Status::Error;
        };
        let (w, h) = state.size();
        let ok = (width == 0 || cx.mem.write_u32(width, w).is_ok())
            && (height == 0 || cx.mem.write_u32(height, h).is_ok());
        if ok { Status::Success } else { Status::Error }
    }
}

/// A Dawn pointer, to move to the window's thread and back.
struct DawnPtr<T>(T);
// SAFETY: Dawn instances and surfaces may move between threads.
unsafe impl<T> Send for DawnPtr<T> {}

/// Result slot of a synchronous map.
struct Mapped(u32);

unsafe extern "C" fn on_mapped(
    status: u32,
    _message: ffi::WGPUStringView,
    userdata1: *mut c_void,
    _userdata2: *mut c_void,
) {
    // SAFETY: the slot the caller waits on, alive until WaitAny returns.
    unsafe { (*userdata1.cast::<Mapped>()).0 = status };
}

/// Copies a four-bytes-per-pixel frame out of the GPU, rows tightly packed,
/// through a buffer the surface keeps between frames.
fn read_back(
    host: &Host,
    configured: &Configured,
    texture: ffi::WGPUTexture,
    cache: &mut Option<(u64, ffi::WGPUBuffer)>,
) -> Option<Vec<u8>> {
    let dawn = host.dawn()?;
    let device = configured.device.raw;
    let (width, height) = (configured.width, configured.height);
    let row = width as usize * 4;
    let padded = row.div_ceil(256) * 256;
    let size = (padded * height as usize) as u64;
    let buffer = match cache.take() {
        Some((cached, buffer)) if cached == size => buffer,
        other => {
            if let Some((_, buffer)) = other {
                // SAFETY: the cache's reference.
                unsafe { ffi::wgpuBufferRelease(buffer) };
            }
            let descriptor = ffi::WGPUBufferDescriptor {
                usage: crate::abi::enums::buffer_usage::COPY_DST
                    | crate::abi::enums::buffer_usage::MAP_READ,
                size,
                ..Default::default()
            };
            // SAFETY: a live device and a descriptor that outlives the call.
            unsafe { ffi::wgpuDeviceCreateBuffer(device, &descriptor) }
        }
    };
    if buffer.is_null() {
        return None;
    }
    let mut pixels = None;
    // SAFETY: live device, texture, buffer and instance; every struct
    // outlives the call that uses it, and the map completes before the wait
    // returns.
    unsafe {
        let encoder = ffi::wgpuDeviceCreateCommandEncoder(device, std::ptr::null());
        let source = ffi::WGPUTexelCopyTextureInfo {
            texture,
            ..Default::default()
        };
        let destination = ffi::WGPUTexelCopyBufferInfo {
            layout: ffi::WGPUTexelCopyBufferLayout {
                offset: 0,
                bytesPerRow: padded as u32,
                rowsPerImage: height,
            },
            buffer,
        };
        let extent = ffi::WGPUExtent3D {
            width,
            height,
            depthOrArrayLayers: 1,
        };
        ffi::wgpuCommandEncoderCopyTextureToBuffer(encoder, &source, &destination, &extent);
        let commands = ffi::wgpuCommandEncoderFinish(encoder, std::ptr::null());
        ffi::wgpuQueueSubmit(configured.device.queue, 1, &commands);
        ffi::wgpuCommandBufferRelease(commands);
        ffi::wgpuCommandEncoderRelease(encoder);
        let mut mapped = Mapped(MapAsyncStatus::Error.as_u32());
        let info = ffi::WGPUBufferMapCallbackInfo {
            mode: CallbackMode::WaitAnyOnly.as_u32(),
            callback: Some(on_mapped),
            userdata1: (&mut mapped as *mut Mapped).cast(),
            ..Default::default()
        };
        let future = ffi::wgpuBufferMapAsync(buffer, map_mode::READ, 0, size as usize, info);
        let mut wait = ffi::WGPUFutureWaitInfo {
            future,
            completed: 0,
        };
        ffi::wgpuInstanceWaitAny(dawn, 1, &mut wait, u64::MAX);
        if mapped.0 == MapAsyncStatus::Success.as_u32() {
            let data = ffi::wgpuBufferGetConstMappedRange(buffer, 0, size as usize);
            if !data.is_null() {
                let view = std::slice::from_raw_parts(data.cast::<u8>(), size as usize);
                let mut out = Vec::with_capacity(row * height as usize);
                for line in view.chunks(padded).take(height as usize) {
                    out.extend_from_slice(&line[..row]);
                }
                pixels = Some(out);
            }
            ffi::wgpuBufferUnmap(buffer);
        }
    }
    *cache = Some((size, buffer));
    pixels
}
