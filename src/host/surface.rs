//! `WGPUSurface`.
//!
//! A surface presents to whatever the embedder's [`SurfaceProvider`] opened
//! for the guest's selector: a real window, which wgpu presents to directly,
//! or a headless target, for which the host plays swap chain itself (one
//! offscreen texture per frame, handed to the sink on present).

use std::sync::{Arc, Mutex};

use raw_window_handle::{
    DisplayHandle, HandleError, HasDisplayHandle, HasWindowHandle, WindowHandle,
};

use super::texture::TextureState;
use super::{DeviceState, Host, Object, ObjectKind, convert, lock};
use crate::abi::api::SurfaceApi;
use crate::abi::enums::{
    CompositeAlphaMode, PredefinedColorSpace, PresentMode, Status, SurfaceGetCurrentTextureStatus,
    TextureFormat, ToneMappingMode, texture_usage,
};
use crate::abi::structs::{SurfaceConfiguration, SurfaceDescriptor, SurfaceTexture};
use crate::abi::{AbiEnum, Cx, Handle, OutPtr};
use crate::surface::{
    Frame, FrameColorSpace, FrameFormat, FrameSink, FrameToneMapping, SurfaceTarget, WindowTarget,
};

/// Hands wgpu an owning reference to the embedder's window.
struct WindowRef(Arc<dyn WindowTarget>);

impl HasWindowHandle for WindowRef {
    fn window_handle(&self) -> Result<WindowHandle<'_>, HandleError> {
        self.0.window_handle()
    }
}

impl HasDisplayHandle for WindowRef {
    fn display_handle(&self) -> Result<DisplayHandle<'_>, HandleError> {
        self.0.display_handle()
    }
}

enum Backing {
    Window {
        window: Arc<dyn WindowTarget>,
        surface: wgpu::Surface<'static>,
    },
    Headless(Arc<dyn FrameSink>),
}

struct Configured {
    device: Arc<DeviceState>,
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    /// `WGPUTextureUsage` bits the guest asked for.
    usage_bits: u64,
    width: u32,
    height: u32,
    view_formats: Vec<wgpu::TextureFormat>,
    color_space: FrameColorSpace,
    tone_mapping: FrameToneMapping,
}

enum CurrentFrame {
    Window(wgpu::SurfaceTexture),
    Headless(Arc<TextureState>),
}

#[derive(Default)]
struct SurfaceInner {
    configured: Option<Configured>,
    frame: Option<CurrentFrame>,
    presented: u64,
}

impl SurfaceInner {
    /// Ends the configuration, for Unconfigure or a Configure replacing it.
    /// A frame acquired and not presented is destroyed, as the surface
    /// contract says: the guest may still hold its texture, and must not be
    /// able to go on using it. A window's frame goes back to wgpu, which
    /// discards it.
    fn retire(&mut self) {
        if let Some(CurrentFrame::Headless(frame)) = self.frame.take() {
            frame.texture.destroy();
        }
        self.configured = None;
    }
}

pub(crate) struct SurfaceState {
    backing: Backing,
    inner: Mutex<SurfaceInner>,
}

impl Drop for SurfaceState {
    fn drop(&mut self) {
        // As a browser's canvas does when it goes.
        self.inner
            .get_mut()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .retire();
    }
}

impl SurfaceState {
    /// The wgpu surface, when this one presents to a real window.
    pub(crate) fn window_surface(&self) -> Option<&wgpu::Surface<'static>> {
        match &self.backing {
            Backing::Window { surface, .. } => Some(surface),
            Backing::Headless(_) => None,
        }
    }

    fn size(&self) -> (u32, u32) {
        match &self.backing {
            Backing::Window { window, .. } => window.size(),
            Backing::Headless(sink) => sink.size(),
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

/// How long, and how many times, a guest that asks an occluded window for a
/// frame is held back before it is told there is none.
const OCCLUDED_FRAME_DELAY: std::time::Duration = std::time::Duration::from_millis(16);
const OCCLUDED_ATTEMPTS: u32 = 12;

/// A window's frame, from `acquire`. A window nobody can see (hidden, or
/// created a moment ago and not on screen yet) has no frame to give, and
/// says so at once. The wait a visible window's vsync provides is spent
/// here instead, in `wait`: a new window gets the time to appear, and a
/// frame loop that simply tries again does not spin. A guest stopped
/// meanwhile stops waiting, and its call traps.
fn window_frame(
    cx: &Cx<'_>,
    mut acquire: impl FnMut() -> wgpu::CurrentSurfaceTexture,
    mut wait: impl FnMut(),
) -> wgpu::CurrentSurfaceTexture {
    let mut attempt = 0;
    loop {
        match acquire() {
            wgpu::CurrentSurfaceTexture::Occluded
                if attempt < OCCLUDED_ATTEMPTS && !cx.stopped() =>
            {
                attempt += 1;
                wait();
            }
            current => break current,
        }
    }
}

impl Host {
    pub(crate) fn create_surface(
        &self,
        instance: Handle,
        descriptor: &SurfaceDescriptor,
    ) -> Handle {
        let Some(instance_state) = self.instance(instance) else {
            return 0;
        };
        // A WASIX guest has no native window handles to offer, so the only
        // surface source that means anything here is the host-provided one.
        let Some(source) = descriptor.chain.wasix_surface_source_canvas() else {
            super::log(|| {
                "wgpuInstanceCreateSurface: chain WGPUWasixSurfaceSourceCanvas \
                 (webgpu_wasix.h); native window sources are not available to WASIX guests"
                    .to_owned()
            });
            return 0;
        };
        if descriptor.chain.items.len() != 1 {
            super::log(|| {
                "wgpuInstanceCreateSurface: exactly one surface source is allowed".to_owned()
            });
            return 0;
        }
        let Some(target) = self.ctx.surfaces.open(&source.selector) else {
            super::log(|| {
                format!(
                    "wgpuInstanceCreateSurface: the embedder provides no target for {:?}",
                    source.selector
                )
            });
            return 0;
        };
        let backing = match target {
            SurfaceTarget::Headless(sink) => Backing::Headless(sink),
            SurfaceTarget::Window(window) => {
                // The window says which thread may attach a surface to it.
                let (created, outcome) = std::sync::mpsc::channel();
                let instance = Arc::clone(&instance_state);
                let target = WindowRef(Arc::clone(&window));
                window.run_on_owner_thread(Box::new(move || {
                    let _ = created.send(instance.instance.create_surface(target));
                }));
                match outcome.recv() {
                    Ok(Ok(surface)) => Backing::Window { window, surface },
                    Ok(Err(error)) => {
                        super::log(|| format!("wgpuInstanceCreateSurface: {error}"));
                        return 0;
                    }
                    Err(_) => {
                        super::log(|| {
                            "wgpuInstanceCreateSurface: the window's thread is gone".to_owned()
                        });
                        return 0;
                    }
                }
            }
        };
        let state = SurfaceState {
            backing,
            inner: Mutex::new(SurfaceInner::default()),
        };
        self.insert(Object::Surface(Arc::new(state)), instance)
    }

    /// Serialized `WGPUSurfaceCapabilities` for the shim:
    ///
    /// ```text
    /// u64 usages; u32 formatCount, presentModeCount, alphaModeCount, reserved;
    /// u32 formats[]; u32 presentModes[]; u32 alphaModes[];
    /// ```
    pub(crate) fn surface_capabilities(&self, surface: Handle, adapter: Handle) -> Option<Vec<u8>> {
        let surface = self.surface(surface)?;
        let adapter = self.adapter(adapter)?;
        let (usages, formats, present_modes, alpha_modes) = match &surface.backing {
            Backing::Window { surface, .. } => {
                let caps = surface.get_capabilities(&adapter.adapter);
                let formats: Vec<u32> = caps
                    .formats
                    .iter()
                    .map(|&format| convert::texture_format_to_abi(format))
                    .filter(|&format| format != TextureFormat::Undefined)
                    .map(AbiEnum::as_u32)
                    .collect();
                let mut present_modes: Vec<u32> = caps
                    .present_modes
                    .iter()
                    .filter_map(|&mode| convert::present_mode_to_abi(mode))
                    .map(AbiEnum::as_u32)
                    .collect();
                present_modes.dedup();
                let alpha_modes: Vec<u32> = caps
                    .alpha_modes
                    .iter()
                    .map(|&mode| convert::alpha_mode_to_abi(mode).as_u32())
                    .collect();
                (
                    u64::from(caps.usages.bits()),
                    formats,
                    present_modes,
                    alpha_modes,
                )
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

    /// Reads a presented headless frame back and hands it to the sink.
    fn present_headless(
        &self,
        sink: &dyn FrameSink,
        configured: &Configured,
        frame: &TextureState,
        index: u64,
    ) {
        let format = match configured.format {
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb => {
                Some(FrameFormat::Rgba8)
            }
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb => {
                Some(FrameFormat::Bgra8)
            }
            _ => None,
        };
        let pixels = match format {
            Some(_) if sink.wants_pixels() => read_back(&configured.device, &frame.texture),
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
    }
}

/// Copies a four-bytes-per-pixel texture out of the GPU, rows tightly packed.
fn read_back(device: &DeviceState, texture: &wgpu::Texture) -> Option<Vec<u8>> {
    let (width, height) = (texture.width(), texture.height());
    let row = width as usize * 4;
    let padded = row.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize)
        * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as usize;
    let buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wasmer-webgpu headless readback"),
        size: (padded * height as usize) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded as u32),
                rows_per_image: Some(height),
            },
        },
        texture.size(),
    );
    device.queue.submit([encoder.finish()]);
    let (sender, receiver) = std::sync::mpsc::channel();
    buffer.map_async(wgpu::MapMode::Read, .., move |result| {
        let _ = sender.send(result.is_ok());
    });
    device
        .device
        .poll(wgpu::PollType::wait_indefinitely())
        .ok()?;
    if !receiver.recv().ok()? {
        return None;
    }
    let view = buffer.get_mapped_range(..);
    let mut pixels = Vec::with_capacity(row * height as usize);
    for line in view.chunks(padded).take(height as usize) {
        pixels.extend_from_slice(&line[..row]);
    }
    Some(pixels)
}

impl SurfaceApi for Host {
    fn surface_configure(&self, _cx: &mut Cx<'_>, surface: Handle, config: &SurfaceConfiguration) {
        let Some(state) = self.surface(surface) else {
            return;
        };
        // Configuring first unconfigures, as the surface contract says,
        // whatever comes of the new configuration: the current frame is gone
        // even when the device or the configuration is refused below.
        let mut inner = lock(&state.inner);
        inner.retire();
        let Some(device) = self.device(config.device) else {
            return;
        };
        let fail =
            |message: &str| self.validation(&device, format!("wgpuSurfaceConfigure: {message}"));
        let Some(format) = convert::texture_format(config.format) else {
            return fail("a surface configuration needs a format.");
        };
        let Some(usage) = convert::texture_usages(config.usage) else {
            return fail("invalid usage.");
        };
        let Some(view_formats) = config
            .view_formats
            .iter()
            .map(|&format| convert::texture_format(format))
            .collect::<Option<Vec<_>>>()
        else {
            return fail("invalid view format.");
        };
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

        match &state.backing {
            Backing::Window { surface, .. } => {
                // The backend presents a window in sRGB and has no way to
                // ask for anything else. Saying so beats showing the wrong
                // colours.
                if (color_space, tone_mapping) != Default::default() {
                    drop(inner);
                    return fail("this surface presents in sRGB with standard tone mapping only.");
                }
                surface.configure(
                    &device.device,
                    &wgpu::SurfaceConfiguration {
                        usage,
                        format,
                        width: config.width,
                        height: config.height,
                        present_mode: convert::present_mode(config.present_mode),
                        desired_maximum_frame_latency: 2,
                        alpha_mode: convert::alpha_mode(config.alpha_mode),
                        view_formats: view_formats.clone(),
                    },
                );
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
            format,
            usage,
            usage_bits: config.usage,
            width: config.width,
            height: config.height,
            view_formats,
            color_space,
            tone_mapping,
        });
    }

    fn surface_get_current_texture(
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
        if let Some(state) = self.surface(surface) {
            let mut inner = lock(&state.inner);
            let inner = &mut *inner;
            // Asking again before presenting is an error, as is asking an
            // unconfigured surface.
            if let (Some(configured), None) = (&inner.configured, &inner.frame) {
                let instance = self.instance_of(surface);
                let wrap = |texture: wgpu::Texture| {
                    Arc::new(TextureState::from_surface(
                        texture,
                        Arc::clone(&configured.device),
                        configured.usage_bits,
                        Arc::clone(&self.gpu_memory),
                    ))
                };
                match &state.backing {
                    Backing::Window { surface, window } => {
                        use wgpu::CurrentSurfaceTexture as Current;
                        let current = window_frame(
                            cx,
                            || surface.get_current_texture(),
                            || window.wait_presentable(OCCLUDED_FRAME_DELAY),
                        );
                        // `webgpu.h` has no word for a window nobody can
                        // see, so the log keeps the two apart.
                        let why = match &current {
                            Current::Occluded => "the window cannot be seen",
                            Current::Timeout => "none became free in time",
                            _ => "",
                        };
                        let (frame, status) = match current {
                            Current::Success(frame) => {
                                (Some(frame), SurfaceGetCurrentTextureStatus::SuccessOptimal)
                            }
                            Current::Suboptimal(frame) => (
                                Some(frame),
                                SurfaceGetCurrentTextureStatus::SuccessSuboptimal,
                            ),
                            // Nothing to draw to right now; the guest skips a frame.
                            Current::Timeout | Current::Occluded => {
                                (None, SurfaceGetCurrentTextureStatus::Timeout)
                            }
                            Current::Outdated => (None, SurfaceGetCurrentTextureStatus::Outdated),
                            Current::Lost => (None, SurfaceGetCurrentTextureStatus::Lost),
                            Current::Validation => (None, SurfaceGetCurrentTextureStatus::Error),
                        };
                        result.status = status;
                        if frame.is_none() {
                            super::log(|| {
                                let why = if why.is_empty() {
                                    String::new()
                                } else {
                                    format!(": {why}")
                                };
                                format!("wgpuSurfaceGetCurrentTexture: no frame ({status:?}{why})")
                            });
                        }
                        if let Some(frame) = frame {
                            result.texture =
                                self.insert(Object::Texture(wrap(frame.texture.clone())), instance);
                            inner.frame = Some(CurrentFrame::Window(frame));
                        }
                    }
                    Backing::Headless(sink) => {
                        // The size the sink wants may have changed since the
                        // guest configured; tell it to reconfigure.
                        let outdated = sink.size() != (configured.width, configured.height);
                        let texture =
                            configured
                                .device
                                .device
                                .create_texture(&wgpu::TextureDescriptor {
                                    label: Some("wasmer-webgpu headless frame"),
                                    size: wgpu::Extent3d {
                                        width: configured.width,
                                        height: configured.height,
                                        depth_or_array_layers: 1,
                                    },
                                    mip_level_count: 1,
                                    sample_count: 1,
                                    dimension: wgpu::TextureDimension::D2,
                                    format: configured.format,
                                    // COPY_SRC is the host's, for reading frames back.
                                    usage: configured.usage | wgpu::TextureUsages::COPY_SRC,
                                    view_formats: &configured.view_formats,
                                });
                        let texture = wrap(texture);
                        result.texture =
                            self.insert(Object::Texture(Arc::clone(&texture)), instance);
                        result.status = if outdated {
                            SurfaceGetCurrentTextureStatus::SuccessSuboptimal
                        } else {
                            SurfaceGetCurrentTextureStatus::SuccessOptimal
                        };
                        inner.frame = Some(CurrentFrame::Headless(texture));
                    }
                }
            }
        }
        let _ = surface_texture.write(&cx.mem, &result);
    }

    fn surface_present(&self, _cx: &mut Cx<'_>, surface: Handle) -> Status {
        let Some(state) = self.surface(surface) else {
            return Status::Error;
        };
        let mut inner = lock(&state.inner);
        let Some(frame) = inner.frame.take() else {
            return Status::Error;
        };
        let index = inner.presented;
        inner.presented += 1;
        match (frame, &state.backing, &inner.configured) {
            (CurrentFrame::Window(frame), backing, Some(_)) => {
                if let Backing::Window { window, .. } = backing {
                    window.pre_present();
                }
                frame.present();
            }
            (CurrentFrame::Headless(frame), Backing::Headless(sink), Some(configured)) => {
                self.present_headless(sink.as_ref(), configured, &frame, index);
                // Presenting consumes the frame's texture.
                frame.texture.destroy();
            }
            _ => return Status::Error,
        }
        Status::Success
    }

    fn surface_unconfigure(&self, _cx: &mut Cx<'_>, surface: Handle) {
        if let Some(state) = self.surface(surface) {
            lock(&state.inner).retire();
        }
    }

    fn surface_set_label(&self, _cx: &mut Cx<'_>, surface: Handle, _label: &str) {
        let _ = self.surface(surface);
    }

    fn surface_add_ref(&self, _cx: &mut Cx<'_>, surface: Handle) {
        self.add_ref(surface, ObjectKind::Surface);
    }

    fn surface_release(&self, _cx: &mut Cx<'_>, surface: Handle) {
        self.release(surface, ObjectKind::Surface);
    }

    fn surface_wasix_get_size(
        &self,
        cx: &mut Cx<'_>,
        surface: Handle,
        width: u32,
        height: u32,
    ) -> Status {
        let Some(state) = self.surface(surface) else {
            return Status::Error;
        };
        let (w, h) = state.size();
        let ok = (width == 0 || cx.mem.write_u32(width, w).is_ok())
            && (height == 0 || cx.mem.write_u32(height, h).is_ok());
        if ok { Status::Success } else { Status::Error }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abi::GuestMemory;
    use std::cell::Cell;
    use wgpu::CurrentSurfaceTexture::Occluded;

    /// Runs `f` with a call context whose guest is stopped from the check
    /// after `checks_before_stop` on.
    fn with_cx(checks_before_stop: u32, f: impl FnOnce(&Cx<'_>)) {
        let mut store = wasmer::Store::default();
        let memory =
            wasmer::Memory::new(&mut store, wasmer::MemoryType::new(1, None, false)).unwrap();
        let checks = Cell::new(0);
        let stopped = || {
            checks.set(checks.get() + 1);
            checks.get() > checks_before_stop
        };
        f(&Cx::new(GuestMemory::new(memory.view(&store)), &stopped));
    }

    #[test]
    fn an_occluded_window_is_asked_again_a_few_times() {
        with_cx(u32::MAX, |cx| {
            let (mut asked, mut waited) = (0, 0);
            let frame = window_frame(
                cx,
                || {
                    asked += 1;
                    Occluded
                },
                || waited += 1,
            );
            assert!(matches!(frame, Occluded));
            assert_eq!((asked, waited), (OCCLUDED_ATTEMPTS + 1, OCCLUDED_ATTEMPTS));
            assert!(!cx.take_stop(), "a guest that wasn't stopped doesn't trap");
        });
    }

    #[test]
    fn a_guest_stopped_while_its_window_is_occluded_stops_waiting_and_traps() {
        with_cx(2, |cx| {
            let mut waited = 0;
            let frame = window_frame(cx, || Occluded, || waited += 1);
            assert!(matches!(frame, Occluded));
            assert_eq!(waited, 2, "no wait after the check that saw the stop");
            assert!(cx.take_stop(), "the call traps when it returns");
            assert!(!cx.take_stop(), "the stop is taken once");
        });
    }
}
