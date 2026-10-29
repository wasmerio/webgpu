//! Large `wgpuQueueWriteTexture` uploads through reused staging buffers.
//!
//! wgpu stages every `write_texture` in a new buffer: for frame-sized uploads
//! (video planes, every frame) that is a fresh allocation whose pages fault in
//! on each call, after the bytes were already copied out of guest memory once.
//! Here the guest's rows are copied, through a cache-resident scratch row,
//! into a buffer still mapped from its previous use. The buffer-to-texture
//! copy is submitted at once, so it is ordered like the write it replaces:
//! after earlier queue writes and submissions, before later ones. The buffer
//! maps again when the GPU is done with it.
//!
//! Whatever this does not handle goes to `write_texture`, and so does every
//! write that turns out to be invalid, so that `write_texture` reports it:
//! the staged copy runs under error scopes of its own.
//!
//! The staging buffers are the host's, not the guest's: they are not charged
//! to its GPU memory budget, where they could make the guest's own
//! allocations fail. Instead, across all of a guest's devices they hold at
//! most [`MAXIMUM_BYTES`], and no more than that budget.

use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

use super::device::DeviceState;
use super::lock;
use crate::abi::GuestMemory;

/// Uploads of at least this many bytes of texels use the staging buffers.
const MINIMUM_BYTES: u64 = 256 * 1024;
const ALIGNMENT: u64 = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT as u64;
/// Staging buffer sizes are rounded up to this, so that uploads of nearby
/// sizes share buffers.
const GRANULE: u64 = 1 << 20;
/// Per device. While all are in flight, uploads use wgpu's own staging.
const MAXIMUM_BUFFERS: usize = 8;
/// Per guest, in bytes. Idle buffers are dropped to stay under it.
pub(crate) const MAXIMUM_BYTES: u64 = 64 << 20;
/// A buffer no upload took for this long is dropped (see [`Uploads::trim`]).
const IDLE_LIFETIME: Duration = Duration::from_secs(2);

thread_local! {
    static SCRATCH: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// The bytes one guest's staging buffers hold, across its devices.
#[derive(Debug)]
pub(crate) struct StagingBudget {
    used: AtomicU64,
    limit: u64,
}

impl StagingBudget {
    pub(crate) fn new(gpu_memory_budget: Option<u64>) -> Self {
        Self {
            used: AtomicU64::new(0),
            limit: gpu_memory_budget.map_or(MAXIMUM_BYTES, |budget| budget.min(MAXIMUM_BYTES)),
        }
    }

    fn reserve(&self, bytes: u64) -> bool {
        self.used
            .try_update(Ordering::AcqRel, Ordering::Acquire, |used| {
                used.checked_add(bytes).filter(|&total| total <= self.limit)
            })
            .is_ok()
    }

    fn release(&self, bytes: u64) {
        self.used.fetch_sub(bytes, Ordering::AcqRel);
    }

    fn available(&self) -> u64 {
        self.limit.saturating_sub(self.used.load(Ordering::Acquire))
    }
}

#[derive(Clone)]
struct Staging {
    buffer: wgpu::Buffer,
    size: u64,
    /// Mapped and not in use.
    ready: Arc<AtomicBool>,
    /// When an upload last took it.
    taken_at: Instant,
}

impl Staging {
    fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Acquire)
    }
}

/// One device's staging buffers.
pub(crate) struct Uploads {
    budget: Arc<StagingBudget>,
    buffers: Mutex<Vec<Staging>>,
}

impl std::fmt::Debug for Uploads {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("Uploads")
    }
}

impl Drop for Uploads {
    fn drop(&mut self) {
        self.clear();
    }
}

impl Uploads {
    pub(crate) fn new(budget: Arc<StagingBudget>) -> Self {
        Self {
            budget,
            buffers: Mutex::new(Vec::new()),
        }
    }

    /// Uploads `data` like `wgpuQueueWriteTexture`. False leaves the write to
    /// the caller, untouched, including any error it has to report.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn write_texture(
        &self,
        device: &DeviceState,
        memory: &GuestMemory<'_>,
        destination: wgpu::TexelCopyTextureInfo<'_>,
        data: u32,
        data_size: u32,
        layout: wgpu::TexelCopyBufferLayout,
        size: wgpu::Extent3d,
    ) -> bool {
        let Some(rows) = Rows::of(
            &Destination::of(&destination),
            memory.size(),
            data,
            data_size,
            &layout,
            &size,
        ) else {
            return false;
        };
        if rows.bytes * rows.count < MINIMUM_BYTES {
            return false;
        }
        let Some(staging) = self.take(device, rows.staged_bytes()) else {
            return false;
        };
        // Some invalid destinations only wgpu can see: a destroyed texture,
        // one whose creation failed. Their errors are captured here, and the
        // write goes to write_texture, which reports it. A submission with an
        // invalid command buffer runs nothing, so the order of queue writes is
        // kept. wgpu reports some failures by panicking; those are caught too.
        let submitted = device.catching(|| {
            {
                let mut view = staging
                    .buffer
                    .slice(..rows.staged_bytes())
                    .get_mapped_range_mut();
                // The mapped range may be write-combined memory: each row is
                // read out of guest memory into a small scratch row that stays
                // in cache, then copied over in one go.
                SCRATCH.with_borrow_mut(|scratch| {
                    scratch.resize(rows.bytes as usize, 0);
                    for row in 0..rows.count {
                        // In bounds: Rows::of checked against the memory's size.
                        let _ = memory.read_into(rows.source(row), scratch);
                        let at = (row * rows.staged_stride) as usize;
                        view.slice(at..at + scratch.len()).copy_from_slice(scratch);
                    }
                });
            }
            staging.buffer.unmap();
            let mut encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("write_texture upload"),
                    });
            encoder.copy_buffer_to_texture(
                wgpu::TexelCopyBufferInfo {
                    buffer: &staging.buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(rows.staged_stride as u32),
                        rows_per_image: Some(size.height),
                    },
                },
                destination,
                size,
            );
            device.queue.submit([encoder.finish()]);
            let ready = Arc::clone(&staging.ready);
            staging
                .buffer
                .slice(..)
                .map_async(wgpu::MapMode::Write, move |result| {
                    // A buffer that fails to map (device lost) is never used again.
                    if result.is_ok() {
                        ready.store(true, Ordering::Release);
                    }
                });
        });
        if let Err((_, message)) = submitted {
            super::log(|| format!("staged texture upload failed, using write_texture: {message}"));
            self.forget(&staging);
            // Should the copy have run after all, repeating it is harmless.
            return false;
        }
        true
    }

    /// Drops the idle buffers no upload took for [`IDLE_LIFETIME`], so that a
    /// guest that stopped uploading doesn't keep their memory. The host calls
    /// this whenever it polls the device; a poll never waits for an upload
    /// that holds the list, it just leaves trimming to the next one.
    pub(crate) fn trim(&self, now: Instant) {
        let mut buffers = match self.buffers.try_lock() {
            Ok(buffers) => buffers,
            Err(TryLockError::Poisoned(poisoned)) => poisoned.into_inner(),
            Err(TryLockError::WouldBlock) => return,
        };
        buffers.retain(|staging| {
            let idle = staging.is_ready()
                && now.saturating_duration_since(staging.taken_at) >= IDLE_LIFETIME;
            if idle {
                self.budget.release(staging.size);
            }
            !idle
        });
    }

    /// Drops every staging buffer: the device was destroyed or lost.
    pub(crate) fn clear(&self) {
        let mut buffers = lock(&self.buffers);
        for staging in buffers.drain(..) {
            self.budget.release(staging.size);
        }
    }

    /// The smallest mapped buffer of at least `size` bytes, or a new one.
    fn take(&self, device: &DeviceState, size: u64) -> Option<Staging> {
        if device.link.lost.load(Ordering::Acquire) {
            // Device loss destroys the buffers; using one would fail.
            self.clear();
            return None;
        }
        let mut buffers = lock(&self.buffers);
        let found = buffers
            .iter_mut()
            .filter(|staging| staging.size >= size && staging.is_ready())
            .min_by_key(|staging| staging.size);
        if let Some(staging) = found {
            staging.ready.store(false, Ordering::Release);
            staging.taken_at = Instant::now();
            return Some(staging.clone());
        }
        let rounded = size.div_ceil(GRANULE) * GRANULE;
        // Idle buffers, all too small for this upload (left from an earlier,
        // smaller stream), are the only room this device can make. Another
        // device of the guest may hold the rest.
        let idle: u64 = buffers
            .iter()
            .filter(|staging| staging.is_ready())
            .map(|staging| staging.size)
            .sum();
        if rounded > self.budget.available() + idle {
            return None;
        }
        if buffers.len() >= MAXIMUM_BUFFERS {
            self.evict(&mut buffers)?;
        }
        while !self.budget.reserve(rounded) {
            self.evict(&mut buffers)?;
        }
        // A failure is this cache's, not the guest's: no error reaches it.
        let created = device.catching(|| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("write_texture staging"),
                size: rounded,
                usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: true,
            })
        });
        let Ok(buffer) = created else {
            self.budget.release(rounded);
            return None;
        };
        let staging = Staging {
            buffer,
            size: rounded,
            ready: Arc::new(AtomicBool::new(false)),
            taken_at: Instant::now(),
        };
        buffers.push(staging.clone());
        Some(staging)
    }

    /// Drops the largest idle buffer, or returns None if all are in flight.
    fn evict(&self, buffers: &mut Vec<Staging>) -> Option<()> {
        let index = buffers
            .iter()
            .enumerate()
            .filter(|(_, staging)| staging.is_ready())
            .max_by_key(|(_, staging)| staging.size)
            .map(|(index, _)| index)?;
        self.budget.release(buffers.swap_remove(index).size);
        Some(())
    }

    fn forget(&self, staging: &Staging) {
        let mut buffers = lock(&self.buffers);
        if let Some(index) = buffers
            .iter()
            .position(|other| Arc::ptr_eq(&other.ready, &staging.ready))
        {
            self.budget.release(buffers.swap_remove(index).size);
        }
    }

    #[cfg(test)]
    fn buffer_sizes(&self) -> Vec<u64> {
        lock(&self.buffers)
            .iter()
            .map(|staging| staging.size)
            .collect()
    }
}

/// What decides whether a write to a texture can be staged.
#[derive(Clone, Copy, Debug)]
struct Destination {
    format: wgpu::TextureFormat,
    size: wgpu::Extent3d,
    dimension: wgpu::TextureDimension,
    mip_level_count: u32,
    sample_count: u32,
    usage: wgpu::TextureUsages,
    mip_level: u32,
    origin: wgpu::Origin3d,
    aspect: wgpu::TextureAspect,
}

impl Destination {
    fn of(info: &wgpu::TexelCopyTextureInfo<'_>) -> Self {
        let texture = info.texture;
        Self {
            format: texture.format(),
            size: texture.size(),
            dimension: texture.dimension(),
            mip_level_count: texture.mip_level_count(),
            sample_count: texture.sample_count(),
            usage: texture.usage(),
            mip_level: info.mip_level,
            origin: info.origin,
            aspect: info.aspect,
        }
    }
}

/// Where the rows of an upload are, in guest memory and staged.
#[derive(Debug, PartialEq)]
struct Rows {
    /// Guest address of the first row.
    first: u64,
    stride: u64,
    count: u64,
    /// Bytes of texels in a row.
    bytes: u64,
    /// `bytes`, aligned as a buffer-to-texture copy requires.
    staged_stride: u64,
}

impl Rows {
    /// The rows of a write this module can stage: a single image of a color
    /// texture, with a destination and data layout that are valid as far as
    /// the texture's descriptor tells. None otherwise.
    fn of(
        destination: &Destination,
        memory_size: u64,
        data: u32,
        data_size: u32,
        layout: &wgpu::TexelCopyBufferLayout,
        size: &wgpu::Extent3d,
    ) -> Option<Self> {
        let format = destination.format;
        if format.is_depth_stencil_format()
            || format.block_dimensions() != (1, 1)
            || destination.aspect != wgpu::TextureAspect::All
            || destination.sample_count != 1
            || !destination.usage.contains(wgpu::TextureUsages::COPY_DST)
            || size.depth_or_array_layers != 1
            || size.width == 0
            || size.height == 0
            // No texture has 32 levels; a level past them can't be measured.
            || destination.mip_level >= destination.mip_level_count.min(32)
        {
            return None;
        }
        // None for a multi-planar format.
        let texel = format.block_copy_size(Some(wgpu::TextureAspect::All))?;
        let extent = destination
            .size
            .mip_level_size(destination.mip_level, destination.dimension);
        let fits = |origin: u32, length: u32, limit: u32| {
            origin.checked_add(length).is_some_and(|end| end <= limit)
        };
        if !fits(destination.origin.x, size.width, extent.width)
            || !fits(destination.origin.y, size.height, extent.height)
            || !fits(destination.origin.z, 1, extent.depth_or_array_layers)
        {
            return None;
        }
        let bytes = u64::from(size.width) * u64::from(texel);
        let count = u64::from(size.height);
        // A write without a stride is one row: too small to stage.
        let stride = u64::from(layout.bytes_per_row?);
        if stride < bytes || layout.rows_per_image.is_some_and(|rows| rows < size.height) {
            return None;
        }
        let end = (count - 1)
            .checked_mul(stride)?
            .checked_add(bytes)?
            .checked_add(layout.offset)?;
        // All the data must be in guest memory, as write_texture reads it all.
        if end > u64::from(data_size) || u64::from(data) + u64::from(data_size) > memory_size {
            return None;
        }
        Some(Self {
            // No overflow: the offset is within the data, in guest memory.
            first: u64::from(data) + layout.offset,
            stride,
            count,
            bytes,
            staged_stride: bytes.div_ceil(ALIGNMENT) * ALIGNMENT,
        })
    }

    fn staged_bytes(&self) -> u64 {
        self.staged_stride * self.count
    }

    fn source(&self, row: u64) -> u32 {
        (self.first + row * self.stride) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_staging_budget_is_the_smaller_of_its_cap_and_the_gpu_budget() {
        assert_eq!(StagingBudget::new(None).limit, MAXIMUM_BYTES);
        assert_eq!(StagingBudget::new(Some(4 << 20)).limit, 4 << 20);
        assert_eq!(StagingBudget::new(Some(1 << 40)).limit, MAXIMUM_BYTES);

        let budget = StagingBudget::new(Some(3 << 20));
        assert!(budget.reserve(2 << 20));
        assert!(!budget.reserve(2 << 20), "over the limit");
        assert_eq!(budget.available(), 1 << 20);
        assert!(budget.reserve(1 << 20));
        budget.release(2 << 20);
        assert!(budget.reserve(2 << 20));
        assert!(!budget.reserve(u64::MAX), "no overflow");
    }

    const MEMORY: u64 = 64 << 20;
    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 256;
    const SIZE: wgpu::Extent3d = wgpu::Extent3d {
        width: WIDTH,
        height: HEIGHT,
        depth_or_array_layers: 1,
    };

    /// A 512 x 256 RGBA8 texture, written whole.
    fn destination() -> Destination {
        Destination {
            format: wgpu::TextureFormat::Rgba8Unorm,
            size: SIZE,
            dimension: wgpu::TextureDimension::D2,
            mip_level_count: 1,
            sample_count: 1,
            usage: wgpu::TextureUsages::COPY_DST,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        }
    }

    fn layout(offset: u64, stride: u32) -> wgpu::TexelCopyBufferLayout {
        wgpu::TexelCopyBufferLayout {
            offset,
            bytes_per_row: Some(stride),
            rows_per_image: None,
        }
    }

    #[test]
    fn a_valid_write_is_staged_row_by_row() {
        // 300 x 256 texels at (100, 0), rows 1300 bytes apart, 36 bytes in.
        let size = wgpu::Extent3d { width: 300, ..SIZE };
        let at = Destination {
            origin: wgpu::Origin3d { x: 100, y: 0, z: 0 },
            ..destination()
        };
        let data_size = 36 + 1300 * 255 + 1200;
        let rows = Rows::of(&at, MEMORY, 4096, data_size, &layout(36, 1300), &size).unwrap();
        assert_eq!(
            rows,
            Rows {
                first: 4096 + 36,
                stride: 1300,
                count: 256,
                bytes: 1200,
                staged_stride: 1280,
            }
        );
        assert_eq!(rows.source(2), 4096 + 36 + 2600);
        // Mip level 1 of layer 2 of a 1024 x 512 array.
        let layered = Destination {
            size: wgpu::Extent3d {
                width: 2 * WIDTH,
                height: 2 * HEIGHT,
                depth_or_array_layers: 3,
            },
            mip_level_count: 2,
            mip_level: 1,
            origin: wgpu::Origin3d { x: 0, y: 0, z: 2 },
            ..destination()
        };
        let whole = WIDTH * HEIGHT * 4;
        assert!(Rows::of(&layered, MEMORY, 0, whole, &layout(0, WIDTH * 4), &SIZE).is_some());
    }

    #[test]
    fn an_invalid_write_is_left_to_write_texture() {
        let whole = WIDTH * HEIGHT * 4;
        let staged = |destination: Destination,
                      data: u32,
                      data_size: u32,
                      layout: wgpu::TexelCopyBufferLayout,
                      size: wgpu::Extent3d| {
            Rows::of(&destination, MEMORY, data, data_size, &layout, &size).is_some()
        };
        let valid = layout(0, WIDTH * 4);
        assert!(staged(destination(), 0, whole, valid, SIZE));
        let cases: [(&str, Destination); 9] = [
            (
                "no CopyDst",
                Destination {
                    usage: wgpu::TextureUsages::COPY_SRC,
                    ..destination()
                },
            ),
            (
                "one texel past the edge",
                Destination {
                    origin: wgpu::Origin3d { x: 1, y: 0, z: 0 },
                    ..destination()
                },
            ),
            (
                "a layer past the last",
                Destination {
                    origin: wgpu::Origin3d { x: 0, y: 0, z: 1 },
                    ..destination()
                },
            ),
            (
                "a mip level past the last",
                Destination {
                    mip_level: 1,
                    ..destination()
                },
            ),
            (
                "mip level 33 of a texture that claims 40",
                Destination {
                    mip_level_count: 40,
                    mip_level: 33,
                    ..destination()
                },
            ),
            (
                "the depth aspect of a color texture",
                Destination {
                    aspect: wgpu::TextureAspect::DepthOnly,
                    ..destination()
                },
            ),
            (
                "a depth format",
                Destination {
                    format: wgpu::TextureFormat::Depth32Float,
                    ..destination()
                },
            ),
            (
                "a compressed format",
                Destination {
                    format: wgpu::TextureFormat::Bc1RgbaUnorm,
                    ..destination()
                },
            ),
            (
                "a multisampled texture",
                Destination {
                    sample_count: 4,
                    ..destination()
                },
            ),
        ];
        for (what, destination) in cases {
            assert!(!staged(destination, 0, whole, valid, SIZE), "{what}");
        }
        let two_layers = wgpu::Extent3d {
            depth_or_array_layers: 2,
            ..SIZE
        };
        let no_stride = wgpu::TexelCopyBufferLayout {
            bytes_per_row: None,
            ..valid
        };
        let short_images = wgpu::TexelCopyBufferLayout {
            rows_per_image: Some(HEIGHT - 1),
            ..valid
        };
        let memory = MEMORY as u32;
        let huge = wgpu::Extent3d {
            width: u32::MAX,
            height: u32::MAX,
            depth_or_array_layers: 1,
        };
        let cases = [
            ("two images", 0, whole, valid, two_layers),
            ("no stride", 0, whole, no_stride, SIZE),
            (
                "a stride shorter than a row",
                0,
                whole,
                layout(0, WIDTH * 4 - 4),
                SIZE,
            ),
            (
                "fewer rows per image than rows",
                0,
                whole,
                short_images,
                SIZE,
            ),
            ("one byte too little data", 0, whole - 1, valid, SIZE),
            (
                "data past the end of memory",
                memory - whole + 1,
                whole,
                valid,
                SIZE,
            ),
            (
                "an offset past the data",
                0,
                whole,
                layout(u64::MAX - 8, WIDTH * 4),
                SIZE,
            ),
            ("the largest extent there is", 0, whole, valid, huge),
        ];
        for (what, data, data_size, layout, size) in cases {
            assert!(
                !staged(destination(), data, data_size, layout, size),
                "{what}"
            );
        }
    }
}

#[cfg(test)]
mod native_gpu_tests {
    // Run explicitly with --ignored: they need a GPU adapter.
    use super::*;
    use crate::WebGpuCtx;
    use crate::host::Host;

    const WIDTH: u32 = 512;
    const HEIGHT: u32 = 256;

    fn device(budget: Option<u64>) -> (Host, Arc<DeviceState>) {
        let mut builder = WebGpuCtx::builder();
        if let Some(budget) = budget {
            builder = builder.max_gpu_memory_bytes(budget);
        }
        let host = Host::new(Arc::clone(&builder.build().inner));
        let native = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = pollster::block_on(native.request_adapter(&wgpu::RequestAdapterOptions {
            force_fallback_adapter: crate::host::test_fallback_adapter(),
            ..Default::default()
        }))
        .expect("native GPU adapter is required, never silently skip");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("native GPU device");
        let state = DeviceState::new(&host, device, queue, adapter.get_info(), 0);
        (host, state)
    }

    fn texture_with(
        device: &DeviceState,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> wgpu::Texture {
        device.device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
    }

    fn texture(device: &DeviceState) -> wgpu::Texture {
        texture_with(
            device,
            wgpu::TextureFormat::Rgba8Unorm,
            wgpu::TextureUsages::COPY_DST,
        )
    }

    /// Uploads the whole texture from guest address 0.
    fn upload(device: &DeviceState, memory: &GuestMemory<'_>, texture: &wgpu::Texture) -> bool {
        device.uploads.write_texture(
            device,
            memory,
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            0,
            WIDTH * HEIGHT * 4,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(WIDTH * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: WIDTH,
                height: HEIGHT,
                depth_or_array_layers: 1,
            },
        )
    }

    fn with_memory(f: impl FnOnce(&GuestMemory<'_>)) {
        let mut store = wasmer::Store::default();
        let memory =
            wasmer::Memory::new(&mut store, wasmer::MemoryType::new(16, None, false)).unwrap();
        f(&GuestMemory::new(memory.view(&store)));
    }

    #[test]
    #[ignore = "requires a native GPU"]
    fn a_mapped_buffer_is_reused_and_the_budget_is_returned() {
        let (host, device) = device(None);
        let texture = texture(&device);
        with_memory(|memory| {
            assert!(upload(&device, memory, &texture));
            assert_eq!(device.uploads.buffer_sizes(), [GRANULE]);
            // In flight: a second upload gets a buffer of its own.
            assert!(upload(&device, memory, &texture));
            assert_eq!(device.uploads.buffer_sizes(), [GRANULE, GRANULE]);
            device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            // Both mapped again: the next uploads reuse them.
            assert!(upload(&device, memory, &texture));
            assert!(upload(&device, memory, &texture));
            assert_eq!(device.uploads.buffer_sizes(), [GRANULE, GRANULE]);
        });
        assert_eq!(host.staging.used.load(Ordering::Acquire), 2 * GRANULE);
        drop(device);
        assert_eq!(host.staging.used.load(Ordering::Acquire), 0);
    }

    #[test]
    #[ignore = "requires a native GPU"]
    fn the_largest_extent_is_left_to_write_texture() {
        let (_host, device) = device(None);
        let texture = texture(&device);
        with_memory(|memory| {
            let staged = device.uploads.write_texture(
                &device,
                memory,
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                0,
                WIDTH * HEIGHT * 4,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(u32::MAX),
                    rows_per_image: None,
                },
                wgpu::Extent3d {
                    width: u32::MAX,
                    height: u32::MAX,
                    depth_or_array_layers: 1,
                },
            );
            assert!(!staged);
        });
        assert!(device.uploads.buffer_sizes().is_empty());
    }

    #[test]
    #[ignore = "requires a native GPU"]
    fn idle_buffers_are_dropped_and_busy_ones_kept() {
        let (host, device) = device(None);
        let texture = texture(&device);
        with_memory(|memory| {
            assert!(upload(&device, memory, &texture));
            let later = Instant::now() + IDLE_LIFETIME;
            // In flight: kept, however long ago it was taken.
            device.uploads.trim(later);
            assert_eq!(device.uploads.buffer_sizes(), [GRANULE]);
            device
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .unwrap();
            // Idle, but taken recently: kept.
            device.uploads.trim(Instant::now());
            assert_eq!(device.uploads.buffer_sizes(), [GRANULE]);
            device.uploads.trim(later);
            assert!(device.uploads.buffer_sizes().is_empty());
        });
        assert_eq!(host.staging.used.load(Ordering::Acquire), 0);
    }

    #[test]
    #[ignore = "requires a native GPU"]
    fn a_small_gpu_budget_leaves_uploads_to_write_texture() {
        let (host, device) = device(Some(GRANULE / 2));
        let texture = texture(&device);
        with_memory(|memory| assert!(!upload(&device, memory, &texture)));
        assert!(device.uploads.buffer_sizes().is_empty());
        assert_eq!(host.staging.used.load(Ordering::Acquire), 0);
    }

    #[test]
    #[ignore = "requires a native GPU"]
    fn a_destroyed_device_drops_its_buffers_at_once() {
        let (host, device) = device(None);
        let texture = texture(&device);
        with_memory(|memory| {
            assert!(upload(&device, memory, &texture));
            device.destroy();
            assert!(device.uploads.buffer_sizes().is_empty());
            assert_eq!(host.staging.used.load(Ordering::Acquire), 0);
            assert!(!upload(&device, memory, &texture));
        });
        assert!(device.uploads.buffer_sizes().is_empty());
    }

    #[test]
    #[ignore = "requires a native GPU"]
    fn a_texture_only_wgpu_knows_is_invalid_is_left_to_write_texture() {
        let (host, device) = device(None);
        // Storage binding is not allowed for an sRGB format: creation fails.
        let failed = texture_with(
            &device,
            wgpu::TextureFormat::Rgba8UnormSrgb,
            wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::STORAGE_BINDING,
        );
        assert_eq!(
            host.events.take_for_test().len(),
            1,
            "the texture is invalid"
        );
        let destroyed = texture(&device);
        destroyed.destroy();
        with_memory(|memory| {
            for texture in [&failed, &destroyed] {
                assert!(!upload(&device, memory, texture));
                assert!(
                    host.events.take_for_test().is_empty(),
                    "the guest heard of the staged copy"
                );
            }
        });
        assert!(device.uploads.buffer_sizes().is_empty());
        assert_eq!(host.staging.used.load(Ordering::Acquire), 0);
    }
}

/// What staging costs and saves, on the GPU to measure:
/// `WEBGPU_MEASURE=1 cargo test --release --features cli --lib upload::measure
/// -- --ignored --nocapture` (`WGPU_BACKEND=vulkan` on Linux for Vulkan).
/// Without `WEBGPU_MEASURE` it does nothing, so that running every ignored
/// test (the GPU tests) doesn't run it.
#[cfg(test)]
mod measure {
    use std::time::{Duration, Instant};

    use super::*;
    use crate::WebGpuCtx;
    use crate::host::Host;

    struct Plane {
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        texel: u32,
    }

    const FRAMES: usize = 240;

    fn device() -> (Host, Arc<DeviceState>, wgpu::Backend) {
        let host = Host::new(Arc::clone(&WebGpuCtx::default().inner));
        let native = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter =
            pollster::block_on(native.request_adapter(&Default::default())).expect("a GPU adapter");
        let backend = adapter.get_info().backend;
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("a device");
        let state = DeviceState::new(&host, device, queue, adapter.get_info(), 0);
        (host, state, backend)
    }

    /// Per-frame CPU time of the writes, at the median and 95th percentile,
    /// the whole run's time once the GPU caught up, and how many writes were
    /// staged.
    fn run(
        device: &Arc<DeviceState>,
        memory: &GuestMemory<'_>,
        planes: &[Plane],
        staged: bool,
    ) -> (Duration, Duration, Duration, usize) {
        let textures: Vec<_> = planes
            .iter()
            .map(|plane| {
                device.device.create_texture(&wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d {
                        width: plane.width,
                        height: plane.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: plane.format,
                    usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
            })
            .collect();
        let mut times = Vec::with_capacity(FRAMES);
        let mut count = 0;
        let started = Instant::now();
        for _ in 0..FRAMES {
            let frame = Instant::now();
            for (plane, texture) in planes.iter().zip(&textures) {
                let destination = wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                };
                let stride = plane.width * plane.texel;
                let data_size = stride * plane.height;
                let layout = wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(stride),
                    rows_per_image: None,
                };
                let size = wgpu::Extent3d {
                    width: plane.width,
                    height: plane.height,
                    depth_or_array_layers: 1,
                };
                if staged
                    && device.uploads.write_texture(
                        device,
                        memory,
                        destination,
                        0,
                        data_size,
                        layout,
                        size,
                    )
                {
                    count += 1;
                } else {
                    // What wgpuQueueWriteTexture does without staging.
                    let bytes = memory.read_vec(0, u64::from(data_size)).unwrap();
                    device
                        .queue
                        .write_texture(destination, &bytes, layout, size);
                }
            }
            times.push(frame.elapsed());
            // The guest's own frame: one submission, and a poll as its loop would.
            device.queue.submit([]);
            device.device.poll(wgpu::PollType::Poll).unwrap();
        }
        device
            .device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        let total = started.elapsed();
        times.sort();
        (times[FRAMES / 2], times[FRAMES * 95 / 100], total, count)
    }

    #[test]
    #[ignore = "measures; needs a GPU and --release to mean anything"]
    fn measure() {
        if std::env::var_os("WEBGPU_MEASURE").is_none() {
            return;
        }
        let (_host, device, backend) = device();
        let mut store = wasmer::Store::default();
        let memory =
            wasmer::Memory::new(&mut store, wasmer::MemoryType::new(512, None, false)).unwrap();
        let memory = GuestMemory::new(memory.view(&store));
        let rgba = |width, height| Plane {
            format: wgpu::TextureFormat::Rgba8Unorm,
            width,
            height,
            texel: 4,
        };
        let workloads: [(&str, Vec<Plane>); 4] = [
            ("RGBA 1280x720", vec![rgba(1280, 720)]),
            ("RGBA 1920x1080", vec![rgba(1920, 1080)]),
            (
                "NV12 1920x1080 (Y + UV planes)",
                vec![
                    Plane {
                        format: wgpu::TextureFormat::R8Unorm,
                        width: 1920,
                        height: 1080,
                        texel: 1,
                    },
                    Plane {
                        format: wgpu::TextureFormat::Rg8Unorm,
                        width: 960,
                        height: 540,
                        texel: 2,
                    },
                ],
            ),
            ("RGBA 512x256 (just over 256 KiB)", vec![rgba(512, 256)]),
        ];
        println!("backend: {backend:?}, {FRAMES} frames each");
        println!(
            "| Workload | Path | Writes, median | Writes, p95 | Whole run | Staged writes | Extra submissions |"
        );
        println!("|---|---|---|---|---|---|---|");
        for (name, planes) in &workloads {
            for staged in [false, true] {
                // A warm-up run, then the measured one.
                run(&device, &memory, planes, staged);
                let (median, p95, total, count) = run(&device, &memory, planes, staged);
                println!(
                    "| {name} | {} | {:.2} ms | {:.2} ms | {:.0} ms | {count} of {} | {count} |",
                    if staged { "staged" } else { "write_texture" },
                    median.as_secs_f64() * 1e3,
                    p95.as_secs_f64() * 1e3,
                    total.as_secs_f64() * 1e3,
                    planes.len() * FRAMES,
                );
            }
        }
        println!("staging buffers held: {:?}", device.uploads.buffer_sizes());
    }
}
