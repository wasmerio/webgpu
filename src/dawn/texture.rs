//! `WGPUTexture`: creation under the GPU memory budget, and destruction.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use super::ffi::{self, Kind};
use super::lower::Lowering;
use super::{Extra, Host, Native};
use crate::abi::enums::{ErrorType, TextureFormat};
use crate::abi::structs::TextureDescriptor;
use crate::abi::{Cx, Handle};

pub(crate) struct TextureState {
    /// Bytes this object still counts for against the guest's budget.
    accounted: AtomicU64,
    gpu_memory: Arc<AtomicU64>,
}

impl Drop for TextureState {
    fn drop(&mut self) {
        self.release_storage();
    }
}

impl TextureState {
    pub(crate) fn new(bytes: u64, gpu_memory: &Arc<AtomicU64>) -> Self {
        super::charge(gpu_memory, bytes);
        Self {
            accounted: AtomicU64::new(bytes),
            gpu_memory: Arc::clone(gpu_memory),
        }
    }

    pub(crate) fn release_storage(&self) {
        super::refund(&self.gpu_memory, self.accounted.swap(0, Ordering::Relaxed));
    }

    pub(crate) fn of(object: &Native) -> Option<&TextureState> {
        match &object.extra {
            Extra::Texture(state) => Some(state),
            _ => None,
        }
    }
}

/// (bytes, block width, block height) of one texel block of `format`.
pub(crate) fn block(format: TextureFormat) -> (u64, u32, u32) {
    use TextureFormat as F;
    match format {
        F::Undefined => (0, 1, 1),
        F::R8Unorm | F::R8Snorm | F::R8Uint | F::R8Sint | F::Stencil8 => (1, 1, 1),
        F::R16Unorm
        | F::R16Snorm
        | F::R16Uint
        | F::R16Sint
        | F::R16Float
        | F::RG8Unorm
        | F::RG8Snorm
        | F::RG8Uint
        | F::RG8Sint
        | F::Depth16Unorm => (2, 1, 1),
        F::R32Float
        | F::R32Uint
        | F::R32Sint
        | F::RG16Unorm
        | F::RG16Snorm
        | F::RG16Uint
        | F::RG16Sint
        | F::RG16Float
        | F::RGBA8Unorm
        | F::RGBA8UnormSrgb
        | F::RGBA8Snorm
        | F::RGBA8Uint
        | F::RGBA8Sint
        | F::BGRA8Unorm
        | F::BGRA8UnormSrgb
        | F::RGB10A2Uint
        | F::RGB10A2Unorm
        | F::RG11B10Ufloat
        | F::RGB9E5Ufloat
        | F::Depth24Plus
        | F::Depth24PlusStencil8
        | F::Depth32Float => (4, 1, 1),
        F::Depth32FloatStencil8 => (5, 1, 1),
        F::RG32Float
        | F::RG32Uint
        | F::RG32Sint
        | F::RGBA16Unorm
        | F::RGBA16Snorm
        | F::RGBA16Uint
        | F::RGBA16Sint
        | F::RGBA16Float => (8, 1, 1),
        F::RGBA32Float | F::RGBA32Uint | F::RGBA32Sint => (16, 1, 1),
        F::BC1RGBAUnorm
        | F::BC1RGBAUnormSrgb
        | F::BC4RUnorm
        | F::BC4RSnorm
        | F::ETC2RGB8Unorm
        | F::ETC2RGB8UnormSrgb
        | F::ETC2RGB8A1Unorm
        | F::ETC2RGB8A1UnormSrgb
        | F::EACR11Unorm
        | F::EACR11Snorm => (8, 4, 4),
        F::BC2RGBAUnorm
        | F::BC2RGBAUnormSrgb
        | F::BC3RGBAUnorm
        | F::BC3RGBAUnormSrgb
        | F::BC5RGUnorm
        | F::BC5RGSnorm
        | F::BC6HRGBUfloat
        | F::BC6HRGBFloat
        | F::BC7RGBAUnorm
        | F::BC7RGBAUnormSrgb
        | F::ETC2RGBA8Unorm
        | F::ETC2RGBA8UnormSrgb
        | F::EACRG11Unorm
        | F::EACRG11Snorm => (16, 4, 4),
        F::ASTC4x4Unorm | F::ASTC4x4UnormSrgb => (16, 4, 4),
        F::ASTC5x4Unorm | F::ASTC5x4UnormSrgb => (16, 5, 4),
        F::ASTC5x5Unorm | F::ASTC5x5UnormSrgb => (16, 5, 5),
        F::ASTC6x5Unorm | F::ASTC6x5UnormSrgb => (16, 6, 5),
        F::ASTC6x6Unorm | F::ASTC6x6UnormSrgb => (16, 6, 6),
        F::ASTC8x5Unorm | F::ASTC8x5UnormSrgb => (16, 8, 5),
        F::ASTC8x6Unorm | F::ASTC8x6UnormSrgb => (16, 8, 6),
        F::ASTC8x8Unorm | F::ASTC8x8UnormSrgb => (16, 8, 8),
        F::ASTC10x5Unorm | F::ASTC10x5UnormSrgb => (16, 10, 5),
        F::ASTC10x6Unorm | F::ASTC10x6UnormSrgb => (16, 10, 6),
        F::ASTC10x8Unorm | F::ASTC10x8UnormSrgb => (16, 10, 8),
        F::ASTC10x10Unorm | F::ASTC10x10UnormSrgb => (16, 10, 10),
        F::ASTC12x10Unorm | F::ASTC12x10UnormSrgb => (16, 12, 10),
        F::ASTC12x12Unorm | F::ASTC12x12UnormSrgb => (16, 12, 12),
    }
}

/// A rough size for accounting: every mip of every layer and sample.
pub(crate) fn texture_bytes(
    format: TextureFormat,
    width: u32,
    height: u32,
    layers: u32,
    mips: u32,
    samples: u32,
) -> u64 {
    let (bytes, block_width, block_height) = block(format);
    let blocks = u64::from(width.div_ceil(block_width.max(1)))
        .saturating_mul(u64::from(height.div_ceil(block_height.max(1))))
        .saturating_mul(u64::from(layers.max(1)));
    let base = blocks
        .saturating_mul(bytes.max(1))
        .saturating_mul(u64::from(samples.max(1)));
    // A full mip chain adds a third.
    if mips > 1 {
        base.saturating_add(base / 3)
    } else {
        base
    }
}

impl Host {
    pub(crate) fn manual_device_create_texture(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &TextureDescriptor,
    ) -> Handle {
        let Some(parent) = self.object(device, Kind::Device) else {
            return 0;
        };
        let Some(state) = parent.device.clone() else {
            return 0;
        };
        let bytes = texture_bytes(
            descriptor.format,
            descriptor.size.width,
            descriptor.size.height,
            descriptor.size.depth_or_array_layers,
            descriptor.mip_level_count,
            descriptor.sample_count,
        );
        if let Some(budget) = self.ctx.limits.max_gpu_memory_bytes
            && self
                .gpu_memory
                .load(Ordering::Relaxed)
                .saturating_add(bytes)
                > budget
        {
            state.inject(
                ErrorType::OutOfMemory,
                "creating this texture would exceed this guest's GPU memory budget",
            );
            return 0;
        }
        let mut l = Lowering::new(self, Some(&parent));
        let native = match l.lower_ptr(descriptor) {
            Ok(native) => native,
            Err(error) => {
                l.fail("wgpuDeviceCreateTexture", error);
                return 0;
            }
        };
        // SAFETY: a live device and a descriptor that outlives the call.
        let raw = unsafe { ffi::wgpuDeviceCreateTexture(state.raw, native) };
        drop(l);
        if raw.is_null() {
            return 0;
        }
        self.insert(
            Arc::new(Native {
                kind: Kind::Texture,
                raw: raw.cast(),
                device: Some(state),
                extra: Extra::Texture(TextureState::new(bytes, &self.gpu_memory)),
            }),
            self.instance_of(device),
        )
    }

    pub(crate) fn manual_texture_destroy(&self, _cx: &mut Cx<'_>, texture: Handle) {
        if let Some(object) = self.object(texture, Kind::Texture) {
            if let Some(state) = TextureState::of(&object) {
                state.release_storage();
            }
            // SAFETY: a live texture.
            unsafe { ffi::wgpuTextureDestroy(object.raw.cast()) };
        }
    }
}
