//! What the Dawn host decides by itself, tested without a GPU: lowering guest
//! values, accounting, mapped-range windows and the shim's wire formats. The
//! conformance programs (tests/manifest_tests.rs) test it against Dawn.

use std::sync::{Arc, Mutex};

use super::buffer::{BufferState, MapInfo};
use super::ffi::{self, Kind};
use super::lower::{Invalid, Lower, Lowering, usize_from_guest};
use super::texture::texture_bytes;
use super::{Host, adapter};
use crate::abi::enums::{BufferMapState, TextureFormat, map_mode};
use crate::abi::structs as g;

fn host() -> Host {
    Host::new(Arc::clone(&crate::WebGpuCtx::default().inner))
}

#[test]
fn size_t_sentinels_survive_the_trip_from_32_to_64_bits() {
    assert_eq!(usize_from_guest(u32::MAX), usize::MAX);
    assert_eq!(usize_from_guest(7), 7);
    assert_eq!(usize_from_guest(u32::MAX - 1), (u32::MAX - 1) as usize);
}

#[test]
fn texture_accounting_counts_blocks_layers_samples_and_mips() {
    // The budget conformance program's 256x256 RGBA8 texture: 256 KiB.
    assert_eq!(
        texture_bytes(TextureFormat::RGBA8Unorm, 256, 256, 1, 1, 1),
        256 << 10
    );
    // Compressed formats count 4x4 blocks.
    assert_eq!(
        texture_bytes(TextureFormat::BC1RGBAUnorm, 8, 8, 1, 1, 1),
        4 * 8
    );
    assert_eq!(
        texture_bytes(TextureFormat::BC1RGBAUnorm, 5, 5, 1, 1, 1),
        4 * 8
    );
    // Layers and samples multiply; a mip chain adds a third.
    assert_eq!(
        texture_bytes(TextureFormat::R8Unorm, 4, 4, 2, 1, 4),
        4 * 4 * 2 * 4
    );
    assert_eq!(
        texture_bytes(TextureFormat::R32Float, 3, 1, 1, 2, 1),
        12 + 4
    );
}

#[test]
fn a_wgsl_source_lowers_to_its_native_chain_link() {
    let host = host();
    let mut l = Lowering::new(&host, None);
    let wgsl = "@compute @workgroup_size(1) fn main() {}";
    let descriptor = g::ShaderModuleDescriptor {
        chain: g::Chain {
            items: vec![g::Chained::ShaderSourceWGSL(g::ShaderSourceWGSL {
                code: wgsl.to_owned(),
            })],
            links: Vec::new(),
        },
        label: "module".to_owned(),
    };
    let native = descriptor.lower(&mut l).expect("lowers");
    assert!(!native.nextInChain.is_null());
    // SAFETY: the chain `l` owns, a ShaderSourceWGSL as its sType says.
    let source = unsafe {
        assert_eq!((*native.nextInChain).sType, g::ShaderSourceWGSL::S_TYPE);
        assert!((*native.nextInChain).next.is_null());
        &*native.nextInChain.cast::<ffi::WGPUShaderSourceWGSL>()
    };
    assert_eq!(source.code.length, wgsl.len());
    // SAFETY: the view of the decoded string, which is still alive.
    let code = unsafe { std::slice::from_raw_parts(source.code.data.cast::<u8>(), wgsl.len()) };
    assert_eq!(code, wgsl.as_bytes());
    assert_eq!(native.label.length, "module".len());
}

#[test]
fn nothing_a_guest_cannot_vouch_for_reaches_dawn() {
    let host = host();
    let mut l = Lowering::new(&host, None);
    // SPIR-V is refused, as on every host.
    let spirv = g::ShaderModuleDescriptor {
        chain: g::Chain {
            items: vec![g::Chained::ShaderSourceSPIRV(Default::default())],
            links: Vec::new(),
        },
        label: String::new(),
    };
    assert!(matches!(spirv.lower(&mut l), Err(Invalid::Unsupported(_))));
    // A native window handle from guest memory would be a pointer into the
    // host: never.
    let metal = g::SurfaceDescriptor {
        chain: g::Chain {
            items: vec![g::Chained::SurfaceSourceMetalLayer(Default::default())],
            links: Vec::new(),
        },
        label: String::new(),
    };
    assert!(matches!(metal.lower(&mut l), Err(Invalid::Unsupported(_))));
}

#[test]
fn object_handles_are_checked_before_dawn_sees_them() {
    let host = host();
    let mut l = Lowering::new(&host, None);
    // A missing required object, a dead or forged handle, and an absent
    // optional one.
    assert!(matches!(
        l.object(Kind::Buffer, 0, "buffer", false),
        Err(Invalid::Missing("buffer"))
    ));
    assert!(matches!(
        l.object(Kind::Buffer, 0x0100_0001, "buffer", false),
        Err(Invalid::Handle { .. })
    ));
    assert!(l.object(Kind::Buffer, 0, "buffer", true).unwrap().is_null());
    // A pipeline layout may leave a bind group slot empty, and nothing else.
    let (count, slots) = l
        .objects(Kind::BindGroupLayout, &[0, 0], "layouts", true)
        .expect("null slots");
    assert_eq!(count, 2);
    // SAFETY: two slots `l` owns.
    assert!(unsafe { (*slots).is_null() && (*slots.add(1)).is_null() });
    assert!(
        l.objects(Kind::CommandBuffer, &[0], "commands", false)
            .is_err()
    );
    let layout = g::PipelineLayoutDescriptor {
        bind_group_layouts: vec![0, 0x0200_0005],
        ..Default::default()
    };
    assert!(matches!(layout.lower(&mut l), Err(Invalid::Handle { .. })));
}

fn mapped(state: BufferMapState, mode: u64, offset: u64, size: u64) -> BufferState {
    BufferState::for_test(
        256,
        MapInfo {
            state,
            mode,
            offset,
            size,
            generation: 0,
        },
    )
}

#[test]
fn mapped_range_copies_widen_to_what_dawn_maps() {
    let buffer = mapped(BufferMapState::Mapped, map_mode::READ, 0, 64);
    // [5, 15) is read through [0, 16): start on 8, span a multiple of 4.
    assert_eq!(buffer.window(5, 10, false), Some((0, 16, 5)));
    // Never past the mapping, even when that leaves the span unaligned.
    assert_eq!(buffer.window(60, 3, false), Some((56, 8, 4)));
    // Outside the mapping, or a write into a read mapping: no.
    assert_eq!(buffer.window(60, 8, false), None);
    assert_eq!(buffer.window(0, 4, true), None);
    let pending = mapped(BufferMapState::Pending, map_mode::WRITE, 0, 64);
    assert_eq!(pending.window(0, 4, true), None);
    let offset = mapped(BufferMapState::Mapped, map_mode::WRITE, 32, 32);
    assert_eq!(offset.window(0, 4, true), None);
    assert_eq!(offset.window(36, 4, true), Some((32, 8, 4)));
}

#[test]
fn adapter_info_is_encoded_as_the_shim_reads_it() {
    let info = ffi::WGPUAdapterInfo {
        vendor: super::view("vendor"),
        architecture: super::view(""),
        device: super::view("gpu"),
        description: super::view("driver"),
        backendType: 5,
        adapterType: 1,
        vendorID: 0x106b,
        deviceID: 7,
        subgroupMinSize: 32,
        subgroupMaxSize: 64,
        ..Default::default()
    };
    let bytes = adapter::encode_info(&info);
    let word = |at: usize| u32::from_le_bytes(bytes[at * 4..at * 4 + 4].try_into().unwrap());
    assert_eq!(
        (0..10).map(word).collect::<Vec<_>>(),
        [5, 1, 0x106b, 7, 32, 64, 6, 0, 3, 6]
    );
    assert_eq!(&bytes[40..], b"vendorgpudriver");
}

#[test]
fn a_late_map_callback_does_not_undo_an_unmap() {
    let map = Arc::new(Mutex::new(MapInfo {
        state: BufferMapState::Pending,
        mode: map_mode::READ,
        offset: 0,
        size: 4,
        generation: 1,
    }));
    // The guest unmaps (generation 2); the callback of request 1 arrives.
    {
        let mut info = map.lock().unwrap();
        info.state = BufferMapState::Unmapped;
        info.generation = 2;
    }
    super::callbacks::settle_map(&map, 1, true);
    assert_eq!(map.lock().unwrap().state, BufferMapState::Unmapped);
    // A current one does map.
    map.lock().unwrap().state = BufferMapState::Pending;
    super::callbacks::settle_map(&map, 2, true);
    assert_eq!(map.lock().unwrap().state, BufferMapState::Mapped);
}

#[test]
fn a_future_dawn_completes_inside_the_call_leaves_nothing_behind() {
    use super::callbacks::{self, Futures, Pending, Simple};
    let host = host();
    let futures = Futures::default();
    let pending = |future| Pending::new(&host.events, &futures, future);
    // The callback runs before the call that started the future returns...
    let early = callbacks::userdata(Simple {
        pending: pending(7),
        kind: crate::abi::EVENT_QUEUE_WORK_DONE,
    });
    // SAFETY: a context boxed for this callback, and a view of a string.
    unsafe { callbacks::status_message(1, super::view(""), early, std::ptr::null_mut()) };
    futures.track(7, ffi::WGPUFuture { id: 70 });
    assert_eq!(
        futures.len_for_test(),
        0,
        "nothing to wait for, nothing kept"
    );
    assert!(futures.dawn_ids(&[7]).is_empty());
    // ...or after, as usual: tracked, then forgotten.
    let late = callbacks::userdata(Simple {
        pending: pending(8),
        kind: crate::abi::EVENT_QUEUE_WORK_DONE,
    });
    futures.track(8, ffi::WGPUFuture { id: 80 });
    assert_eq!(futures.dawn_ids(&[8]), [80]);
    // SAFETY: as above.
    unsafe { callbacks::status_message(1, super::view(""), late, std::ptr::null_mut()) };
    assert_eq!(futures.len_for_test(), 0);
    assert!(host.events.is_ready(&[7]) && host.events.is_ready(&[8]));
}

#[test]
fn a_refused_map_request_does_not_settle_the_pending_one() {
    let map = Arc::new(Mutex::new(MapInfo {
        state: BufferMapState::Pending,
        mode: map_mode::READ,
        offset: 0,
        size: 4,
        generation: 3,
    }));
    // A second request while one is pending: Dawn fails it at once.
    super::callbacks::settle_map(&map, super::buffer::REFUSED_MAP, false);
    assert_eq!(map.lock().unwrap().state, BufferMapState::Pending);
    // The first one's callback still maps the buffer.
    super::callbacks::settle_map(&map, 3, true);
    assert_eq!(map.lock().unwrap().state, BufferMapState::Mapped);
}
