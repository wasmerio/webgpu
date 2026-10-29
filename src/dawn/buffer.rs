//! `WGPUBuffer`.
//!
//! Dawn keeps the map state machine; the host keeps what the shim needs to
//! know of the current mapping (`wasmer_wgpu_buffer_map_info`) and the
//! buffer's share of the GPU memory budget. Mapped bytes go between Dawn's
//! mapping and guest memory in one copy, Wasmer's.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use super::callbacks::{self, Map, Pending};
use super::ffi::{self, Kind};
use super::lower::Lowering;
use super::{Extra, Host, Native, Record, charge, exclusive, lock, refund, shared};
use crate::abi::enums::{
    BufferMapState, CallbackMode, ErrorType, MapAsyncStatus, Status, map_mode,
};
use crate::abi::structs::BufferDescriptor;
use crate::abi::{AbiEnum, Cx, EVENT_BUFFER_MAP, Handle};

/// The generation of a map request the host knows Dawn will refuse: never
/// the buffer's current one.
pub(super) const REFUSED_MAP: u64 = u64::MAX;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MapInfo {
    pub(crate) state: BufferMapState,
    /// `WGPUMapMode` bits of the current (or pending) mapping.
    pub(crate) mode: u64,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    /// Counts map requests, so a late callback can tell it is stale.
    pub(crate) generation: u64,
}

pub(crate) struct BufferState {
    pub(crate) size: u64,
    pub(crate) map: Arc<Mutex<MapInfo>>,
    /// Held shared while bytes are copied out of the mapping, and
    /// exclusively while they are copied into it (the host never writes
    /// bytes another thread reads), while Dawn unmaps or destroys the buffer
    /// (either frees the mapping, and Dawn keeps nothing alive for a pointer
    /// it handed out), and from a map request's bookkeeping until Dawn has
    /// the request (so Dawn sees requests, unmaps and destroys in the order
    /// the bookkeeping does). Not `map`: Dawn runs the map callback, which
    /// takes that, inside MapAsync, Unmap and Destroy.
    access: RwLock<()>,
    /// Bytes this object still counts for against the guest's budget.
    accounted: AtomicU64,
    gpu_memory: Arc<AtomicU64>,
}

impl Drop for BufferState {
    fn drop(&mut self) {
        self.release_storage();
    }
}

impl BufferState {
    /// Gives the object's bytes back to the guest's budget, once: when the
    /// guest destroys it, or else when its last reference goes.
    pub(crate) fn release_storage(&self) {
        refund(&self.gpu_memory, self.accounted.swap(0, Ordering::Relaxed));
    }

    pub(crate) fn of(object: &Native) -> Option<&BufferState> {
        match &object.extra {
            Extra::Buffer(state) => Some(state),
            _ => None,
        }
    }

    /// Where `[offset, offset + size)` of the current mapping starts in the
    /// range to ask Dawn for, and that range. Dawn, like the JavaScript API,
    /// wants mapped ranges that start on a multiple of 8 and span a multiple
    /// of 4; the copy calls of the C API have no such rule, so the range is
    /// widened within the mapping.
    pub(super) fn window(&self, offset: u64, size: u64, write: bool) -> Option<(u64, u64, usize)> {
        let map = *lock(&self.map);
        if map.state != BufferMapState::Mapped {
            return None;
        }
        if write && map.mode & map_mode::WRITE == 0 {
            return None;
        }
        let end = offset.checked_add(size)?;
        if offset < map.offset || end > map.offset + map.size {
            return None;
        }
        let start = offset & !7;
        let wide_end = (end.checked_add(3)? & !3)
            .min(map.offset + map.size)
            .max(end);
        Some((start, wide_end - start, (offset - start) as usize))
    }

    #[cfg(test)]
    pub(super) fn for_test(size: u64, map: MapInfo) -> Self {
        Self {
            size,
            map: Arc::new(Mutex::new(map)),
            access: RwLock::new(()),
            accounted: AtomicU64::new(0),
            gpu_memory: Arc::new(AtomicU64::new(0)),
        }
    }

    fn end_mapping(&self) {
        let mut map = lock(&self.map);
        map.state = BufferMapState::Unmapped;
        map.generation += 1;
    }
}

impl Host {
    pub(crate) fn manual_device_create_buffer(
        &self,
        _cx: &mut Cx<'_>,
        device: Handle,
        descriptor: &BufferDescriptor,
    ) -> Handle {
        let Some(parent) = self.object(device, Kind::Device) else {
            return 0;
        };
        let Some(state) = parent.device.clone() else {
            return 0;
        };
        if let Some(budget) = self.ctx.limits.max_gpu_memory_bytes {
            let used = self.gpu_memory.load(Ordering::Relaxed);
            if used.saturating_add(descriptor.size) > budget {
                state.inject(
                    ErrorType::OutOfMemory,
                    &format!(
                        "creating a {} byte buffer would exceed this guest's GPU memory budget",
                        descriptor.size
                    ),
                );
                return 0;
            }
        }
        let mut l = Lowering::new(self, Some(&parent));
        let native = match l.lower_ptr(descriptor) {
            Ok(native) => native,
            Err(error) => {
                l.fail("wgpuDeviceCreateBuffer", error);
                return 0;
            }
        };
        // SAFETY: a live device and a descriptor that outlives the call.
        let raw = unsafe { ffi::wgpuDeviceCreateBuffer(state.raw, native) };
        drop(l);
        if raw.is_null() {
            return 0;
        }
        charge(&self.gpu_memory, descriptor.size);
        let map = MapInfo {
            state: if descriptor.mapped_at_creation {
                BufferMapState::Mapped
            } else {
                BufferMapState::Unmapped
            },
            mode: if descriptor.mapped_at_creation {
                map_mode::WRITE
            } else {
                0
            },
            offset: 0,
            size: if descriptor.mapped_at_creation {
                descriptor.size
            } else {
                0
            },
            generation: 0,
        };
        self.insert(
            Arc::new(Native {
                kind: Kind::Buffer,
                raw: raw.cast(),
                device: Some(state),
                extra: Extra::Buffer(BufferState {
                    size: descriptor.size,
                    map: Arc::new(Mutex::new(map)),
                    access: RwLock::new(()),
                    accounted: AtomicU64::new(descriptor.size),
                    gpu_memory: Arc::clone(&self.gpu_memory),
                }),
            }),
            self.instance_of(device),
        )
    }

    /// Starts `wgpuBufferMapAsync`. The outcome arrives as an event record.
    pub(crate) fn buffer_map_async(
        &self,
        buffer: Handle,
        mode: u64,
        offset: u64,
        size: u64,
        future: u64,
    ) {
        let fail = |message: &str| {
            self.events.push(Record {
                future,
                kind: EVENT_BUFFER_MAP,
                status: MapAsyncStatus::Error.as_u32(),
                message: message.to_owned(),
                ..Record::default()
            });
        };
        let Some(object) = self.object(buffer, Kind::Buffer) else {
            fail("Invalid buffer.");
            return;
        };
        let Some(state) = BufferState::of(&object) else {
            fail("Invalid buffer.");
            return;
        };
        // WGPU_WHOLE_MAP_SIZE: everything from the offset on.
        let size = if size == u64::from(u32::MAX) {
            state.size.saturating_sub(offset)
        } else {
            size
        };
        // Held until Dawn has the request: an Unmap, a Destroy or another
        // request on another thread comes wholly before or wholly after it.
        // Otherwise an Unmap between the bookkeeping and Dawn's MapAsync
        // would end a mapping that does not exist yet, leaving the host
        // saying unmapped about a buffer Dawn then maps.
        let _requesting = exclusive(&state.access);
        let generation = {
            let mut map = lock(&state.map);
            if map.state == BufferMapState::Unmapped
                && offset
                    .checked_add(size)
                    .is_some_and(|end| end <= state.size)
            {
                map.generation += 1;
                *map = MapInfo {
                    state: BufferMapState::Pending,
                    mode,
                    offset,
                    size,
                    generation: map.generation,
                };
                map.generation
            } else {
                // A request Dawn will refuse leaves the current mapping (or
                // the one pending) as it is: its callback must not settle it.
                REFUSED_MAP
            }
        };
        let context = Map {
            pending: Pending::new(&self.events, &self.futures, future),
            map: Arc::clone(&state.map),
            generation,
        };
        let info = ffi::WGPUBufferMapCallbackInfo {
            mode: CallbackMode::AllowSpontaneous.as_u32(),
            callback: Some(callbacks::buffer_map),
            userdata1: callbacks::userdata(context),
            ..Default::default()
        };
        // SAFETY: a live buffer; the callback owns its context.
        let dawn_future = unsafe {
            ffi::wgpuBufferMapAsync(
                object.raw.cast(),
                mode,
                usize::try_from(offset).unwrap_or(usize::MAX),
                usize::try_from(size).unwrap_or(usize::MAX),
                info,
            )
        };
        self.futures.track(future, dawn_future);
    }

    /// `{ u64 offset; u64 size; u32 mode; u32 state }` of the current mapping.
    pub(crate) fn buffer_map_info(&self, buffer: Handle) -> Option<[u8; 24]> {
        let object = self.object(buffer, Kind::Buffer)?;
        let map = *lock(&BufferState::of(&object)?.map);
        let mut out = [0u8; 24];
        out[0..8].copy_from_slice(&map.offset.to_le_bytes());
        out[8..16].copy_from_slice(&map.size.to_le_bytes());
        out[16..20].copy_from_slice(&(map.mode as u32).to_le_bytes());
        out[20..24].copy_from_slice(&map.state.as_u32().to_le_bytes());
        Some(out)
    }

    /// Copies between the current mapping and guest memory: into the mapping
    /// for a write, out of it otherwise. The buffer's and its device's access
    /// locks are held throughout, so another guest thread's Unmap or Destroy,
    /// or the device's destruction, waits until the copy is done. Guest
    /// memory may change under the copy (other guest threads run), so
    /// Wasmer's copy, which allows that, is the one used.
    fn copy_mapped(
        &self,
        cx: &mut Cx<'_>,
        buffer: Handle,
        offset: u32,
        data: u32,
        size: u32,
        write: bool,
    ) -> Status {
        let Some(object) = self.object(buffer, Kind::Buffer) else {
            return Status::Error;
        };
        let Some(state) = BufferState::of(&object) else {
            return Status::Error;
        };
        let _device = object.device.as_ref().map(|device| shared(&device.access));
        // Copies out of a mapping can run side by side; one into it runs alone.
        let _buffer = if write {
            (None, Some(exclusive(&state.access)))
        } else {
            (Some(shared(&state.access)), None)
        };
        let Some((start, len, skip)) = state.window(u64::from(offset), u64::from(size), write)
        else {
            return Status::Error;
        };
        if size == 0 {
            return Status::Success;
        }
        // SAFETY: a mapped buffer (for writing, to write); the range lies
        // within the mapping, which the locks above keep.
        let mapped = unsafe {
            if write {
                ffi::wgpuBufferGetMappedRange(object.raw.cast(), start as usize, len as usize)
            } else {
                ffi::wgpuBufferGetConstMappedRange(object.raw.cast(), start as usize, len as usize)
                    .cast_mut()
            }
        };
        if mapped.is_null() {
            return Status::Error;
        }
        // SAFETY: `size` bytes at `skip` of the mapped range, which the locks
        // above keep mapped. While the host writes them, no other thread
        // reads or writes them (the buffer's lock is held exclusively); while
        // it reads them, none writes them. They are Dawn's memory, never the
        // guest's.
        let copied = unsafe {
            let mapped = mapped.cast::<u8>().add(skip);
            if write {
                let mapped = std::slice::from_raw_parts_mut(mapped, size as usize);
                cx.mem.read_into(data, mapped)
            } else {
                let mapped = std::slice::from_raw_parts(mapped.cast_const(), size as usize);
                cx.mem.write_bytes(data, mapped)
            }
        };
        match copied {
            Ok(()) => Status::Success,
            Err(_) => Status::Error,
        }
    }

    pub(crate) fn manual_buffer_read_mapped_range(
        &self,
        cx: &mut Cx<'_>,
        buffer: Handle,
        offset: u32,
        data: u32,
        size: u32,
    ) -> Status {
        self.copy_mapped(cx, buffer, offset, data, size, false)
    }

    pub(crate) fn manual_buffer_write_mapped_range(
        &self,
        cx: &mut Cx<'_>,
        buffer: Handle,
        offset: u32,
        data: u32,
        size: u32,
    ) -> Status {
        self.copy_mapped(cx, buffer, offset, data, size, true)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn manual_command_encoder_copy_buffer_to_buffer(
        &self,
        _cx: &mut Cx<'_>,
        command_encoder: Handle,
        source: Handle,
        source_offset: u64,
        destination: Handle,
        destination_offset: u64,
        size: u64,
    ) {
        const FUNCTION: &str = "wgpuCommandEncoderCopyBufferToBuffer";
        let Some(this) = self.object(command_encoder, Kind::CommandEncoder) else {
            return;
        };
        let mut l = Lowering::new(self, Some(&this));
        let source_object = match self.object(source, Kind::Buffer) {
            Some(object) => object,
            None => {
                return l.fail(
                    FUNCTION,
                    super::lower::Invalid::Handle {
                        member: "source",
                        handle: source,
                    },
                );
            }
        };
        let target = match l.object(Kind::Buffer, destination, "destination", false) {
            Ok(target) => target,
            Err(error) => return l.fail(FUNCTION, error),
        };
        // WGPU_WHOLE_SIZE: the rest of the source.
        let size = if size == u64::MAX {
            BufferState::of(&source_object)
                .map_or(0, |state| state.size.saturating_sub(source_offset))
        } else {
            size
        };
        // SAFETY: live encoder and buffers.
        unsafe {
            ffi::wgpuCommandEncoderCopyBufferToBuffer(
                this.raw.cast(),
                source_object.raw.cast(),
                source_offset,
                target.cast(),
                destination_offset,
                size,
            )
        };
    }

    pub(crate) fn manual_buffer_unmap(&self, _cx: &mut Cx<'_>, buffer: Handle) {
        let Some(object) = self.object(buffer, Kind::Buffer) else {
            return;
        };
        let state = BufferState::of(&object);
        // No copy is under way while Dawn frees the mapping.
        let _buffer = state.map(|state| exclusive(&state.access));
        if let Some(state) = state {
            state.end_mapping();
        }
        // SAFETY: a live buffer.
        unsafe { ffi::wgpuBufferUnmap(object.raw.cast()) };
    }

    pub(crate) fn manual_buffer_destroy(&self, _cx: &mut Cx<'_>, buffer: Handle) {
        let Some(object) = self.object(buffer, Kind::Buffer) else {
            return;
        };
        let state = BufferState::of(&object);
        let _buffer = state.map(|state| exclusive(&state.access));
        if let Some(state) = state {
            state.end_mapping();
            state.release_storage();
        }
        // SAFETY: a live buffer.
        unsafe { ffi::wgpuBufferDestroy(object.raw.cast()) };
    }
}
