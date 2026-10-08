//! `WGPUBuffer`.
//!
//! The host only ever copies between a mapping and guest memory
//! (`wgpuBufferReadMappedRange` / `wgpuBufferWriteMappedRange`). The pointers
//! `wgpuBufferGetMappedRange` returns are a guest-side concept: the shim
//! allocates them in the guest heap and flushes them through these two calls.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::{DeviceState, Host, Object, ObjectKind, Record, convert, lock};
use crate::abi::api::BufferApi;
use crate::abi::enums::{BufferMapState, MapAsyncStatus, Status, map_mode};
use crate::abi::structs::BufferDescriptor;
use crate::abi::{AbiEnum, Cx, EVENT_BUFFER_MAP, Handle};

pub(crate) fn label(text: &str) -> Option<&str> {
    (!text.is_empty()).then_some(text)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MapInfo {
    pub(crate) state: BufferMapState,
    /// `WGPUMapMode` bits of the current (or pending) mapping.
    pub(crate) mode: u64,
    pub(crate) offset: u64,
    pub(crate) size: u64,
    /// The pending map was cancelled by `Unmap` or `Destroy`.
    aborted: bool,
}

pub(crate) struct BufferState {
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) device: Arc<DeviceState>,
    pub(crate) size: u64,
    /// `WGPUBufferUsage` bits, as the guest gave them.
    pub(crate) usage: u64,
    pub(crate) map: Mutex<MapInfo>,
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
        let bytes = self.accounted.swap(0, Ordering::Relaxed);
        self.gpu_memory.fetch_sub(bytes, Ordering::Relaxed);
    }
}

impl BufferState {
    /// A copy window over `[offset, offset + size)` of the current mapping:
    /// the wgpu range to request, and where the wanted bytes start in it.
    ///
    /// wgpu, like the JavaScript API, wants mapped ranges that start on a
    /// multiple of 8 and span a multiple of 4. The copy calls of the C API
    /// have no such rule, so the range is widened within the mapping.
    fn window(&self, offset: u64, size: u64, write: bool) -> Option<(std::ops::Range<u64>, usize)> {
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
        let wide_end = (end.checked_add(3)? & !3).min(map.offset + map.size);
        Some((start..wide_end.max(end), (offset - start) as usize))
    }
}

impl Host {
    pub(crate) fn create_buffer(&self, device: Handle, descriptor: &BufferDescriptor) -> Handle {
        let Some(device_state) = self.device(device) else {
            return 0;
        };
        let Some(usage) = convert::buffer_usages(descriptor.usage) else {
            self.validation(
                &device_state,
                format!("{:#x} is not a valid WGPUBufferUsage", descriptor.usage),
            );
            return 0;
        };
        if let Some(budget) = self.ctx.limits.max_gpu_memory_bytes {
            let used = self.gpu_memory.load(Ordering::Relaxed);
            if used.saturating_add(descriptor.size) > budget {
                self.error(
                    &device_state,
                    crate::abi::enums::ErrorType::OutOfMemory,
                    format!(
                        "creating a {} byte buffer would exceed this guest's GPU memory budget",
                        descriptor.size
                    ),
                );
                return 0;
            }
        }
        let buffer = device_state.device.create_buffer(&wgpu::BufferDescriptor {
            label: label(&descriptor.label),
            size: descriptor.size,
            usage,
            mapped_at_creation: descriptor.mapped_at_creation,
        });
        self.gpu_memory
            .fetch_add(descriptor.size, Ordering::Relaxed);
        let map = if descriptor.mapped_at_creation {
            MapInfo {
                state: BufferMapState::Mapped,
                mode: map_mode::WRITE,
                offset: 0,
                size: descriptor.size,
                aborted: false,
            }
        } else {
            MapInfo {
                state: BufferMapState::Unmapped,
                mode: 0,
                offset: 0,
                size: 0,
                aborted: false,
            }
        };
        let state = BufferState {
            buffer,
            device: device_state,
            size: descriptor.size,
            usage: descriptor.usage,
            map: Mutex::new(map),
            accounted: AtomicU64::new(descriptor.size),
            gpu_memory: Arc::clone(&self.gpu_memory),
        };
        self.insert(Object::Buffer(Arc::new(state)), self.instance_of(device))
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
        let events = Arc::clone(&self.events);
        let fail = move |events: &super::Events, message: &str| {
            events.push(Record {
                future,
                kind: EVENT_BUFFER_MAP,
                status: MapAsyncStatus::Error.as_u32(),
                message: message.to_owned(),
                ..Record::default()
            });
        };
        let Some(state) = self.buffer(buffer) else {
            fail(&events, "Invalid buffer.");
            return;
        };
        let wgpu_mode = match mode {
            map_mode::READ => wgpu::MapMode::Read,
            map_mode::WRITE => wgpu::MapMode::Write,
            _ => {
                self.validation(&state.device, "A map mode must be exactly Read or Write.");
                fail(&events, "A map mode must be exactly Read or Write.");
                return;
            }
        };
        // WGPU_WHOLE_MAP_SIZE: everything from the offset on.
        let size = if size == u64::from(u32::MAX) {
            state.size.saturating_sub(offset)
        } else {
            size
        };
        {
            let mut map = lock(&state.map);
            if map.state != BufferMapState::Unmapped {
                drop(map);
                self.validation(
                    &state.device,
                    "The buffer is already mapped or has a map pending.",
                );
                fail(
                    &events,
                    "The buffer is already mapped or has a map pending.",
                );
                return;
            }
            let Some(end) = offset.checked_add(size).filter(|&end| end <= state.size) else {
                drop(map);
                self.validation(&state.device, "The mapped range is outside the buffer.");
                fail(&events, "The mapped range is outside the buffer.");
                return;
            };
            let _ = end;
            *map = MapInfo {
                state: BufferMapState::Pending,
                mode,
                offset,
                size,
                aborted: false,
            };
        }
        let callback_state = Arc::clone(&state);
        state
            .buffer
            .map_async(wgpu_mode, offset..offset + size, move |result| {
                let mut map = lock(&callback_state.map);
                let (status, message) = match result {
                    Ok(()) if map.state == BufferMapState::Pending => {
                        map.state = BufferMapState::Mapped;
                        (MapAsyncStatus::Success, String::new())
                    }
                    // Unmapped or destroyed in the meantime.
                    Ok(()) => (MapAsyncStatus::Aborted, "The map was cancelled.".to_owned()),
                    Err(error) => {
                        let aborted = map.aborted;
                        if map.state == BufferMapState::Pending {
                            map.state = BufferMapState::Unmapped;
                        }
                        if aborted {
                            (MapAsyncStatus::Aborted, "The map was cancelled.".to_owned())
                        } else {
                            (MapAsyncStatus::Error, error.to_string())
                        }
                    }
                };
                drop(map);
                events.push(Record {
                    future,
                    kind: EVENT_BUFFER_MAP,
                    status: status.as_u32(),
                    message,
                    ..Record::default()
                });
            });
    }

    /// `{ u64 offset; u64 size; u32 mode; u32 state }` of the current mapping.
    pub(crate) fn buffer_map_info(&self, buffer: Handle) -> Option<[u8; 24]> {
        let state = self.buffer(buffer)?;
        let map = *lock(&state.map);
        let mut out = [0u8; 24];
        out[0..8].copy_from_slice(&map.offset.to_le_bytes());
        out[8..16].copy_from_slice(&map.size.to_le_bytes());
        out[16..20].copy_from_slice(&(map.mode as u32).to_le_bytes());
        out[20..24].copy_from_slice(&map.state.as_u32().to_le_bytes());
        Some(out)
    }

    /// Marks the buffer unmapped, noting a cancelled pending map.
    fn end_mapping(state: &BufferState) {
        let mut map = lock(&state.map);
        if map.state == BufferMapState::Pending {
            map.aborted = true;
        }
        map.state = BufferMapState::Unmapped;
    }
}

impl BufferApi for Host {
    fn buffer_read_mapped_range(
        &self,
        cx: &mut Cx<'_>,
        buffer: Handle,
        offset: u32,
        data: u32,
        size: u32,
    ) -> Status {
        let Some(state) = self.buffer(buffer) else {
            return Status::Error;
        };
        let Some((range, skip)) = state.window(u64::from(offset), u64::from(size), false) else {
            return Status::Error;
        };
        if size == 0 {
            return Status::Success;
        }
        // wgpu panics on a range it will not map; the window above is
        // within the mapping, so this is a backstop, not the expected path.
        let Ok(view) = super::catch_expected_panic(|| state.buffer.get_mapped_range(range)) else {
            return Status::Error;
        };
        match cx.mem.write_bytes(data, &view[skip..skip + size as usize]) {
            Ok(()) => Status::Success,
            Err(_) => Status::Error,
        }
    }

    fn buffer_write_mapped_range(
        &self,
        cx: &mut Cx<'_>,
        buffer: Handle,
        offset: u32,
        data: u32,
        size: u32,
    ) -> Status {
        let Some(state) = self.buffer(buffer) else {
            return Status::Error;
        };
        let Some((range, skip)) = state.window(u64::from(offset), u64::from(size), true) else {
            return Status::Error;
        };
        if size == 0 {
            return Status::Success;
        }
        let Ok(bytes) = cx.mem.read_vec(data, u64::from(size)) else {
            return Status::Error;
        };
        let Ok(mut view) = super::catch_expected_panic(|| state.buffer.get_mapped_range_mut(range))
        else {
            return Status::Error;
        };
        view.slice(skip..skip + size as usize)
            .copy_from_slice(&bytes);
        Status::Success
    }

    fn buffer_set_label(&self, _cx: &mut Cx<'_>, buffer: Handle, _label: &str) {
        let _ = self.buffer(buffer);
    }

    fn buffer_get_usage(&self, _cx: &mut Cx<'_>, buffer: Handle) -> u64 {
        self.buffer(buffer).map_or(0, |state| state.usage)
    }

    fn buffer_get_size(&self, _cx: &mut Cx<'_>, buffer: Handle) -> u64 {
        self.buffer(buffer).map_or(0, |state| state.size)
    }

    fn buffer_get_map_state(&self, _cx: &mut Cx<'_>, buffer: Handle) -> BufferMapState {
        self.buffer(buffer)
            .map_or(BufferMapState::Unmapped, |state| lock(&state.map).state)
    }

    fn buffer_unmap(&self, _cx: &mut Cx<'_>, buffer: Handle) {
        if let Some(state) = self.buffer(buffer) {
            Self::end_mapping(&state);
            state.buffer.unmap();
        }
    }

    fn buffer_destroy(&self, _cx: &mut Cx<'_>, buffer: Handle) {
        if let Some(state) = self.buffer(buffer) {
            Self::end_mapping(&state);
            state.buffer.destroy();
            state.release_storage();
        }
    }

    fn buffer_add_ref(&self, _cx: &mut Cx<'_>, buffer: Handle) {
        self.add_ref(buffer, ObjectKind::Buffer);
    }
}
