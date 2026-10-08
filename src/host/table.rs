//! The handle table: the guest's opaque `WGPU*` values, and what they refer to.
//!
//! This is the safety boundary of the native host. The guest only ever holds
//! 32-bit handles; each is checked for liveness (a generation tag catches
//! stale ones) and for type before any wgpu object is touched, so a confused
//! or hostile guest can at worst get a validation error.

use std::sync::Arc;

use super::bind::{BindGroupState, PipelineLayoutState};
use super::encoder::{CommandBufferState, EncoderState};
use super::pass::{BundleEncoderState, ComputePassState, RenderBundleState, RenderPassState};
use super::pipeline::{ComputePipelineState, RenderPipelineState};
use super::shader::ShaderState;
use super::texture::{QuerySetState, TextureViewState};
use super::{
    AdapterState, BufferState, DeviceState, Host, InstanceState, SurfaceState, TextureState,
};
use crate::abi::Handle;

macro_rules! objects {
    ($( $variant:ident($state:ty) => $getter:ident ),* $(,)?) => {
        /// What a handle refers to. Cloning is a reference-count bump.
        #[derive(Clone)]
        #[allow(dead_code)] // `ExternalTexture`, see below
        pub(crate) enum Object {
            $( $variant(Arc<$state>), )*
        }

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub(crate) enum ObjectKind {
            $( $variant, )*
        }

        impl Object {
            pub(crate) fn kind(&self) -> ObjectKind {
                match self {
                    $( Object::$variant(_) => ObjectKind::$variant, )*
                }
            }
        }

        impl Host {
            $(
                /// The object behind `handle`, if it is live and of this type.
                pub(crate) fn $getter(&self, handle: Handle) -> Option<Arc<$state>> {
                    match self.object(handle) {
                        Some(Object::$variant(state)) => Some(state),
                        other => {
                            super::log(|| format!(
                                "expected a {} handle, got {:#x} ({})",
                                stringify!($variant),
                                handle,
                                match &other {
                                    Some(object) => format!("a {:?}", object.kind()),
                                    None if handle == 0 => "null".to_owned(),
                                    None => "not a live handle".to_owned(),
                                },
                            ));
                            None
                        }
                    }
                }
            )*
        }
    };
}

objects! {
    Instance(InstanceState) => instance,
    Adapter(AdapterState) => adapter,
    Device(DeviceState) => device,
    // A queue is its device's queue: same state, separate handle.
    Queue(DeviceState) => queue,
    Buffer(BufferState) => buffer,
    Texture(TextureState) => texture,
    TextureView(TextureViewState) => texture_view,
    Sampler(wgpu::Sampler) => sampler,
    BindGroupLayout(wgpu::BindGroupLayout) => bind_group_layout,
    BindGroup(BindGroupState) => bind_group,
    PipelineLayout(PipelineLayoutState) => pipeline_layout,
    ShaderModule(ShaderState) => shader_module,
    ComputePipeline(ComputePipelineState) => compute_pipeline,
    RenderPipeline(RenderPipelineState) => render_pipeline,
    CommandEncoder(EncoderState) => command_encoder,
    CommandBuffer(CommandBufferState) => command_buffer,
    ComputePassEncoder(ComputePassState) => compute_pass,
    RenderPassEncoder(RenderPassState) => render_pass,
    RenderBundleEncoder(BundleEncoderState) => render_bundle_encoder,
    RenderBundle(RenderBundleState) => render_bundle,
    QuerySet(QuerySetState) => query_set,
    Surface(SurfaceState) => surface,
    // webgpu.h has the type but no way to create one, so nothing ever
    // constructs this variant; it exists for the type's AddRef/Release.
    ExternalTexture(()) => external_texture,
}

struct Entry {
    object: Object,
    /// References held by the guest. Internal users (a bind group keeping its
    /// buffer alive, say) hold the `Arc` instead.
    refs: u32,
    /// The instance this object descends from: events are delivered per
    /// instance.
    instance: Handle,
}

struct Slot {
    generation: u8,
    entry: Option<Entry>,
}

const INDEX_BITS: u32 = 24;
const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;

pub(super) struct Table {
    slots: Vec<Slot>,
    free: Vec<u32>,
    live: usize,
    max_live: Option<usize>,
}

impl Table {
    pub(super) fn new(max_live: Option<usize>) -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
            max_live,
        }
    }

    fn handle(index: u32, generation: u8) -> Handle {
        (u32::from(generation) << INDEX_BITS) | (index + 1)
    }

    fn entry(&self, handle: Handle) -> Option<&Entry> {
        let index = (handle & INDEX_MASK).checked_sub(1)? as usize;
        let slot = self.slots.get(index)?;
        if u32::from(slot.generation) != handle >> INDEX_BITS {
            return None;
        }
        slot.entry.as_ref()
    }

    fn entry_mut(&mut self, handle: Handle) -> Option<&mut Entry> {
        let index = (handle & INDEX_MASK).checked_sub(1)? as usize;
        let slot = self.slots.get_mut(index)?;
        if u32::from(slot.generation) != handle >> INDEX_BITS {
            return None;
        }
        slot.entry.as_mut()
    }

    pub(super) fn insert(&mut self, object: Object, instance: Handle) -> Handle {
        if self.max_live.is_some_and(|max| self.live >= max) {
            return 0;
        }
        let entry = Entry {
            object,
            refs: 1,
            instance,
        };
        let index = match self.free.pop() {
            Some(index) => index,
            None => {
                if self.slots.len() as u32 >= INDEX_MASK {
                    return 0;
                }
                self.slots.push(Slot {
                    generation: 0,
                    entry: None,
                });
                self.slots.len() as u32 - 1
            }
        };
        let slot = &mut self.slots[index as usize];
        slot.entry = Some(entry);
        self.live += 1;
        let handle = Self::handle(index, slot.generation);
        // An instance descends from itself.
        if instance == 0
            && let Some(entry) = slot.entry.as_mut()
            && matches!(entry.object, Object::Instance(_))
        {
            entry.instance = handle;
        }
        handle
    }

    pub(super) fn get(&self, handle: Handle) -> Option<Object> {
        self.entry(handle).map(|entry| entry.object.clone())
    }

    pub(super) fn instance_of(&self, handle: Handle) -> Handle {
        self.entry(handle).map_or(0, |entry| entry.instance)
    }

    pub(super) fn add_ref(&mut self, handle: Handle, kind: ObjectKind) -> bool {
        match self.entry_mut(handle) {
            Some(entry) if entry.object.kind() == kind => {
                entry.refs = entry.refs.saturating_add(1);
                true
            }
            _ => false,
        }
    }

    /// `Ok(Some(object))` when the last guest reference went away.
    pub(super) fn release(
        &mut self,
        handle: Handle,
        kind: ObjectKind,
    ) -> Result<Option<Object>, ()> {
        let entry = self.entry_mut(handle).ok_or(())?;
        if entry.object.kind() != kind {
            return Err(());
        }
        entry.refs -= 1;
        if entry.refs > 0 {
            return Ok(None);
        }
        let index = (handle & INDEX_MASK) - 1;
        let slot = &mut self.slots[index as usize];
        let entry = slot.entry.take().expect("entry was just found");
        // The next occupant of this slot gets a different handle value, so a
        // stale handle is detected instead of aliasing the new object.
        slot.generation = slot.generation.wrapping_add(1);
        self.free.push(index);
        self.live -= 1;
        Ok(Some(entry.object))
    }

    pub(super) fn live(&self) -> usize {
        self.live
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn external() -> Object {
        Object::ExternalTexture(Arc::new(()))
    }

    #[test]
    fn handles_are_nonzero_typed_and_counted() {
        let mut table = Table::new(None);
        let handle = table.insert(external(), 9);
        assert_ne!(handle, 0);
        assert_eq!(table.instance_of(handle), 9);
        assert!(table.add_ref(handle, ObjectKind::ExternalTexture));
        assert!(!table.add_ref(handle, ObjectKind::Buffer), "wrong type");
        assert!(matches!(
            table.release(handle, ObjectKind::ExternalTexture),
            Ok(None)
        ));
        assert!(matches!(
            table.release(handle, ObjectKind::ExternalTexture),
            Ok(Some(_))
        ));
        assert!(table.get(handle).is_none());
        assert_eq!(table.live(), 0);
    }

    #[test]
    fn a_stale_handle_does_not_alias_the_slot_s_next_object() {
        let mut table = Table::new(None);
        let first = table.insert(external(), 0);
        table.release(first, ObjectKind::ExternalTexture).unwrap();
        let second = table.insert(external(), 0);
        assert_ne!(first, second, "the slot is reused under a new generation");
        assert!(table.get(first).is_none());
        assert!(table.get(second).is_some());
        assert!(table.release(first, ObjectKind::ExternalTexture).is_err());
    }

    #[test]
    fn the_object_limit_yields_null_handles() {
        let mut table = Table::new(Some(1));
        let first = table.insert(external(), 0);
        assert_ne!(first, 0);
        assert_eq!(table.insert(external(), 0), 0);
        table.release(first, ObjectKind::ExternalTexture).unwrap();
        assert_ne!(table.insert(external(), 0), 0);
    }

    #[test]
    fn garbage_handles_are_rejected() {
        let mut table = Table::new(None);
        table.insert(external(), 0);
        for handle in [0, 2, 0xFFFF_FFFF, 0x0100_0001] {
            assert!(table.get(handle).is_none(), "{handle:#x}");
            assert!(!table.add_ref(handle, ObjectKind::ExternalTexture));
        }
    }
}
