//! The handle table: the guest's opaque `WGPU*` values, and what they refer to.
//!
//! This is the safety boundary of the native hosts. The guest only ever holds
//! 32-bit handles; each is checked for liveness (a generation tag catches
//! stale ones) and for type before any GPU object is touched, so a confused
//! or hostile guest can at worst get a validation error.

use crate::abi::Handle;

/// What a host keeps behind a handle. Cloning is a reference-count bump.
pub(crate) trait TableObject: Clone {
    type Kind: Copy + PartialEq + std::fmt::Debug;
    fn kind(&self) -> Self::Kind;
    /// An instance descends from itself.
    fn is_instance(&self) -> bool;
}

struct Entry<O> {
    object: O,
    /// References held by the guest. Internal users (a bind group keeping its
    /// buffer alive, say) hold the `Arc` instead.
    refs: u32,
    /// The instance this object descends from: events are delivered per
    /// instance.
    instance: Handle,
}

struct Slot<O> {
    generation: u8,
    entry: Option<Entry<O>>,
}

const INDEX_BITS: u32 = 24;
const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;

/// The handles of one guest process, and what each refers to.
pub(crate) struct Table<O> {
    slots: Vec<Slot<O>>,
    free: Vec<u32>,
    live: usize,
    max_live: Option<usize>,
}

impl<O: TableObject> Table<O> {
    pub(crate) fn new(max_live: Option<usize>) -> Self {
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

    fn entry(&self, handle: Handle) -> Option<&Entry<O>> {
        let index = (handle & INDEX_MASK).checked_sub(1)? as usize;
        let slot = self.slots.get(index)?;
        if u32::from(slot.generation) != handle >> INDEX_BITS {
            return None;
        }
        slot.entry.as_ref()
    }

    fn entry_mut(&mut self, handle: Handle) -> Option<&mut Entry<O>> {
        let index = (handle & INDEX_MASK).checked_sub(1)? as usize;
        let slot = self.slots.get_mut(index)?;
        if u32::from(slot.generation) != handle >> INDEX_BITS {
            return None;
        }
        slot.entry.as_mut()
    }

    pub(crate) fn insert(&mut self, object: O, instance: Handle) -> Handle {
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
            && entry.object.is_instance()
        {
            entry.instance = handle;
        }
        handle
    }

    pub(crate) fn get(&self, handle: Handle) -> Option<O> {
        self.entry(handle).map(|entry| entry.object.clone())
    }

    pub(crate) fn instance_of(&self, handle: Handle) -> Handle {
        self.entry(handle).map_or(0, |entry| entry.instance)
    }

    pub(crate) fn add_ref(&mut self, handle: Handle, kind: O::Kind) -> bool {
        match self.entry_mut(handle) {
            Some(entry) if entry.object.kind() == kind => {
                entry.refs = entry.refs.saturating_add(1);
                true
            }
            _ => false,
        }
    }

    /// `Ok(Some(object))` when the last guest reference went away.
    pub(crate) fn release(&mut self, handle: Handle, kind: O::Kind) -> Result<Option<O>, ()> {
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

    pub(crate) fn live(&self) -> usize {
        self.live
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Kind {
        Thing,
        Other,
    }

    #[derive(Clone)]
    struct Thing;

    impl TableObject for Thing {
        type Kind = Kind;
        fn kind(&self) -> Kind {
            Kind::Thing
        }
        fn is_instance(&self) -> bool {
            false
        }
    }

    fn external() -> Thing {
        Thing
    }

    #[test]
    fn handles_are_nonzero_typed_and_counted() {
        let mut table = Table::<Thing>::new(None);
        let handle = table.insert(external(), 9);
        assert_ne!(handle, 0);
        assert_eq!(table.instance_of(handle), 9);
        assert!(table.add_ref(handle, Kind::Thing));
        assert!(!table.add_ref(handle, Kind::Other), "wrong type");
        assert!(matches!(table.release(handle, Kind::Thing), Ok(None)));
        assert!(matches!(table.release(handle, Kind::Thing), Ok(Some(_))));
        assert!(table.get(handle).is_none());
        assert_eq!(table.live(), 0);
    }

    #[test]
    fn a_stale_handle_does_not_alias_the_slot_s_next_object() {
        let mut table = Table::<Thing>::new(None);
        let first = table.insert(external(), 0);
        table.release(first, Kind::Thing).unwrap();
        let second = table.insert(external(), 0);
        assert_ne!(first, second, "the slot is reused under a new generation");
        assert!(table.get(first).is_none());
        assert!(table.get(second).is_some());
        assert!(table.release(first, Kind::Thing).is_err());
    }

    #[test]
    fn the_object_limit_yields_null_handles() {
        let mut table = Table::<Thing>::new(Some(1));
        let first = table.insert(external(), 0);
        assert_ne!(first, 0);
        assert_eq!(table.insert(external(), 0), 0);
        table.release(first, Kind::Thing).unwrap();
        assert_ne!(table.insert(external(), 0), 0);
    }

    #[test]
    fn garbage_handles_are_rejected() {
        let mut table = Table::<Thing>::new(None);
        table.insert(external(), 0);
        for handle in [0, 2, 0xFFFF_FFFF, 0x0100_0001] {
            assert!(table.get(handle).is_none(), "{handle:#x}");
            assert!(!table.add_ref(handle, Kind::Thing));
        }
    }
}
