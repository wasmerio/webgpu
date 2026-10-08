//! The guest ABI: how `webgpu.h` looks from inside a wasm32 module.
//!
//! Everything layout-shaped here is generated from the pinned `webgpu.json`
//! (`codegen/gen.py`), and `guest/tests/abi_check.gen.c` makes the C compiler
//! confirm the generator's model. This module holds the hand-written pieces
//! the generated code builds on: bounds-checked guest memory access and the
//! small traits the readers implement.

// The generated modules keep the layout their generator gives them.
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::all, non_camel_case_types, clippy::upper_case_acronyms)]
#[rustfmt::skip]
#[path = "enums.gen.rs"]
pub mod enums;
#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::all)]
#[rustfmt::skip]
#[path = "structs.gen.rs"]
pub mod structs;

#[cfg(not(target_arch = "wasm32"))]
#[allow(clippy::all, non_snake_case)]
#[rustfmt::skip]
#[path = "api.gen.rs"]
pub(crate) mod api;

#[rustfmt::skip]
#[path = "imports.gen.rs"]
mod imports;

pub use imports::*;

/// Core wasm value type of an import parameter or result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ValType {
    I32,
    I64,
    F32,
    F64,
}

/// Name and core wasm signature of one import.
#[derive(Clone, Copy, Debug)]
pub struct ImportSig {
    pub name: &'static str,
    pub params: &'static [ValType],
    pub results: &'static [ValType],
    /// The import may block, so a browser host has to suspend the guest for it.
    pub suspending: bool,
}

/// Looks up an import of the WebGPU module by name.
pub fn import_sig(name: &str) -> Option<&'static ImportSig> {
    IMPORTS.iter().find(|sig| sig.name == name)
}

#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::marker::PhantomData;

    use wasmer::MemoryView;

    /// A guest object handle. Zero is the null handle.
    pub type Handle = u32;

    /// Largest array a descriptor may carry. Counts are guest-controlled, so a
    /// ceiling keeps one call from sizing a multi-gigabyte host allocation.
    pub const MAX_ARRAY_LENGTH: u32 = 1 << 20;
    /// Largest string the host copies out of guest memory (shader sources
    /// included).
    pub const MAX_STRING_BYTES: u32 = 64 << 20;
    /// `WGPU_STRLEN`: the string view is NUL-terminated.
    pub const STRLEN: u32 = u32::MAX;

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub enum AbiError {
        /// The guest passed an address range outside its linear memory.
        OutOfBounds {
            ptr: u32,
            len: u64,
        },
        InvalidEnum {
            ty: &'static str,
            value: u32,
        },
        UnknownSType(u32),
        ChainTooLong,
        ArrayTooLong(u32),
        StringTooLong,
        /// A pointer that webgpu.h does not mark `WGPU_NULLABLE` was null.
        NullPointer(&'static str),
        /// The guest was stopped while the call waited: the call traps, so
        /// that the guest's thread ends instead of coming back for more.
        Stopped,
    }

    impl std::fmt::Display for AbiError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::OutOfBounds { ptr, len } => {
                    write!(f, "{len} bytes at {ptr:#x} are outside guest memory")
                }
                Self::InvalidEnum { ty, value } => write!(f, "{value:#x} is not a valid {ty}"),
                Self::UnknownSType(s_type) => {
                    write!(f, "sType {s_type:#x} is not allowed in this chain")
                }
                Self::ChainTooLong => f.write_str("nextInChain list is too long"),
                Self::ArrayTooLong(count) => write!(f, "array of {count} elements is too long"),
                Self::StringTooLong => f.write_str("string is too long"),
                Self::NullPointer(name) => write!(f, "`{name}` must not be null"),
                Self::Stopped => f.write_str("the program was stopped"),
            }
        }
    }

    impl std::error::Error for AbiError {}

    pub type AbiResult<T> = Result<T, AbiError>;

    /// An enum of the C API with its numeric values.
    pub trait AbiEnum: Sized + Copy {
        const NAME: &'static str;
        fn from_u32(value: u32) -> Option<Self>;
        fn as_u32(self) -> u32;
    }

    /// A struct that can be decoded from its wasm32 layout.
    pub trait GuestRead: Sized {
        const SIZE: u32;
        const ALIGN: u32;
        fn read(mem: &GuestMemory<'_>, ptr: u32) -> AbiResult<Self>;
    }

    /// A struct the host stores back into guest memory.
    pub trait GuestWrite {
        fn write(&self, mem: &GuestMemory<'_>, ptr: u32) -> AbiResult<()>;
    }

    /// An out-parameter: the guest address of a `T` the host fills in.
    pub struct OutPtr<T> {
        pub ptr: u32,
        _marker: PhantomData<T>,
    }

    impl<T> Clone for OutPtr<T> {
        fn clone(&self) -> Self {
            *self
        }
    }
    impl<T> Copy for OutPtr<T> {}

    impl<T> OutPtr<T> {
        pub fn new(ptr: u32) -> Self {
            Self {
                ptr,
                _marker: PhantomData,
            }
        }

        pub fn is_null(&self) -> bool {
            self.ptr == 0
        }
    }

    impl<T: GuestRead> OutPtr<T> {
        /// Decodes what the guest put there (its chain, for example).
        pub fn read(&self, mem: &GuestMemory<'_>) -> AbiResult<T> {
            T::read(mem, self.ptr)
        }
    }

    impl<T: GuestWrite> OutPtr<T> {
        pub fn write(&self, mem: &GuestMemory<'_>, value: &T) -> AbiResult<()> {
            value.write(mem, self.ptr)
        }
    }

    /// Per-call context handed to every [`api`](super::api) method.
    pub struct Cx<'a> {
        pub mem: GuestMemory<'a>,
        /// Whether the calling guest has been stopped meanwhile. A call that
        /// waits asks between slices, and gives up with
        /// [`AbiError::Stopped`].
        pub stopped: &'a dyn Fn() -> bool,
    }

    /// Bounds-checked access to the calling instance's linear memory.
    ///
    /// Guest memory is shared with every guest thread, so nothing here hands
    /// out references into it: all reads and writes are copies.
    pub struct GuestMemory<'a> {
        view: MemoryView<'a>,
        /// The first access that fell outside the memory, if any.
        fault: std::cell::Cell<Option<(u32, u64)>>,
    }

    impl<'a> GuestMemory<'a> {
        pub fn new(view: MemoryView<'a>) -> Self {
            Self {
                view,
                fault: std::cell::Cell::new(None),
            }
        }

        pub fn size(&self) -> u64 {
            self.view.data_size()
        }

        /// An access outside the memory. It is remembered as well as
        /// returned: whatever the caller makes of the error, the import ends
        /// in a trap (see [`GuestMemory::take_fault`]).
        fn fault(&self, ptr: u32, len: u64) -> AbiError {
            if self.fault.get().is_none() {
                self.fault.set(Some((ptr, len)));
            }
            AbiError::OutOfBounds { ptr, len }
        }

        /// The first out-of-bounds access made through this view, once. A
        /// wild guest pointer is a fault of the guest, never a status code:
        /// the import that saw one traps even if its implementation went on.
        pub fn take_fault(&self) -> Option<AbiError> {
            self.fault
                .take()
                .map(|(ptr, len)| AbiError::OutOfBounds { ptr, len })
        }

        pub fn read_into(&self, ptr: u32, out: &mut [u8]) -> AbiResult<()> {
            self.view
                .read(u64::from(ptr), out)
                .map_err(|_| self.fault(ptr, out.len() as u64))
        }

        pub fn read_bytes<const N: usize>(&self, ptr: u32) -> AbiResult<[u8; N]> {
            let mut out = [0u8; N];
            self.read_into(ptr, &mut out)?;
            Ok(out)
        }

        pub fn read_vec(&self, ptr: u32, len: u64) -> AbiResult<Vec<u8>> {
            if len == 0 {
                return Ok(Vec::new());
            }
            if u64::from(ptr) + len > self.size() {
                return Err(self.fault(ptr, len));
            }
            let mut out = vec![0u8; len as usize];
            self.read_into(ptr, &mut out)?;
            Ok(out)
        }

        pub fn write_bytes(&self, ptr: u32, data: &[u8]) -> AbiResult<()> {
            self.view
                .write(u64::from(ptr), data)
                .map_err(|_| self.fault(ptr, data.len() as u64))
        }

        pub fn read_u32(&self, ptr: u32) -> AbiResult<u32> {
            Ok(u32::from_le_bytes(self.read_bytes::<4>(ptr)?))
        }

        pub fn read_u64(&self, ptr: u32) -> AbiResult<u64> {
            Ok(u64::from_le_bytes(self.read_bytes::<8>(ptr)?))
        }

        pub fn write_u32(&self, ptr: u32, value: u32) -> AbiResult<()> {
            self.write_bytes(ptr, &value.to_le_bytes())
        }

        pub fn write_u64(&self, ptr: u32, value: u64) -> AbiResult<()> {
            self.write_bytes(ptr, &value.to_le_bytes())
        }

        /// Bytes of a `WGPUStringView { data, length }` with non-null `data`.
        fn string_bytes(&self, data: u32, length: u32) -> AbiResult<Vec<u8>> {
            if length != STRLEN {
                if length > MAX_STRING_BYTES {
                    return Err(AbiError::StringTooLong);
                }
                return self.read_vec(data, u64::from(length));
            }
            // NUL-terminated: scan in chunks, never past the end of memory.
            let mut out = Vec::new();
            let mut at = u64::from(data);
            let end = self.size();
            let mut chunk = [0u8; 256];
            loop {
                if at >= end {
                    return Err(self.fault(data, out.len() as u64 + 1));
                }
                let n = chunk.len().min((end - at) as usize);
                self.view
                    .read(at, &mut chunk[..n])
                    .map_err(|_| self.fault(data, out.len() as u64 + n as u64))?;
                if let Some(nul) = chunk[..n].iter().position(|&b| b == 0) {
                    out.extend_from_slice(&chunk[..nul]);
                    return Ok(out);
                }
                out.extend_from_slice(&chunk[..n]);
                if out.len() as u64 > u64::from(MAX_STRING_BYTES) {
                    return Err(AbiError::StringTooLong);
                }
                at += n as u64;
            }
        }

        /// A non-null input string: null `data` reads as the empty string.
        pub fn read_string(&self, data: u32, length: u32) -> AbiResult<String> {
            if data == 0 {
                return Ok(String::new());
            }
            Ok(String::from_utf8_lossy(&self.string_bytes(data, length)?).into_owned())
        }

        /// A nullable input string: `{NULL, WGPU_STRLEN}` is the null value and
        /// `{NULL, 0}` the empty string.
        pub fn read_nullable_string(&self, data: u32, length: u32) -> AbiResult<Option<String>> {
            if data == 0 {
                return Ok(if length == STRLEN {
                    None
                } else {
                    Some(String::new())
                });
            }
            Ok(Some(
                String::from_utf8_lossy(&self.string_bytes(data, length)?).into_owned(),
            ))
        }

        /// A `WGPUStringView` passed by value arrives as a pointer to it.
        pub fn read_string_view(&self, ptr: u32) -> AbiResult<String> {
            let b = self.read_bytes::<8>(ptr)?;
            self.read_string(rd_u32(&b, 0), rd_u32(&b, 4))
        }

        pub fn read_nullable_string_view(&self, ptr: u32) -> AbiResult<Option<String>> {
            let b = self.read_bytes::<8>(ptr)?;
            self.read_nullable_string(rd_u32(&b, 0), rd_u32(&b, 4))
        }

        pub fn read_opt<T: GuestRead>(&self, ptr: u32) -> AbiResult<Option<Box<T>>> {
            if ptr == 0 {
                return Ok(None);
            }
            Ok(Some(Box::new(T::read(self, ptr)?)))
        }

        pub fn read_required<T: GuestRead>(&self, ptr: u32, name: &'static str) -> AbiResult<T> {
            if ptr == 0 {
                return Err(AbiError::NullPointer(name));
            }
            T::read(self, ptr)
        }

        fn check_array(&self, ptr: u32, count: u32, element: u32) -> AbiResult<()> {
            if count > MAX_ARRAY_LENGTH {
                return Err(AbiError::ArrayTooLong(count));
            }
            let len = u64::from(count) * u64::from(element);
            if count != 0 && u64::from(ptr) + len > self.size() {
                return Err(self.fault(ptr, len));
            }
            Ok(())
        }

        pub fn read_array<T: GuestRead>(&self, ptr: u32, count: u32) -> AbiResult<Vec<T>> {
            self.check_array(ptr, count, T::SIZE)?;
            (0..count)
                .map(|i| T::read(self, ptr + i * T::SIZE))
                .collect()
        }

        pub fn read_u32_array(&self, ptr: u32, count: u32) -> AbiResult<Vec<u32>> {
            self.check_array(ptr, count, 4)?;
            let bytes = self.read_vec(ptr, u64::from(count) * 4)?;
            Ok(bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| u32::from_le_bytes(*c))
                .collect())
        }

        pub fn read_u64_array(&self, ptr: u32, count: u32) -> AbiResult<Vec<u64>> {
            self.check_array(ptr, count, 8)?;
            let bytes = self.read_vec(ptr, u64::from(count) * 8)?;
            Ok(bytes
                .as_chunks::<8>()
                .0
                .iter()
                .map(|c| u64::from_le_bytes(*c))
                .collect())
        }

        pub fn read_enum_array<E: AbiEnum>(&self, ptr: u32, count: u32) -> AbiResult<Vec<E>> {
            self.read_u32_array(ptr, count)?
                .into_iter()
                .map(|value| E::from_u32(value).ok_or(AbiError::InvalidEnum { ty: E::NAME, value }))
                .collect()
        }
    }

    pub(crate) fn rd_u16(b: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([b[at], b[at + 1]])
    }

    pub(crate) fn rd_u32(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }

    pub(crate) fn rd_u64(b: &[u8], at: usize) -> u64 {
        u64::from_le_bytes(b[at..at + 8].try_into().expect("8 bytes"))
    }

    // Used by generated writers only when a host-written struct has a u16.
    #[allow(dead_code)]
    pub(crate) fn wr_u16(b: &mut [u8], at: usize, value: u16) {
        b[at..at + 2].copy_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn wr_u32(b: &mut [u8], at: usize, value: u32) {
        b[at..at + 4].copy_from_slice(&value.to_le_bytes());
    }

    pub(crate) fn wr_u64(b: &mut [u8], at: usize, value: u64) {
        b[at..at + 8].copy_from_slice(&value.to_le_bytes());
    }
}
