//! Turning decoded guest values into native ones, for one call.
//!
//! A native `webgpu.h` struct holds pointers: to strings, arrays, nested
//! structs and chained extensions. [`Lowering`] owns everything those point
//! to until the call that uses them returns, and resolves the guest's object
//! handles to the GPU objects behind them. The generated `Lower`
//! implementations (`lower.gen.rs`) and forwards (`forward.gen.rs`) are
//! written against it.

use std::any::Any;
use std::ffi::c_void;

use super::ffi::{self, Kind, WGPUStringView};
use super::{Host, Object, log};
use crate::abi::enums::ErrorType;
use crate::abi::{AbiEnum, Handle};

/// Why a guest value has no native form. Reported as a validation error on
/// the device the call is about (or logged, for calls with no device).
#[derive(Debug)]
pub(crate) enum Invalid {
    /// `member` names an object of the wrong type, or no live object.
    Handle {
        member: &'static str,
        handle: Handle,
    },
    /// A required object is missing.
    Missing(&'static str),
    /// Something no guest may pass to the GPU library.
    Unsupported(&'static str),
}

impl std::fmt::Display for Invalid {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Invalid::Handle { member, handle } => {
                write!(f, "{member} is not a valid object ({handle:#x})")
            }
            Invalid::Missing(member) => write!(f, "{member} is required"),
            Invalid::Unsupported(what) => write!(f, "{what} are not supported"),
        }
    }
}

/// A decoded guest struct with a native form.
pub(crate) trait Lower {
    type Native: Copy;
    fn lower(&self, l: &mut Lowering<'_>) -> Result<Self::Native, Invalid>;
}

/// A webgpu.h enum value the guest left `undefined` (1), as `default`; any
/// other value as it is.
pub(crate) fn undefined_as(value: u32, default: u32) -> u32 {
    if value == 1 { default } else { value }
}

/// `size_t` values travel as 32 bits in the guest; `SIZE_MAX` stays the
/// sentinel it is (`WGPU_STRLEN`, `WGPU_WHOLE_MAP_SIZE`).
pub(crate) fn usize_from_guest(value: u32) -> usize {
    if value == u32::MAX {
        usize::MAX
    } else {
        value as usize
    }
}

/// The native forms of one call's arguments, and what they point to.
pub(crate) struct Lowering<'a> {
    host: &'a Host,
    /// The object the call is about: its device gets the call's errors.
    this: Option<&'a Object>,
    /// Owned storage for everything the native values point to. Boxes do
    /// not move, so pointers into them stay valid while `self` lives.
    keep: Vec<Box<dyn Any>>,
    /// Objects resolved for this call, kept alive until it returns: a guest
    /// thread may release a handle while another one's call still uses it.
    objects: Vec<Object>,
}

impl<'a> Lowering<'a> {
    pub(crate) fn new(host: &'a Host, this: Option<&'a Object>) -> Self {
        Self {
            host,
            this,
            keep: Vec::new(),
            objects: Vec::new(),
        }
    }

    /// Keeps `value` until the call returns; returns where it lives.
    pub(crate) fn store<T: 'static>(&mut self, value: T) -> *mut T {
        let mut boxed = Box::new(value);
        let ptr: *mut T = &mut *boxed;
        self.keep.push(boxed);
        ptr
    }

    fn store_slice<T: 'static>(&mut self, values: Vec<T>) -> (usize, *const T) {
        if values.is_empty() {
            return (0, std::ptr::null());
        }
        let boxed = values.into_boxed_slice();
        let ptr = boxed.as_ptr();
        let len = boxed.len();
        self.keep.push(Box::new(boxed));
        (len, ptr)
    }

    /// A string view of `text`, which outlives the call (it is part of the
    /// decoded arguments).
    pub(crate) fn str(&mut self, text: &str) -> WGPUStringView {
        WGPUStringView {
            data: text.as_ptr().cast(),
            length: text.len(),
        }
    }

    pub(crate) fn opt_str(&mut self, text: Option<&str>) -> WGPUStringView {
        match text {
            Some(text) => self.str(text),
            None => WGPUStringView::default(),
        }
    }

    /// The GPU object behind `handle`. The null handle is null when the
    /// member is optional, and an error otherwise.
    pub(crate) fn object(
        &mut self,
        kind: Kind,
        handle: Handle,
        member: &'static str,
        optional: bool,
    ) -> Result<*mut c_void, Invalid> {
        if handle == 0 {
            return if optional {
                Ok(std::ptr::null_mut())
            } else {
                Err(Invalid::Missing(member))
            };
        }
        let object = self
            .host
            .object(handle, kind)
            .ok_or(Invalid::Handle { member, handle })?;
        let raw = object.raw;
        self.objects.push(object);
        Ok(raw)
    }

    /// The GPU objects behind `handles`; null ones only if `nullable`.
    pub(crate) fn objects(
        &mut self,
        kind: Kind,
        handles: &[u32],
        member: &'static str,
        nullable: bool,
    ) -> Result<(usize, *const *mut c_void), Invalid> {
        let mut raws = Vec::with_capacity(handles.len());
        for &handle in handles {
            raws.push(self.object(kind, handle, member, nullable)?);
        }
        Ok(self.store_slice(raws))
    }

    pub(crate) fn array<T: Lower>(
        &mut self,
        values: &[T],
    ) -> Result<(usize, *const T::Native), Invalid>
    where
        T::Native: 'static,
    {
        let mut natives = Vec::with_capacity(values.len());
        for value in values {
            natives.push(value.lower(self)?);
        }
        Ok(self.store_slice(natives))
    }

    pub(crate) fn enums<E: AbiEnum>(&mut self, values: &[E]) -> (usize, *const u32) {
        self.store_slice(values.iter().map(|value| value.as_u32()).collect())
    }

    pub(crate) fn u32s(&mut self, values: &[u32]) -> (usize, *const u32) {
        self.store_slice(values.to_vec())
    }

    pub(crate) fn boxed<T: Lower>(&mut self, value: Option<&T>) -> Result<*const T::Native, Invalid>
    where
        T::Native: 'static,
    {
        match value {
            Some(value) => self.lower_ptr(value),
            None => Ok(std::ptr::null()),
        }
    }

    pub(crate) fn lower_ptr<T: Lower>(&mut self, value: &T) -> Result<*const T::Native, Invalid>
    where
        T::Native: 'static,
    {
        let native = value.lower(self)?;
        Ok(self.store(native).cast_const())
    }

    pub(crate) fn lower_opt<T: Lower>(
        &mut self,
        value: Option<&T>,
    ) -> Result<*const T::Native, Invalid>
    where
        T::Native: 'static,
    {
        self.boxed(value)
    }

    /// Reports why the call could not be made: a validation error on the
    /// device of the object it is about, as the GPU library would have.
    pub(crate) fn fail(&self, function: &'static str, error: Invalid) {
        let message = format!("{function}: {error}");
        match self.this.and_then(|this| this.device.as_ref()) {
            Some(device) => device.inject(ErrorType::Validation, &message),
            None => log(|| message),
        }
    }
}

// A forward resolves a few objects; this keeps the common case allocation
// free of surprises in profiles.
const _: () =
    assert!(std::mem::size_of::<ffi::WGPUStringView>() == 2 * std::mem::size_of::<usize>());
