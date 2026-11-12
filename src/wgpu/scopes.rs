//! WebGPU error scopes.
//!
//! `webgpu.h` keeps one error-scope stack per device per thread. wgpu's own
//! scopes are thread-local too, but they are guard objects that cannot be
//! stored or sent, which does not fit a handle table shared by guest threads.
//! So every device installs one uncaptured-error handler and the scopes live
//! here: an error is first offered to the calling thread's stack for that
//! device, and only becomes an uncaptured-error event if no scope takes it.

use std::cell::RefCell;

use crate::abi::enums::{ErrorFilter, ErrorType};

struct Scope {
    device: u64,
    filter: ErrorFilter,
    /// The first error the scope captured.
    error: Option<(ErrorType, String)>,
}

thread_local! {
    static SCOPES: RefCell<Vec<Scope>> = const { RefCell::new(Vec::new()) };
}

fn matches(filter: ErrorFilter, ty: ErrorType) -> bool {
    matches!(
        (filter, ty),
        (ErrorFilter::Validation, ErrorType::Validation)
            | (ErrorFilter::OutOfMemory, ErrorType::OutOfMemory)
            | (ErrorFilter::Internal, ErrorType::Internal)
    )
}

pub(crate) fn push(device: u64, filter: ErrorFilter) {
    SCOPES.with_borrow_mut(|scopes| {
        scopes.push(Scope {
            device,
            filter,
            error: None,
        });
    });
}

/// Pops the calling thread's innermost scope of `device`. `None` when that
/// stack is empty; otherwise the error the scope captured, if any.
pub(crate) fn pop(device: u64) -> Option<Option<(ErrorType, String)>> {
    SCOPES.with_borrow_mut(|scopes| {
        let at = scopes.iter().rposition(|scope| scope.device == device)?;
        Some(scopes.remove(at).error)
    })
}

/// Offers an error to the calling thread's scopes of `device`. Returns
/// whether one of them took it.
pub(crate) fn capture(device: u64, ty: ErrorType, message: &str) -> bool {
    SCOPES.with_borrow_mut(|scopes| {
        let Some(scope) = scopes
            .iter_mut()
            .rev()
            .find(|scope| scope.device == device && matches(scope.filter, ty))
        else {
            return false;
        };
        if scope.error.is_none() {
            scope.error = Some((ty, message.to_owned()));
        }
        true
    })
}

/// Forgets the calling thread's scopes of a device that is gone.
pub(crate) fn forget(device: u64) {
    SCOPES.with_borrow_mut(|scopes| scopes.retain(|scope| scope.device != device));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_innermost_matching_scope_captures_the_first_error() {
        push(1, ErrorFilter::Validation);
        push(1, ErrorFilter::OutOfMemory);
        push(2, ErrorFilter::Validation);

        assert!(capture(1, ErrorType::Validation, "first"));
        assert!(capture(1, ErrorType::Validation, "second"));
        assert!(!capture(1, ErrorType::Internal, "nobody listens"));

        assert_eq!(pop(1), Some(None), "the out-of-memory scope saw nothing");
        assert_eq!(
            pop(1),
            Some(Some((ErrorType::Validation, "first".to_owned())))
        );
        assert_eq!(pop(1), None, "device 1 has no scope left");
        assert_eq!(pop(2), Some(None));
    }

    #[test]
    fn scopes_are_per_thread() {
        push(7, ErrorFilter::Validation);
        std::thread::spawn(|| {
            assert!(!capture(7, ErrorType::Validation, "other thread"));
            assert_eq!(pop(7), None);
        })
        .join()
        .unwrap();
        assert_eq!(pop(7), Some(None));
    }
}
