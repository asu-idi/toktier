//! Minimal borrowed-input surface needed by the vendored pretokenizer.

/// Lightweight reference used by the reference pretokenizer state machine.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DocRef<'a>(pub &'a [u8]);

impl<'a> From<&'a [u8]> for DocRef<'a> {
    fn from(value: &'a [u8]) -> Self {
        Self(value)
    }
}

impl<'a> std::ops::Deref for DocRef<'a> {
    type Target = &'a [u8];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
