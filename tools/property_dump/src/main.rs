//! Write the character classes the fast CPU pre-tokenizer reads, for every
//! code point, as the `icu_properties` data this build resolves answers them.
//!
//! The engine's own source file is compiled in unchanged, so the dump is the
//! engine's answer and not a transcription of it. The output is six bytes
//! per code point, for every value below 0x110000 in order: `class_of`,
//! `ds_class_of`, `o200k_class_of`, `kimi_class_of`, the general category
//! (255 for a surrogate), and White_Space (255 for a surrogate). Every
//! accessor of the engine that reads the property data is one of these
//! columns or a function of them (`is_letter`, `is_number` and
//! `is_other_complete` of the general category and White_Space; the packed
//! tables behind `ClassTable` and `DsClassTable` are the first two columns).
//!
//! Usage: `toktier-property-dump <output path>`.

#![allow(dead_code, unused_imports)]

#[path = "../../../crates/toktier-gigatoken-core/src/pretokenize/unicode.rs"]
mod unicode;

use std::io::Write;

/// Bytes written per code point.
const COLUMNS: usize = 6;
/// One past the largest code point.
const CODE_POINTS: u32 = 0x110000;

fn main() {
    let output = std::env::args()
        .nth(1)
        .expect("usage: toktier-property-dump <output path>");
    let mut bytes = Vec::with_capacity(CODE_POINTS as usize * COLUMNS);
    for cp in 0..CODE_POINTS {
        let scalar = char::from_u32(cp);
        bytes.push(unicode::class_of(cp) as u8);
        bytes.push(unicode::ds_class_of(cp) as u8);
        bytes.push(unicode::o200k_class_of(cp) as u8);
        bytes.push(unicode::kimi_class_of(cp) as u8);
        bytes.push(scalar.map_or(255, |c| unicode::get_general_category(c) as u8));
        bytes.push(scalar.map_or(255, |c| u8::from(unicode::is_whitespace(c))));
    }
    std::fs::File::create(&output)
        .and_then(|mut file| file.write_all(&bytes))
        .unwrap_or_else(|error| panic!("cannot write {output}: {error}"));
    eprintln!("wrote {} bytes to {output}", bytes.len());
}
