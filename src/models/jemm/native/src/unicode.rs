//! Python 3.12 printability from Unicode 15.0 general categories.
//! Source: https://www.unicode.org/Public/15.0.0/ucd/UnicodeData.txt
//! Source SHA256: 806e9aed65037197f1ec85e12be6e8cd870fc5608b4de0fffd990f689f376a73.
//! Table stores alternating printable range boundaries as little-endian u32.
//! Data license: ../LICENSE.unicode.
const BOUNDARIES: &[u8] = include_bytes!("unicode15_printable.bin");
pub fn is_printable(c: char) -> bool {
    let mut low = 0usize;
    let mut high = BOUNDARIES.len() / 4;
    while low < high {
        let middle = (low + high) / 2;
        let bytes = &BOUNDARIES[middle * 4..middle * 4 + 4];
        let point = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        if point <= c as u32 {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    low % 2 == 1
}
