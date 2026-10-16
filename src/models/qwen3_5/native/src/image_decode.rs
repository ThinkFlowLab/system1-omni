//! Bounded PNG/JPEG decoding to the interleaved RGB8 that Pillow's
//! `Image.open(...).convert("RGB")` gives, for the native vision workers.
//!
//! Single-frame PNG and JPEG only: APNG and MPO containers are refused rather than
//! decoded to their first picture. Sixteen-bit samples follow Pillow's conversions.

use std::io::Cursor;

use anyhow::{Context, Result, ensure};
use image::{DynamicImage, ImageFormat};

/// Source limits checked from the header before decoding.
#[derive(Clone, Copy, Debug)]
pub struct DecodeLimits {
    pub max_side: usize,
    pub max_pixels: usize,
    /// Longest side over shortest side.
    pub max_aspect: usize,
    /// Largest decoder allocation in bytes.
    pub max_alloc: u64,
}

/// An interleaved RGB8 image.
pub struct DecodedImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

/// Decode `raw`, whose bytes must be a `format` image, within `limits`.
pub fn decode_rgb8(raw: &[u8], format: ImageFormat, limits: &DecodeLimits) -> Result<DecodedImage> {
    ensure!(
        matches!(format, ImageFormat::Png | ImageFormat::Jpeg),
        "only PNG and JPEG images are supported"
    );
    ensure!(
        image::guess_format(raw)? == format,
        "image format does not match MIME type"
    );
    if format == ImageFormat::Jpeg {
        ensure_single_jpeg(raw)?;
    }
    let mut decoder_limits = image::Limits::default();
    decoder_limits.max_image_width = Some(limits.max_side as u32);
    decoder_limits.max_image_height = Some(limits.max_side as u32);
    decoder_limits.max_alloc = Some(limits.max_alloc);
    let mut header = image::ImageReader::with_format(Cursor::new(raw), format);
    header.limits(decoder_limits.clone());
    let (width, height) = header.into_dimensions()?;
    let (w, h) = (width as usize, height as usize);
    ensure!(
        w > 0
            && h > 0
            && w <= limits.max_side
            && h <= limits.max_side
            && w * h <= limits.max_pixels
            && w.max(h) <= limits.max_aspect * w.min(h),
        "image dimensions exceed supported limits"
    );
    let decoded = if format == ImageFormat::Png {
        let decoder =
            image::codecs::png::PngDecoder::with_limits(Cursor::new(raw), decoder_limits)?;
        ensure!(!decoder.is_apng()?, "image must be single-frame");
        DynamicImage::from_decoder(decoder)?
    } else {
        let mut reader = image::ImageReader::with_format(Cursor::new(raw), format);
        reader.limits(decoder_limits);
        reader.decode()?
    };
    Ok(DecodedImage {
        width: w,
        height: h,
        // byte 25 of a PNG is its IHDR color type; 0 is grayscale without alpha
        rgb: pillow_rgb(
            decoded,
            format == ImageFormat::Png && raw.get(25) == Some(&0),
        ),
    })
}

fn ensure_single_jpeg(raw: &[u8]) -> Result<()> {
    // MPF APP2 identifies an MPO container. Pillow reports it as MPO rather
    // than JPEG; decoding it as JPEG would silently select the first picture.
    // Walk header segments only, so arbitrary metadata/entropy bytes cannot
    // be mistaken for an MPF marker.
    let mut offset = 2; // SOI was checked by guess_format.
    while offset < raw.len() {
        ensure!(raw[offset] == 0xff, "invalid JPEG marker");
        while raw.get(offset) == Some(&0xff) {
            offset += 1;
        }
        let marker = *raw.get(offset).context("truncated JPEG marker")?;
        offset += 1;
        match marker {
            0xda | 0xd9 => break,           // SOS / EOI; the decoder checks the rest.
            0x01 | 0xd0..=0xd8 => continue, // Standalone markers.
            _ => {}
        }
        let length = raw
            .get(offset..offset + 2)
            .context("truncated JPEG segment")?;
        let length = u16::from_be_bytes([length[0], length[1]]) as usize;
        ensure!(length >= 2, "invalid JPEG segment length");
        let segment = raw
            .get(offset + 2..offset + length)
            .context("truncated JPEG segment")?;
        ensure!(
            marker != 0xe2 || !segment.starts_with(b"MPF\0"),
            "image must be single-frame; MPF/MPO containers are unsupported"
        );
        offset += length;
    }
    Ok(())
}

fn pillow_rgb(image: DynamicImage, png_grayscale: bool) -> Vec<u8> {
    use DynamicImage::*;
    match image {
        ImageLuma16(p) => p.pixels().flat_map(|v| [v[0].min(255) as u8; 3]).collect(),
        ImageLumaA16(p) if png_grayscale => {
            p.pixels().flat_map(|v| [v[0].min(255) as u8; 3]).collect()
        }
        ImageLumaA16(p) => p.pixels().flat_map(|v| [(v[0] >> 8) as u8; 3]).collect(),
        ImageRgb16(p) => p
            .pixels()
            .flat_map(|v| [(v[0] >> 8) as u8, (v[1] >> 8) as u8, (v[2] >> 8) as u8])
            .collect(),
        ImageRgba16(p) => p
            .pixels()
            .flat_map(|v| [(v[0] >> 8) as u8, (v[1] >> 8) as u8, (v[2] >> 8) as u8])
            .collect(),
        other => other.to_rgb8().into_raw(),
    }
}
