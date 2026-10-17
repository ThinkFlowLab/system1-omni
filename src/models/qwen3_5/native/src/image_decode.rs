//! Bounded single-frame PNG/JPEG data URL decoding shared by native workers.
use anyhow::{Context, Result, ensure};
use base64::Engine;
use std::io::Cursor;

pub const MAX_IMAGE_BYTES: usize = 4 * 1024 * 1024;

pub struct DecodedImage {
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
}

pub fn decode_data_url(url: &str) -> Result<DecodedImage> {
    let (prefix, encoded) = url.split_once(',').context("invalid image data URL")?;
    let format = match prefix {
        "data:image/png;base64" => image::ImageFormat::Png,
        "data:image/jpeg;base64" => image::ImageFormat::Jpeg,
        _ => anyhow::bail!("only inline PNG/JPEG images are supported"),
    };
    ensure!(
        encoded.len() <= 4 * MAX_IMAGE_BYTES.div_ceil(3),
        "encoded image exceeds 4 MiB"
    );
    let raw = base64::engine::general_purpose::STANDARD
        .decode(encoded)
        .context("invalid base64 image")?;
    ensure!(raw.len() <= MAX_IMAGE_BYTES, "image exceeds 4 MiB");
    ensure!(
        image::guess_format(&raw)? == format,
        "image format does not match MIME type"
    );
    if format == image::ImageFormat::Jpeg {
        ensure_single_jpeg(&raw)?;
    }
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(2048);
    limits.max_image_height = Some(2048);
    limits.max_alloc = Some(32 * 1024 * 1024);
    let mut header = image::ImageReader::with_format(Cursor::new(&raw), format);
    header.limits(limits.clone());
    let (width, height) = header.into_dimensions()?;
    let (w, h) = (width as usize, height as usize);
    ensure!(
        w > 0 && h > 0 && w <= 2048 && h <= 2048 && w * h <= 1048576 && w.max(h) <= 200 * w.min(h),
        "image dimensions exceed supported limits"
    );
    let decoded = if format == image::ImageFormat::Png {
        let decoder = image::codecs::png::PngDecoder::with_limits(Cursor::new(&raw), limits)?;
        ensure!(!decoder.is_apng()?, "image must be single-frame");
        image::DynamicImage::from_decoder(decoder)?
    } else {
        let mut reader = image::ImageReader::with_format(Cursor::new(&raw), format);
        reader.limits(limits);
        reader.decode()?
    };
    Ok(DecodedImage {
        width: w,
        height: h,
        rgb: pillow_rgb(
            decoded,
            format == image::ImageFormat::Png && raw.get(25) == Some(&0),
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

fn pillow_rgb(image: image::DynamicImage, png_grayscale: bool) -> Vec<u8> {
    use image::DynamicImage::*;
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
