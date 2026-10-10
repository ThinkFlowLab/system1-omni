//! Image decoding and pixel values against the reference's `Image.open(...).convert("RGB")`
//! and processor (tests/omnijev/data/pixels.json).
mod common;

use base64::Engine;
use omni_omnijev_native::{contract::Image, processing};
use omni_qwen3_5_native::image_decode::{DecodeLimits, decode_rgb8};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// The generator's pattern: channel values from x and y.
fn pattern(width: u32, height: u32, x: u32, y: u32) -> [u8; 3] {
    let _ = (width, height);
    let (x, y) = (x as u64, y as u64);
    [
        ((x * 7 + y * 3) % 256) as u8,
        ((x * y + 13) % 256) as u8,
        (((x ^ y) * 5) % 256) as u8,
    ]
}

/// The case's file: redrawn for pattern cases, stored otherwise.
fn file(case: &Value) -> Vec<u8> {
    let [w, h] = [0, 1].map(|i| case["size"][i].as_u64().unwrap() as u32);
    let mut bytes = Vec::new();
    let mut out = std::io::Cursor::new(&mut bytes);
    match case["kind"].as_str().unwrap() {
        "pattern" => image::RgbImage::from_fn(w, h, |x, y| image::Rgb(pattern(w, h, x, y)))
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap(),
        "pattern_rgba" => image::RgbaImage::from_fn(w, h, |x, y| {
            let [r, g, b] = pattern(w, h, x, y);
            image::Rgba([r, g, b, ((x + y) % 256) as u8])
        })
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap(),
        "pattern_gray" => {
            image::GrayImage::from_fn(w, h, |x, y| image::Luma([pattern(w, h, x, y)[0]]))
                .write_to(&mut out, image::ImageFormat::Png)
                .unwrap()
        }
        _ => {
            return base64::engine::general_purpose::STANDARD
                .decode(case["bytes"].as_str().unwrap())
                .unwrap();
        }
    }
    bytes
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn f32_sha256(values: &[f32]) -> String {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    sha256(&bytes)
}

#[test]
fn decoding_and_pixel_values_match_reference() {
    let fixture = common::fixture("pixels.json");
    let limits = DecodeLimits {
        max_side: 4096,
        max_pixels: 4096 * 4096,
        max_aspect: 200,
        max_alloc: 4096 * 4096 * 8,
    };
    for case in fixture["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let format = match case["format"].as_str().unwrap() {
            "jpeg" => image::ImageFormat::Jpeg,
            _ => image::ImageFormat::Png,
        };
        let bytes = file(case);
        let decoded = decode_rgb8(&bytes, format, &limits).unwrap();
        let size = [decoded.width, decoded.height];
        assert_eq!(serde_json::json!(size), case["size"], "{name}");
        let image = Image {
            format,
            width: decoded.width,
            height: decoded.height,
            bytes,
        };
        let pixels = processing::pixels(&image).unwrap();
        if format == image::ImageFormat::Png {
            assert_eq!(
                sha256(&decoded.rgb),
                case["rgb_sha256"].as_str().unwrap(),
                "{name}"
            );
            assert_eq!(
                serde_json::json!(pixels.image_grid_thw),
                case["image_grid_thw"],
                "{name}"
            );
            assert_eq!(
                f32_sha256(&pixels.pixel_values),
                case["pixel_values_sha256"].as_str().unwrap(),
                "{name}"
            );
        } else {
            // JPEG decoders may differ by rounding: the bytes against Pillow's, and the
            // processor alone on Pillow's pixels.
            let reference = base64::engine::general_purpose::STANDARD
                .decode(case["rgb"].as_str().unwrap())
                .unwrap();
            let worst = decoded
                .rgb
                .iter()
                .zip(&reference)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            let changed = decoded
                .rgb
                .iter()
                .zip(&reference)
                .filter(|(a, b)| a != b)
                .count();
            eprintln!(
                "{name}: {changed} of {} values differ from Pillow's, by at most {worst}",
                reference.len()
            );
            assert!(worst <= 3, "{name}: {worst}");
            let from_reference =
                processing::preprocess(decoded.width, decoded.height, &reference).unwrap();
            assert_eq!(
                f32_sha256(&from_reference.pixel_values),
                case["pixel_values_sha256"].as_str().unwrap(),
                "{name}"
            );
        }
    }
}
