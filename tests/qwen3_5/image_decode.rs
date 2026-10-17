use base64::Engine;
use image::{DynamicImage, ImageBuffer, ImageFormat};
use omni_qwen3_5_native::image_decode::{MAX_IMAGE_BYTES, decode_data_url};
use std::io::Cursor;

fn data_url(mime: &str, raw: &[u8]) -> String {
    format!(
        "data:image/{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(raw)
    )
}

fn png(image: DynamicImage) -> Vec<u8> {
    let mut raw = Cursor::new(Vec::new());
    image.write_to(&mut raw, ImageFormat::Png).unwrap();
    raw.into_inner()
}

fn rejects(url: &str, message: &str) {
    let error = decode_data_url(url).err().expect("image must be rejected");
    assert!(error.to_string().contains(message), "{error}");
}

#[test]
fn decodes_png_and_jpeg_to_rgb() {
    let raw = png(DynamicImage::ImageRgba8(
        ImageBuffer::from_raw(2, 1, vec![20, 40, 60, 0, 80, 100, 120, 128]).unwrap(),
    ));
    let decoded = decode_data_url(&data_url("png", &raw)).unwrap();
    assert_eq!((decoded.width, decoded.height), (2, 1));
    assert_eq!(decoded.rgb, [20, 40, 60, 80, 100, 120]);

    let mut raw = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut raw)
        .encode(&[20, 40, 60], 1, 1, image::ExtendedColorType::Rgb8)
        .unwrap();
    let decoded = decode_data_url(&data_url("jpeg", &raw)).unwrap();
    assert_eq!((decoded.width, decoded.height), (1, 1));
    assert_eq!(decoded.rgb.len(), 3);
}

#[test]
fn rejects_non_inline_urls_invalid_base64_and_mime_mismatch() {
    rejects("https://example.com/image.png", "invalid image data URL");
    rejects("data:image/webp;base64,AAAA", "only inline PNG/JPEG");
    rejects("data:image/png;base64,%%%", "invalid base64 image");
    let raw = png(DynamicImage::new_rgb8(1, 1));
    rejects(&data_url("jpeg", &raw), "does not match MIME type");
}

#[test]
fn enforces_encoded_and_decoded_four_mib_limits() {
    let mut raw = png(DynamicImage::new_rgb8(1, 1));
    raw.resize(MAX_IMAGE_BYTES, 0);
    assert!(decode_data_url(&data_url("png", &raw)).is_ok());
    // Both lengths encode to the same base64 length, exercising the raw limit.
    raw.push(0);
    rejects(&data_url("png", &raw), "image exceeds 4 MiB");
    rejects(
        &format!(
            "data:image/png;base64,{}",
            "A".repeat(4 * MAX_IMAGE_BYTES.div_ceil(3) + 4)
        ),
        "encoded image exceeds 4 MiB",
    );
}

#[test]
fn enforces_source_dimensions_area_and_aspect_ratio() {
    for (width, height, accepted) in [
        (2048, 512, true),
        (2049, 400, false),
        (400, 2049, false),
        (1024, 1024, true),
        (1024, 1025, false),
        (200, 1, true),
        (201, 1, false),
    ] {
        let raw = png(DynamicImage::new_rgb8(width, height));
        assert_eq!(
            decode_data_url(&data_url("png", &raw)).is_ok(),
            accepted,
            "{width}x{height}"
        );
    }
}

#[test]
fn rejects_apng_instead_of_silently_selecting_a_frame() {
    // Pillow: RGB 1x1 red.save(format="PNG", save_all=True,
    // append_images=[RGB 1x1 blue], duration=100, loop=0).
    rejects(
        concat!(
            "data:image/png;base64,",
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAACGFjVEwAAAACAAAAAPONk3",
            "AAAAAaZmNUTAAAAAAAAAABAAAAAQAAAAAAAAAAAAEACgAAWn8w0AAAAAxJREFUeJxj+M/A",
            "AAADAQEAyf6S7wAAABpmY1RMAAAAAQAAAAEAAAABAAAAAAAAAAAAAQAKAADBDNoEAAAAEG",
            "ZkQVQAAAACeJxjYGD4DwABAwEAaL/t9AAAAABJRU5ErkJggg=="
        ),
        "single-frame",
    );
}

#[test]
fn preserves_pillow_rgb_conversion_for_sixteen_bit_png() {
    for (image, expected) in [
        (
            DynamicImage::ImageLuma16(ImageBuffer::from_raw(3, 1, vec![128, 256, 65535]).unwrap()),
            vec![128, 128, 128, 255, 255, 255, 255, 255, 255],
        ),
        (
            DynamicImage::ImageLumaA16(ImageBuffer::from_raw(1, 1, vec![0x1234, 0]).unwrap()),
            vec![0x12; 3],
        ),
        (
            DynamicImage::ImageRgb16(
                ImageBuffer::from_raw(1, 1, vec![0x1234, 0x5678, 0x9abc]).unwrap(),
            ),
            vec![0x12, 0x56, 0x9a],
        ),
        (
            DynamicImage::ImageRgba16(
                ImageBuffer::from_raw(1, 1, vec![0x1234, 0x5678, 0x9abc, 0]).unwrap(),
            ),
            vec![0x12, 0x56, 0x9a],
        ),
    ] {
        let decoded = decode_data_url(&data_url("png", &png(image))).unwrap();
        assert_eq!(decoded.rgb, expected);
    }
    // Pillow I;16 with tRNS keeps grayscale's clamp-to-255 conversion.
    let decoded = decode_data_url(concat!(
        "data:image/png;base64,",
        "iVBORw0KGgoAAAANSUhEUgAAAAMAAAABEAAAAABuG5crAAAAAnRSTlMBAG+I/HkAAAAP",
        "SURBVHicY2BoYGT4/x8ABYgCgNJVtjQAAAAASUVORK5CYII="
    ))
    .unwrap();
    assert_eq!(decoded.rgb, [128, 128, 128, 255, 255, 255, 255, 255, 255]);
}
