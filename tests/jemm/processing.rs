use omni_jemm_native::processing::{decode_images, image_positions};
use serde_json::json;
#[test]
fn image_decoder_accepts_rgb_png_data_urls_and_whitespace() {
    use base64::Engine;
    let mut bytes = std::io::Cursor::new(Vec::new());
    image::RgbImage::from_pixel(32, 32, image::Rgb([12, 34, 56]))
        .write_to(&mut bytes, image::ImageFormat::Png)
        .unwrap();
    let data = base64::engine::general_purpose::STANDARD.encode(bytes.into_inner());
    let images = decode_images(&json!([format!("data:image/png;base64,\n\u{1c}{data}")])).unwrap();
    assert_eq!(images[0].get_pixel(0, 0).0, [12, 34, 56]);
    assert!(decode_images(&json!(["bad"])).is_err());
    assert!(decode_images(&json!(["a", "b", "c", "d", "e"])).is_err());
    assert!(decode_images(&json!(null)).unwrap().is_empty());
}
#[test]
fn multimodal_positions_restart_text_after_grid_maximum() {
    // 2x3 merged rows, then two text tokens, then a 1x2 image.
    let ids = [1, 99, 99, 99, 99, 99, 99, 2, 3, 99, 99, 4];
    let p = image_positions(&ids, 99, &[[1, 4, 6], [1, 2, 4]]).unwrap();
    assert_eq!(p[0], [0, 1, 1, 1, 1, 1, 1, 4, 5, 6, 6, 8]);
    assert_eq!(p[1], [0, 1, 1, 1, 2, 2, 2, 4, 5, 6, 6, 8]);
    assert_eq!(p[2], [0, 1, 2, 3, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(image_positions(&ids, 99, &[[1, 4, 4]]).is_err());
}
