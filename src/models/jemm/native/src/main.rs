#[tokio::main]
async fn main() -> anyhow::Result<()> {
    omni_jemm_native::server::run().await
}
