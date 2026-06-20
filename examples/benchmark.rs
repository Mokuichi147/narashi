//! 用語集 (tests/data/glossary.txt) に対してモデルの共通スペックを算出する。
//!
//! モデルを差し替えるたびに実行し、同じ指標で精度・速度を横並び比較するための
//! ベンチマークです。
//!
//! ```sh
//! cargo run --example benchmark                 # 既定モデル (e5-small)
//! cargo run --example benchmark -- paraphrase   # paraphrase-multilingual
//! cargo run --example benchmark -- mpnet        # paraphrase-mpnet-base
//! cargo run --example benchmark -- large        # e5-large
//! cargo run --example benchmark -- bge-zh       # 別系統: BGE 中国語特化
//! cargo run --example benchmark -- all-minilm   # 別系統: 英語 MiniLM
//! cargo run --example benchmark -- clip         # 別系統: CLIP テキスト
//! cargo run --example benchmark -- base 75      # モデルと閾値を指定
//! ```

use anyhow::Result;
use narashi::eval::{default_glossary, evaluate_with_load};
use narashi::{DEFAULT_THRESHOLD, EmbeddingModel, Narashi, Options};
use std::time::Instant;

fn main() -> Result<()> {
    let arg = std::env::args().nth(1);
    let (model, label) = match arg.as_deref() {
        Some("paraphrase") => (
            EmbeddingModel::ParaphraseMLMiniLML12V2,
            "paraphrase-MiniLM-L12-v2",
        ),
        Some("paraphrase-q") => (
            EmbeddingModel::ParaphraseMLMiniLML12V2Q,
            "paraphrase-MiniLM-L12-v2 (quantized)",
        ),
        Some("mpnet") => (
            EmbeddingModel::ParaphraseMLMpnetBaseV2,
            "paraphrase-mpnet-base-v2",
        ),
        Some("base") => (EmbeddingModel::MultilingualE5Base, "e5-base"),
        Some("large") => (EmbeddingModel::MultilingualE5Large, "e5-large"),
        // 別系統モデル (比較用ベースライン)
        Some("bge-zh") => (EmbeddingModel::BGESmallZHV15, "bge-small-zh-v1.5"),
        Some("all-minilm") => (EmbeddingModel::AllMiniLML6V2, "all-MiniLM-L6-v2 (英語)"),
        Some("clip") => (EmbeddingModel::ClipVitB32, "clip-ViT-B-32-text"),
        // 既定はライブラリの既定モデル (e5-small)
        _ => (EmbeddingModel::MultilingualE5Small, "e5-small"),
    };
    let threshold = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_THRESHOLD);

    let glossary = default_glossary();

    let t0 = Instant::now();
    let n = Narashi::with_options(Options::new().with_model(model))?;
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let report = evaluate_with_load(&n, &glossary, threshold, label, load_ms)?;
    println!("{report}");
    Ok(())
}
