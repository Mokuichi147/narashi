//! 用語集 (tests/data/glossary.txt) に対してモデルの共通スペックを算出する。
//!
//! モデルを差し替えるたびに実行し、同じ指標で精度・速度を横並び比較するための
//! ベンチマークです。
//!
//! ```sh
//! cargo run --example benchmark                 # 既定モデル (bge-m3)
//! cargo run --example benchmark -- bge-m3       # bge-m3 (既定・clusterF1最高0.725・誤統合最小)
//! cargo run --example benchmark -- gte          # gte-multilingual-base (速度重視の代替)
//! cargo run --example benchmark -- granite-278m # granite-278m-multilingual (clusterF1高め・誤統合多め)
//! cargo run --example benchmark -- distiluse    # distiluse-base-multilingual-cased-v2 (軽量代替)
//! cargo run --example benchmark -- small        # multilingual-e5-small
//! cargo run --example benchmark -- paraphrase   # paraphrase-multilingual
//! cargo run --example benchmark -- mpnet        # paraphrase-mpnet-base
//! cargo run --example benchmark -- large        # e5-large
//! cargo run --example benchmark -- gte          # gte-multilingual-base (既定・ユーザー定義)
//! cargo run --example benchmark -- bge-zh       # 別系統: BGE 中国語特化
//! cargo run --example benchmark -- all-minilm   # 別系統: 英語 MiniLM
//! cargo run --example benchmark -- clip         # 別系統: CLIP テキスト
//! cargo run --example benchmark -- base 75      # モデルと閾値を指定
//! ```

use anyhow::Result;
use narashi::eval::{default_glossary, evaluate_with_load};
use narashi::{DEFAULT_THRESHOLD, EmbeddingModel, Model, Narashi, Options, UserModel};
use std::time::Instant;

fn main() -> Result<()> {
    let arg = std::env::args().nth(1);
    let (model, label): (Model, &str) = match arg.as_deref() {
        Some("paraphrase") => (
            EmbeddingModel::ParaphraseMLMiniLML12V2.into(),
            "paraphrase-MiniLM-L12-v2",
        ),
        Some("paraphrase-q") => (
            EmbeddingModel::ParaphraseMLMiniLML12V2Q.into(),
            "paraphrase-MiniLM-L12-v2 (quantized)",
        ),
        Some("mpnet") => (
            EmbeddingModel::ParaphraseMLMpnetBaseV2.into(),
            "paraphrase-mpnet-base-v2",
        ),
        Some("base") => (EmbeddingModel::MultilingualE5Base.into(), "e5-base"),
        Some("large") => (EmbeddingModel::MultilingualE5Large.into(), "e5-large"),
        // ユーザー定義モデル (組み込みカタログに無い多言語候補)
        Some("gte") => (
            UserModel::GteMultilingualBase.into(),
            "gte-multilingual-base",
        ),
        Some("distiluse") => (
            UserModel::DistiluseMultilingualV2.into(),
            "distiluse-multilingual-v2",
        ),
        Some("granite-97m-r2") => (
            UserModel::GraniteMultilingual97mR2.into(),
            "granite-97m-multilingual-r2",
        ),
        Some("granite-107m") => (
            UserModel::GraniteMultilingual107m.into(),
            "granite-107m-multilingual",
        ),
        Some("granite-278m") => (
            UserModel::GraniteMultilingual278m.into(),
            "granite-278m-multilingual",
        ),
        Some("granite-311m-r2") => (
            UserModel::GraniteMultilingual311mR2.into(),
            "granite-311m-multilingual-r2",
        ),
        Some("bge-m3") => (UserModel::BgeM3.into(), "bge-m3"),
        Some("arctic-l") => (
            UserModel::ArcticEmbedLV2.into(),
            "snowflake-arctic-embed-l-v2.0",
        ),
        // Candle バックエンド専用 (ONNX 変換が外部重み付きで fastembed では読めないモデル)
        Some("e5-instruct") => (UserModel::E5LargeInstruct.into(), "e5-large-instruct"),
        // 別系統モデル (比較用ベースライン)
        Some("bge-zh") => (EmbeddingModel::BGESmallZHV15.into(), "bge-small-zh-v1.5"),
        Some("all-minilm") => (
            EmbeddingModel::AllMiniLML6V2.into(),
            "all-MiniLM-L6-v2 (英語)",
        ),
        Some("clip") => (EmbeddingModel::ClipVitB32.into(), "clip-ViT-B-32-text"),
        Some("small") => (EmbeddingModel::MultilingualE5Small.into(), "e5-small"),
        // 既定はライブラリの既定モデル (bge-m3)
        _ => (UserModel::BgeM3.into(), "bge-m3"),
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
