//! 一時検証: 指定モデルを 1 度ロードし、閾値を 1 刻みでスイープして
//! clusterF1 の曲線(ピーク位置・鋭さ)を細かく確認する。
//!
//! ```sh
//! cargo run --example fine_sweep -- gte
//! cargo run --example fine_sweep -- small
//! ```
use anyhow::Result;
use narashi::eval::{default_glossary, sweep};
use narashi::{EmbeddingModel, Model, Narashi, Options, UserModel};

fn main() -> Result<()> {
    let arg = std::env::args().nth(1);
    let (model, label): (Model, &str) = match arg.as_deref() {
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
        // Candle バックエンド専用
        Some("e5-instruct") => (UserModel::E5LargeInstruct.into(), "e5-large-instruct"),
        Some("large") => (EmbeddingModel::MultilingualE5Large.into(), "e5-large"),
        Some("base") => (EmbeddingModel::MultilingualE5Base.into(), "e5-base"),
        Some("mpnet") => (EmbeddingModel::ParaphraseMLMpnetBaseV2.into(), "mpnet"),
        Some("paraphrase") => (
            EmbeddingModel::ParaphraseMLMiniLML12V2.into(),
            "paraphrase-MiniLM-L12",
        ),
        _ => (EmbeddingModel::MultilingualE5Small.into(), "e5-small"),
    };

    let glossary = default_glossary();
    let n = Narashi::with_options(Options::new().with_model(model))?;

    // 40〜95 を 1 刻み
    let thresholds: Vec<f32> = (40..=95).map(|t| t as f32).collect();
    let rows = sweep(&n, &glossary, &thresholds)?;

    println!("== fine sweep: {label} ==");
    println!("閾値 | クラスタF1 (   P  /   R  ) | 分類F1 | 誤統合");
    let mut best = (0.0_f64, 0.0_f32);
    for r in &rows {
        if r.cluster_f1 > best.0 {
            best = (r.cluster_f1, r.threshold);
        }
    }
    for r in &rows {
        let mark = if r.threshold == best.1 {
            " <= peak"
        } else {
            ""
        };
        println!(
            "{:>4.0} |   {:.3}   ( {:.3} / {:.3} ) | {:.3} | {:>5}{}",
            r.threshold,
            r.cluster_f1,
            r.cluster_precision,
            r.cluster_recall,
            r.class_f1,
            r.false_merges,
            mark
        );
    }
    println!("peak clusterF1 = {:.3} @ 閾値 {:.0}", best.0, best.1);
    Ok(())
}
