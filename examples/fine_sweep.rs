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
        Some("large") => (EmbeddingModel::MultilingualE5Large.into(), "e5-large"),
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
    println!("閾値 | クラスタF1 (   P  /   R  ) | 分類F1");
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
            "{:>4.0} |   {:.3}   ( {:.3} / {:.3} ) | {:.3}{}",
            r.threshold, r.cluster_f1, r.cluster_precision, r.cluster_recall, r.class_f1, mark
        );
    }
    println!("peak clusterF1 = {:.3} @ 閾値 {:.0}", best.0, best.1);
    Ok(())
}
