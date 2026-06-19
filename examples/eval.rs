//! スコア校正と既定閾値を実データで確認するための評価サンプル。
//!
//! ```sh
//! cargo run --example eval                 # 既定 (E5 small)
//! cargo run --example eval -- paraphrase   # paraphrase-multilingual を使用
//! ```
//!
//! 類似ペアが非類似ペアより明確に高いスコアになり、選定した閾値で
//! 「統合すべきペアだけが閾値を超える」ことを確認する。

use anyhow::Result;
use narashi::{EmbeddingModel, Narashi, Options};

fn main() -> Result<()> {
    let model = match std::env::args().nth(1).as_deref() {
        Some("small") => EmbeddingModel::MultilingualE5Small,
        Some("base") => EmbeddingModel::MultilingualE5Base,
        Some("paraphrase-q") => EmbeddingModel::ParaphraseMLMiniLML12V2Q,
        // 既定はライブラリの既定モデル (paraphrase-multilingual-MiniLM-L12-v2)
        _ => EmbeddingModel::ParaphraseMLMiniLML12V2,
    };

    let n = Narashi::with_options(Options::new().with_model(model))?;

    // (a, b, 統合されるべきか)
    let cases: &[(&str, &str, bool)] = &[
        ("白い背景", "白背景", true),
        ("漫画", "マンガ", true),
        ("猫", "ネコ", true),
        ("コンピュータ", "コンピューター", true),
        ("頬紅", "照れ", false),
        ("犬", "自動車", false),
        ("赤", "青", false),
    ];

    println!("{:<14} {:<14} {:>7}  期待", "A", "B", "score");
    println!("{}", "-".repeat(48));
    for &(a, b, should_merge) in cases {
        let s = n.similarity(a, b)?;
        let expect = if should_merge { "統合" } else { "非統合" };
        println!("{a:<14} {b:<14} {s:>7.1}  {expect}");
    }
    Ok(())
}
