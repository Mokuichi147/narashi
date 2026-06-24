//! 全ペアのスコアを TSV でダンプする(難易度分析用)。
//! `pos<TAB>score<TAB>term_a<TAB>term_b` を 1 ペア 1 行で出力。
//!
//! ONNX 勢:  cargo run --release --features onnx --example pairdump -- bge-m3 > x.tsv
//! Candle 勢: cargo run --release --no-default-features --features candle --example pairdump -- qwen3 > x.tsv
use anyhow::Result;
use narashi::eval::{all_scored_pairs, default_glossary};
use narashi::{Model, Narashi, Options};

fn pick(arg: Option<&str>) -> (Model, &'static str) {
    use narashi::UserModel as U;
    match arg {
        Some("bge-m3") => (U::BgeM3.into(), "bge-m3"),
        Some("gte") => (U::GteMultilingualBase.into(), "gte"),
        Some("distiluse") => (U::DistiluseMultilingualV2.into(), "distiluse"),
        Some("granite-278m") => (U::GraniteMultilingual278m.into(), "granite-278m"),
        Some("qwen3") => (U::Qwen3Embedding0_6B.into(), "qwen3"),
        Some("qwen3-4b") => (U::Qwen3Embedding4B.into(), "qwen3-4b"),
        Some("e5-instruct") => (U::E5LargeInstruct.into(), "e5-instruct"),
        #[cfg(feature = "onnx")]
        Some("small") => (narashi::EmbeddingModel::MultilingualE5Small.into(), "small"),
        #[cfg(feature = "onnx")]
        Some("large") => (narashi::EmbeddingModel::MultilingualE5Large.into(), "large"),
        _ => (U::BgeM3.into(), "bge-m3"),
    }
}

fn main() -> Result<()> {
    let arg = std::env::args().nth(1);
    let (model, _label) = pick(arg.as_deref());
    let n = Narashi::with_options(Options::new().with_model(model))?;
    let pairs = all_scored_pairs(&n, &default_glossary())?;
    for (a, b, pos, s) in pairs {
        println!("{}\t{:.2}\t{}\t{}", if pos { 1 } else { 0 }, s, a, b);
    }
    Ok(())
}
