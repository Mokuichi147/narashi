//! 連鎖暴走(over-merge)の堅牢性ベンチ。
//!
//! 用語集(tests/data/glossary.txt)に大量のディストラクタ(tests/data/distractors.txt:
//! 数字・単一文字などのハブ語 + 多ドメインの無関係タグ)を連結し、閾値を 1 刻みで
//! スイープする。各閾値で「最大予測クラスタが全語の何%を飲み込んだか
//! (largest_cluster_ratio)」「そこに何個の無関係な正解グループが巻き込まれたか
//! (groups_in_largest)」を表示し、単連結クラスタリングが巨大ゴミクラスタへ崩壊する
//! 「暴走オンセット閾値」を明示する。
//!
//! 既定ベンチ(benchmark / fine_sweep)はクリーンな小規模用語集のため連鎖暴走を
//! 再現できない。このストレスベンチは規模 × ハブ密度で暴走を再現し、運用閾値を
//! 「clusterF1 のピーク」ではなく「暴走オンセットより安全マージンを取った側」で
//! 選ぶための判断材料を与える。
//!
//! ```sh
//! cargo run --example robustness                 # 既定 (bge-m3)
//! cargo run --example robustness -- gte
//! cargo run --example robustness -- distiluse
//! # Candle 勢(qwen3 系)は --no-default-features --features candle でビルドし
//! # release で実行する(数百語 × 低速のため時間がかかる)。
//! cargo run --release --no-default-features --features candle --example robustness -- qwen3
//! ```
use anyhow::Result;
#[cfg(feature = "onnx")]
use narashi::EmbeddingModel;
use narashi::eval::{default_distractors, default_glossary, sweep};
use narashi::{Model, Narashi, Options, UserModel};

/// このしきい値以上を「暴走」と判定する経験則(表示用)。
/// 最大クラスタが全語の 10% 以上、または無関係な 5 グループ以上を横断したら暴走とみなす。
const RUNAWAY_RATIO: f64 = 0.10;
const RUNAWAY_GROUPS: usize = 5;

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
        Some("qwen3") => (UserModel::Qwen3Embedding0_6B.into(), "qwen3-embedding-0.6b"),
        Some("qwen3-4b") => (UserModel::Qwen3Embedding4B.into(), "qwen3-embedding-4b"),
        Some("qwen3-8b") => (UserModel::Qwen3Embedding8B.into(), "qwen3-embedding-8b"),
        #[cfg(feature = "onnx")]
        Some("large") => (EmbeddingModel::MultilingualE5Large.into(), "e5-large"),
        #[cfg(feature = "onnx")]
        Some("base") => (EmbeddingModel::MultilingualE5Base.into(), "e5-base"),
        #[cfg(feature = "onnx")]
        Some("small") => (EmbeddingModel::MultilingualE5Small.into(), "e5-small"),
        #[cfg(feature = "onnx")]
        Some("mpnet") => (EmbeddingModel::ParaphraseMLMpnetBaseV2.into(), "mpnet"),
        #[cfg(feature = "onnx")]
        Some("paraphrase") => (
            EmbeddingModel::ParaphraseMLMiniLML12V2.into(),
            "paraphrase-MiniLM-L12",
        ),
        // 既定はライブラリの既定モデル
        _ => (UserModel::BgeM3.into(), "bge-m3"),
    };

    // 用語集 + ディストラクタを連結したストレス用語集。
    let glossary = default_glossary().extended(&default_distractors());
    let total = glossary.term_count();
    let real_terms = default_glossary().term_count();
    let distractors = default_distractors().term_count();

    let n = Narashi::with_options(Options::new().with_model(model))?;

    // 0〜100 を 1 刻み
    let thresholds: Vec<f32> = (0..=100).map(|t| t as f32).collect();
    let rows = sweep(&n, &glossary, &thresholds)?;

    println!("== robustness (連鎖暴走) sweep: {label} ==");
    println!(
        "総語数 {total} (用語集 {real_terms} + ディストラクタ {distractors})  ハブ語と無関係タグで暴走を再現"
    );
    println!("閾値 | クラスタF1 (   P  /   R  ) | 誤統合 | 最大クラスタ率 | 巻込グループ数");

    // 暴走オンセット = 暴走判定を満たした「最も高い」閾値(これより上で安全)。
    let mut runaway_top: Option<f32> = None;
    // clusterF1 のピーク(従来の比較軸)。
    let mut peak = (0.0_f64, 0.0_f32);
    for r in &rows {
        if r.cluster_f1 > peak.0 {
            peak = (r.cluster_f1, r.threshold);
        }
        let runaway =
            r.largest_cluster_ratio >= RUNAWAY_RATIO || r.groups_in_largest >= RUNAWAY_GROUPS;
        if runaway {
            runaway_top = Some(runaway_top.map_or(r.threshold, |t| t.max(r.threshold)));
        }
        let mark = if runaway { " <= 暴走" } else { "" };
        println!(
            "{:>4.0} |   {:.3}   ( {:.3} / {:.3} ) | {:>5} | {:>10.1}% | {:>6}{}",
            r.threshold,
            r.cluster_f1,
            r.cluster_precision,
            r.cluster_recall,
            r.false_merges,
            r.largest_cluster_ratio * 100.0,
            r.groups_in_largest,
            mark
        );
    }

    println!("---");
    println!("clusterF1 ピーク = {:.3} @ 閾値 {:.0}", peak.0, peak.1);
    match runaway_top {
        Some(t) => {
            println!(
                "連鎖暴走は 閾値 {t:.0} 以下で発生(最大クラスタ率 >= {:.0}% または 巻込グループ >= {})。",
                RUNAWAY_RATIO * 100.0,
                RUNAWAY_GROUPS
            );
            println!("→ 運用閾値は暴走オンセット({t:.0})より安全マージンを取った上側で選ぶこと。");
            if peak.1 <= t {
                println!(
                    "⚠ clusterF1 ピーク閾値 {:.0} は暴走域内。ピークに合わせると実データで崩壊する。",
                    peak.1
                );
            }
        }
        None => println!(
            "スイープ範囲では連鎖暴走を検知せず(最大クラスタ率 < {:.0}% を維持)。",
            RUNAWAY_RATIO * 100.0
        ),
    }
    Ok(())
}
