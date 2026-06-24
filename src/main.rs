use anyhow::Result;
use clap::{Parser, ValueEnum};
#[cfg(feature = "onnx")]
use narashi::EmbeddingModel;
use narashi::{DEFAULT_THRESHOLD, Group, Model, Narashi, Options, UserModel};
use std::path::PathBuf;

/// CLI から選択できる埋め込みモデル
///
/// ONNX バックエンドのモデルは `onnx` 機能、Candle バックエンドのモデルは `candle` 機能が
/// 有効なビルドでのみ選択できる(既定ビルドは両方有効)。
#[derive(Copy, Clone, Debug, ValueEnum)]
enum ModelArg {
    /// bge-m3 (既定・ONNX勢でclusterF1最高0.699・誤統合も最小7件・1024次元・約1.06GB・約3倍低速)
    #[cfg(feature = "onnx")]
    BgeM3,
    /// gte-multilingual-base (高適合率・CJKに強い・速度重視の代替・768次元・約1.2GB)
    #[cfg(feature = "onnx")]
    Gte,
    /// granite-278m-multilingual (clusterF1高め0.682だが誤統合が多め28件・日本語明示学習・768次元・約1.1GB)
    #[cfg(feature = "onnx")]
    Granite,
    /// distiluse-multilingual-v2 (軽量代替・高適合率・約0.54GB)
    #[cfg(feature = "onnx")]
    Distiluse,
    /// 多言語 E5 small (高適合率かつ最速級・軽量 約0.45GB)
    #[cfg(feature = "onnx")]
    Small,
    /// 多言語 E5 large (E5系の上限・約8倍低速)
    #[cfg(feature = "onnx")]
    Large,
    /// 多言語 E5 base (small に劣後・非推奨)
    #[cfg(feature = "onnx")]
    Base,
    /// paraphrase-multilingual-MiniLM-L12-v2 (高再現率・要 高め閾値)
    #[cfg(feature = "onnx")]
    Paraphrase,
    /// paraphrase-multilingual-mpnet-base-v2 (再現率最優先・要 高め閾値)
    #[cfg(feature = "onnx")]
    Mpnet,
    /// paraphrase-multilingual-MiniLM-L12-v2 量子化版 (高速)
    #[cfg(feature = "onnx")]
    ParaphraseQ,
    /// multilingual-e5-large-instruct (Candle・ONNX非依存環境向け・clusterF1 0.645で誤統合最多75件・低速・1024次元)
    #[cfg(feature = "candle")]
    E5Instruct,
    /// Qwen3-Embedding-0.6B (Candle・clusterF1 0.764でbge-m3超だが低速・1024次元・精度最優先/ONNX非依存向け)
    #[cfg(feature = "candle")]
    Qwen3,
    /// Qwen3-Embedding-4B (Candle・clusterF1最高0.956・誤統合7件だが超低速≈3.9秒/語・約8GB RAM・GPU/バッチ向け)
    #[cfg(feature = "candle")]
    #[value(name = "qwen3-4b")]
    Qwen34b,
    /// Qwen3-Embedding-8B (Candle・eval用・約16GB RAM 必須・さらに低速)
    #[cfg(feature = "candle")]
    #[value(name = "qwen3-8b")]
    Qwen38b,
}

/// 既定モデル。ONNX が有効なら bge-m3、Candle のみなら Qwen3-Embedding
/// (Candle 勢では clusterF1 最高 0.764・誤統合少で最良。e5-instruct より精度・安全性が高い)。
#[cfg(feature = "onnx")]
const DEFAULT_MODEL_ARG: ModelArg = ModelArg::BgeM3;
#[cfg(all(not(feature = "onnx"), feature = "candle"))]
const DEFAULT_MODEL_ARG: ModelArg = ModelArg::Qwen3;

impl From<ModelArg> for Model {
    fn from(m: ModelArg) -> Self {
        match m {
            #[cfg(feature = "onnx")]
            ModelArg::Small => EmbeddingModel::MultilingualE5Small.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Base => EmbeddingModel::MultilingualE5Base.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Large => EmbeddingModel::MultilingualE5Large.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Paraphrase => EmbeddingModel::ParaphraseMLMiniLML12V2.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Mpnet => EmbeddingModel::ParaphraseMLMpnetBaseV2.into(),
            #[cfg(feature = "onnx")]
            ModelArg::ParaphraseQ => EmbeddingModel::ParaphraseMLMiniLML12V2Q.into(),
            #[cfg(feature = "onnx")]
            ModelArg::BgeM3 => UserModel::BgeM3.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Gte => UserModel::GteMultilingualBase.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Granite => UserModel::GraniteMultilingual278m.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Distiluse => UserModel::DistiluseMultilingualV2.into(),
            #[cfg(feature = "candle")]
            ModelArg::E5Instruct => UserModel::E5LargeInstruct.into(),
            #[cfg(feature = "candle")]
            ModelArg::Qwen3 => UserModel::Qwen3Embedding0_6B.into(),
            #[cfg(feature = "candle")]
            ModelArg::Qwen34b => UserModel::Qwen3Embedding4B.into(),
            #[cfg(feature = "candle")]
            ModelArg::Qwen38b => UserModel::Qwen3Embedding8B.into(),
        }
    }
}

#[derive(Parser)]
#[command(name = "narashi", about = "表記ゆれ解消ツール")]
struct Cli {
    /// 類似度の閾値 (0-100)
    #[arg(short, long, default_value_t = DEFAULT_THRESHOLD)]
    threshold: f32,

    /// 使用する埋め込みモデル
    #[arg(long, value_enum, default_value_t = DEFAULT_MODEL_ARG)]
    model: ModelArg,

    /// モデルキャッシュの保存先 (既定: OSのTEMPフォルダ下)
    #[arg(long, env = "NARASHI_CACHE_DIR")]
    cache_dir: Option<PathBuf>,

    /// 比較するテキスト (2つ以上)
    #[arg(required = true, num_args = 2..)]
    texts: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut opts = Options::new().with_model(cli.model);
    if let Some(dir) = cli.cache_dir {
        opts = opts.with_cache_dir(dir);
    }
    let n = Narashi::with_options(opts)?;

    if cli.texts.len() == 2 {
        let a = &cli.texts[0];
        let b = &cli.texts[1];
        let sim = n.similarity(a, b)?;
        println!("{a} ⇔ {b}: {sim:.1}");
        if sim >= cli.threshold {
            let groups = n.normalize(&cli.texts, cli.threshold)?;
            if let Some(g) = groups.iter().find(|g| g.members.len() > 1) {
                println!("→ 「{}」に統合", g.canonical);
            }
        } else {
            println!("(閾値 {:.1} 未満のため統合なし)", cli.threshold);
        }
    } else {
        let groups = n.normalize(&cli.texts, cli.threshold)?;
        for g in &groups {
            print_group(g);
        }
    }
    Ok(())
}

fn print_group(g: &Group) {
    if g.members.len() == 1 {
        println!("[単独] {}", g.canonical);
    } else {
        let others: Vec<&str> = g
            .members
            .iter()
            .filter(|m| m.as_str() != g.canonical)
            .map(|s| s.as_str())
            .collect();
        println!("[統合] {} ← {}", g.canonical, others.join(", "));
    }
}
