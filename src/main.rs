use anyhow::Result;
use clap::{Parser, ValueEnum};
use narashi::{EmbeddingModel, Group, Narashi, Options};
use std::path::PathBuf;

/// CLI から選択できる埋め込みモデル
#[derive(Copy, Clone, Debug, ValueEnum)]
enum ModelArg {
    /// paraphrase-multilingual-MiniLM-L12-v2 (既定・対称類似度向け)
    Paraphrase,
    /// paraphrase-multilingual-MiniLM-L12-v2 量子化版 (高速)
    ParaphraseQ,
    /// 多言語 E5 small (検索向けチューニング・同規模)
    Small,
    /// 多言語 E5 base (高次元・低速)
    Base,
}

impl From<ModelArg> for EmbeddingModel {
    fn from(m: ModelArg) -> Self {
        match m {
            ModelArg::Small => EmbeddingModel::MultilingualE5Small,
            ModelArg::Base => EmbeddingModel::MultilingualE5Base,
            ModelArg::Paraphrase => EmbeddingModel::ParaphraseMLMiniLML12V2,
            ModelArg::ParaphraseQ => EmbeddingModel::ParaphraseMLMiniLML12V2Q,
        }
    }
}

#[derive(Parser)]
#[command(name = "narashi", about = "表記ゆれ解消ツール")]
struct Cli {
    /// 類似度の閾値 (0-100)
    #[arg(short, long, default_value_t = 70.0)]
    threshold: f32,

    /// 使用する埋め込みモデル
    #[arg(long, value_enum, default_value_t = ModelArg::Paraphrase)]
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
    let mut opts = Options::new().with_model(cli.model.into());
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
