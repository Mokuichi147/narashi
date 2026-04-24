use anyhow::Result;
use clap::Parser;
use narashi::{Group, Narashi, Options};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "narashi", about = "表記ゆれ解消ツール")]
struct Cli {
    /// 類似度の閾値 (0-100)
    #[arg(short, long, default_value_t = 95.0)]
    threshold: f32,

    /// モデルキャッシュの保存先 (既定: OSのTEMPフォルダ下)
    #[arg(long, env = "NARASHI_CACHE_DIR")]
    cache_dir: Option<PathBuf>,

    /// 比較するテキスト (2つ以上)
    #[arg(required = true, num_args = 2..)]
    texts: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut opts = Options::new();
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
