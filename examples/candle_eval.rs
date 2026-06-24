//! 一時検証(Candle 専用): ONNX Runtime を介さず動く Candle バックエンドのモデルだけを
//! 評価する。`benchmark` / `fine_sweep` は `onnx` フィーチャ必須でビルドに ORT バイナリが
//! 要るため、ORT を取得できない環境では本 example を candle 単独でビルドして回す。
//!
//! ```sh
//! cargo run --no-default-features --features candle --example candle_eval -- qwen3
//! cargo run --no-default-features --features candle --example candle_eval -- qwen3-4b
//! ```
use anyhow::Result;
use narashi::eval::{default_glossary, evaluate_with_load, sweep};
use narashi::{DEFAULT_THRESHOLD, Model, Narashi, Options, UserModel};
use std::time::Instant;

fn main() -> Result<()> {
    let arg = std::env::args().nth(1);
    let (model, label): (Model, &str) = match arg.as_deref() {
        Some("e5-instruct") => (UserModel::E5LargeInstruct.into(), "e5-large-instruct"),
        Some("qwen3-4b") => (UserModel::Qwen3Embedding4B.into(), "qwen3-embedding-4b"),
        Some("qwen3-8b") => (UserModel::Qwen3Embedding8B.into(), "qwen3-embedding-8b"),
        _ => (UserModel::Qwen3Embedding0_6B.into(), "qwen3-embedding-0.6b"),
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

    // 1 刻み(40〜95)の真ピークは追加の埋め込みパスが要る(Candle デコーダは低速)。
    // 第 3 引数 `fine` を渡したときだけ実行する。
    if std::env::args().nth(3).as_deref() == Some("fine") {
        let thresholds: Vec<f32> = (40..=95).map(|t| t as f32).collect();
        let rows = sweep(&n, &glossary, &thresholds)?;
        let mut peak = (0.0_f64, 0.0_f32, 0.0_f64, 0.0_f64, 0usize);
        for r in &rows {
            if r.cluster_f1 > peak.0 {
                peak = (
                    r.cluster_f1,
                    r.threshold,
                    r.cluster_precision,
                    r.cluster_recall,
                    r.false_merges,
                );
            }
        }
        println!(
            "\n== 1 刻み真ピーク == clusterF1={:.3} @閾値{:.0} (P={:.3} R={:.3} 誤統合={})",
            peak.0, peak.1, peak.2, peak.3, peak.4
        );
    }
    Ok(())
}
