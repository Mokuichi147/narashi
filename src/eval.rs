//! モデル評価のための共通ロジックと、用語集ベースのベンチマーク。
//!
//! 埋め込みモデルを差し替えるたびに同じ「物差し」で精度・速度を比較できるよう、
//! 評価用の用語集 (同義語グループ) と共通指標をまとめています。
//!
//! ```no_run
//! use narashi::{Narashi, DEFAULT_THRESHOLD};
//! use narashi::eval::{default_glossary, evaluate};
//!
//! let glossary = default_glossary();
//! let n = Narashi::new()?;
//! let report = evaluate(&n, &glossary, DEFAULT_THRESHOLD, "default")?;
//! println!("{report}");
//! # anyhow::Ok(())
//! ```

use crate::{Narashi, dot};
use anyhow::Result;
use std::fmt;
use std::time::Instant;

/// 評価用の用語集。各要素が 1 つの同義語グループ。
///
/// 同じグループ内のペアは統合されるべき (正例)、異なるグループ間のペアは
/// 統合されるべきでない (負例) として扱われます。メンバが 1 つだけのグループは
/// 「単独語 (ディストラクタ)」で、他のどれとも統合されません。
#[derive(Debug, Clone)]
pub struct Glossary {
    /// 同義語グループの一覧
    pub groups: Vec<Vec<String>>,
}

impl Glossary {
    /// 行ベースのテキストをパースする。
    ///
    /// - 1 行 = 1 グループ、メンバはカンマ (`,`) 区切り
    /// - `#` で始まる行と空行は無視
    /// - 各メンバは前後の空白をトリム
    pub fn parse(text: &str) -> Self {
        let groups = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| {
                line.split(',')
                    .map(str::trim)
                    .filter(|m| !m.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .filter(|g| !g.is_empty())
            .collect();
        Self { groups }
    }

    /// 全用語をフラットに並べ、各用語の所属グループ ID を返す。
    pub fn flatten(&self) -> (Vec<String>, Vec<usize>) {
        let mut terms = Vec::new();
        let mut group_ids = Vec::new();
        for (gid, group) in self.groups.iter().enumerate() {
            for term in group {
                terms.push(term.clone());
                group_ids.push(gid);
            }
        }
        (terms, group_ids)
    }

    /// 含まれる用語の総数
    pub fn term_count(&self) -> usize {
        self.groups.iter().map(Vec::len).sum()
    }
}

/// クレートに同梱された既定の評価用用語集を返す。
pub fn default_glossary() -> Glossary {
    Glossary::parse(include_str!("../tests/data/glossary.txt"))
}

/// 固定閾値での分類精度 (ペア単位、推移閉包なし)
#[derive(Debug, Clone)]
pub struct Classification {
    /// 判定に用いた閾値
    pub threshold: f32,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
    pub accuracy: f64,
    pub true_positive: usize,
    pub false_positive: usize,
    pub true_negative: usize,
    pub false_negative: usize,
}

/// 閾値走査による最適点と、正例・負例スコアの分離度
#[derive(Debug, Clone)]
pub struct ThresholdScan {
    /// F1 を最大化する閾値
    pub best_threshold: f32,
    /// そのときの F1
    pub best_f1: f64,
    /// 正例ペアの最小スコア
    pub min_positive: f32,
    /// 負例ペアの最大スコア
    pub max_negative: f32,
    /// 分離マージン (`min_positive - max_negative`、正なら完全分離可能)
    pub margin: f32,
}

/// `normalize` の出力 (推移閉包あり) と正解グループの一致度
#[derive(Debug, Clone)]
pub struct Clustering {
    /// クラスタリングに用いた閾値
    pub threshold: f32,
    pub pair_precision: f64,
    pub pair_recall: f64,
    pub pair_f1: f64,
    /// 正解グループと完全一致したクラスタ数
    pub exact_group_match: usize,
    /// 正解グループ数
    pub expected_groups: usize,
    /// 予測されたクラスタ数
    pub predicted_clusters: usize,
}

/// 速度指標
#[derive(Debug, Clone)]
pub struct Speed {
    /// モデルのロード時間 (ms)
    pub load_ms: f64,
    /// 全用語の埋め込みにかかった時間 (ms)
    pub embed_ms: f64,
    /// 1 用語あたりの平均埋め込み時間 (ms)
    pub per_text_ms: f64,
    /// 埋め込んだ用語数
    pub term_count: usize,
}

/// 1 モデル分のベンチマーク結果 (共通スペック)
#[derive(Debug, Clone)]
pub struct Benchmark {
    /// 対象モデルのラベル
    pub model: String,
    pub classification: Classification,
    pub scan: ThresholdScan,
    pub clustering: Clustering,
    pub speed: Speed,
}

/// 2 つのカウントから precision/recall/f1 を計算する。
fn prf(tp: usize, fp: usize, fn_: usize) -> (f64, f64, f64) {
    let precision = if tp + fp == 0 {
        0.0
    } else {
        tp as f64 / (tp + fp) as f64
    };
    let recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f64 / (tp + fn_) as f64
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    (precision, recall, f1)
}

/// 用語集に対してモデルを評価し、共通スペックを算出する。
///
/// `load_ms` を別途渡す代わりにモデルロードもここで計測したい場合は
/// [`evaluate`] を、ロード時間を呼び出し側で計測したい場合は
/// [`evaluate_with_load`] を使ってください。
pub fn evaluate(n: &Narashi, glossary: &Glossary, threshold: f32, model: &str) -> Result<Benchmark> {
    evaluate_with_load(n, glossary, threshold, model, 0.0)
}

/// [`evaluate`] と同じだが、別途計測したモデルロード時間 (ms) を受け取る。
pub fn evaluate_with_load(
    n: &Narashi,
    glossary: &Glossary,
    threshold: f32,
    model: &str,
    load_ms: f64,
) -> Result<Benchmark> {
    let (terms, group_ids) = glossary.flatten();
    let num = terms.len();

    // --- 埋め込み (速度計測も兼ねる) ---
    let t0 = Instant::now();
    let embeddings = n.embed_normalized(&terms)?;
    let embed_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // --- 全ペアのスコアとラベル ---
    let mut scored: Vec<(f32, bool)> = Vec::with_capacity(num * num / 2);
    let mut min_positive = f32::INFINITY;
    let mut max_negative = f32::NEG_INFINITY;
    for i in 0..num {
        for j in (i + 1)..num {
            let s = n.score(dot(&embeddings[i], &embeddings[j]));
            let positive = group_ids[i] == group_ids[j];
            if positive {
                min_positive = min_positive.min(s);
            } else {
                max_negative = max_negative.max(s);
            }
            scored.push((s, positive));
        }
    }

    // --- 固定閾値での分類精度 ---
    let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    for &(s, positive) in &scored {
        match (positive, s >= threshold) {
            (true, true) => tp += 1,
            (false, true) => fp += 1,
            (false, false) => tn += 1,
            (true, false) => fn_ += 1,
        }
    }
    let (precision, recall, f1) = prf(tp, fp, fn_);
    let total = scored.len().max(1);
    let classification = Classification {
        threshold,
        precision,
        recall,
        f1,
        accuracy: (tp + tn) as f64 / total as f64,
        true_positive: tp,
        false_positive: fp,
        true_negative: tn,
        false_negative: fn_,
    };

    // --- 閾値走査 (0.0〜100.0 を 0.5 刻み) で F1 最大点を探索 ---
    let mut best_threshold = threshold;
    let mut best_f1 = 0.0;
    let mut cand = 0.0;
    while cand <= 100.0 {
        let (mut btp, mut bfp, mut bfn) = (0usize, 0usize, 0usize);
        for &(s, positive) in &scored {
            match (positive, s >= cand) {
                (true, true) => btp += 1,
                (false, true) => bfp += 1,
                (true, false) => bfn += 1,
                (false, false) => {}
            }
        }
        let (_, _, cf1) = prf(btp, bfp, bfn);
        if cf1 > best_f1 {
            best_f1 = cf1;
            best_threshold = cand;
        }
        cand += 0.5;
    }
    let scan = ThresholdScan {
        best_threshold,
        best_f1,
        min_positive: if min_positive.is_finite() {
            min_positive
        } else {
            0.0
        },
        max_negative: if max_negative.is_finite() {
            max_negative
        } else {
            0.0
        },
        margin: min_positive - max_negative,
    };

    // --- クラスタ一致度 (normalize の推移閉包込みの結果と正解グループを比較) ---
    let groups = n.normalize(&terms, threshold)?;
    // 各用語 -> 予測クラスタ ID
    let mut cluster_of: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (cid, g) in groups.iter().enumerate() {
        for m in &g.members {
            cluster_of.insert(m.as_str(), cid);
        }
    }
    let (mut ctp, mut cfp, mut cfn) = (0usize, 0usize, 0usize);
    for i in 0..num {
        for j in (i + 1)..num {
            let same_pred = cluster_of.get(terms[i].as_str()) == cluster_of.get(terms[j].as_str());
            let same_true = group_ids[i] == group_ids[j];
            match (same_true, same_pred) {
                (true, true) => ctp += 1,
                (false, true) => cfp += 1,
                (true, false) => cfn += 1,
                (false, false) => {}
            }
        }
    }
    let (cp, cr, cf1) = prf(ctp, cfp, cfn);
    // 正解グループと完全一致した予測クラスタの数
    let predicted_sets: Vec<std::collections::BTreeSet<&str>> = groups
        .iter()
        .map(|g| g.members.iter().map(String::as_str).collect())
        .collect();
    let exact_group_match = glossary
        .groups
        .iter()
        .filter(|expected| {
            let want: std::collections::BTreeSet<&str> =
                expected.iter().map(String::as_str).collect();
            predicted_sets.contains(&want)
        })
        .count();
    let clustering = Clustering {
        threshold,
        pair_precision: cp,
        pair_recall: cr,
        pair_f1: cf1,
        exact_group_match,
        expected_groups: glossary.groups.len(),
        predicted_clusters: groups.len(),
    };

    Ok(Benchmark {
        model: model.to_string(),
        classification,
        scan,
        clustering,
        speed: Speed {
            load_ms,
            embed_ms,
            per_text_ms: if num == 0 {
                0.0
            } else {
                embed_ms / num as f64
            },
            term_count: num,
        },
    })
}

impl fmt::Display for Benchmark {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.classification;
        let s = &self.scan;
        let cl = &self.clustering;
        let sp = &self.speed;
        writeln!(f, "===== narashi ベンチマーク: {} =====", self.model)?;
        writeln!(
            f,
            "用語数: {}  ペア数: {}",
            sp.term_count,
            sp.term_count * sp.term_count.saturating_sub(1) / 2
        )?;
        writeln!(f, "-- 分類精度 (閾値 {:.1}) --", c.threshold)?;
        writeln!(
            f,
            "  Precision={:.3}  Recall={:.3}  F1={:.3}  Accuracy={:.3}",
            c.precision, c.recall, c.f1, c.accuracy
        )?;
        writeln!(
            f,
            "  TP={} FP={} TN={} FN={}",
            c.true_positive, c.false_positive, c.true_negative, c.false_negative
        )?;
        writeln!(f, "-- 最適閾値・分離マージン --")?;
        writeln!(
            f,
            "  best_threshold={:.1}  best_F1={:.3}",
            s.best_threshold, s.best_f1
        )?;
        writeln!(
            f,
            "  正例min={:.1}  負例max={:.1}  margin={:.1}",
            s.min_positive, s.max_negative, s.margin
        )?;
        writeln!(f, "-- クラスタ一致度 (normalize, 閾値 {:.1}) --", cl.threshold)?;
        writeln!(
            f,
            "  pairF1={:.3} (P={:.3} R={:.3})  完全一致グループ {}/{}  クラスタ数={}",
            cl.pair_f1,
            cl.pair_precision,
            cl.pair_recall,
            cl.exact_group_match,
            cl.expected_groups,
            cl.predicted_clusters
        )?;
        writeln!(f, "-- 速度 --")?;
        write!(
            f,
            "  load={:.0}ms  embed={:.0}ms  per_text={:.2}ms",
            sp.load_ms, sp.embed_ms, sp.per_text_ms
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_skips_comments_and_blanks() {
        let g = Glossary::parse("# c\n\n猫, ネコ ,ねこ\n犬,イヌ\n\n自動車\n");
        assert_eq!(g.groups.len(), 3);
        assert_eq!(g.groups[0], vec!["猫", "ネコ", "ねこ"]);
        assert_eq!(g.groups[2], vec!["自動車"]);
        assert_eq!(g.term_count(), 6);
    }

    #[test]
    fn flatten_assigns_group_ids() {
        let g = Glossary::parse("a,b\nc");
        let (terms, ids) = g.flatten();
        assert_eq!(terms, vec!["a", "b", "c"]);
        assert_eq!(ids, vec![0, 0, 1]);
    }

    #[test]
    fn default_glossary_is_nonempty() {
        let g = default_glossary();
        assert!(g.groups.len() >= 5, "groups: {}", g.groups.len());
        assert!(g.term_count() >= 10);
    }

    #[test]
    fn prf_basic() {
        let (p, r, f1) = prf(8, 2, 2);
        assert!((p - 0.8).abs() < 1e-9);
        assert!((r - 0.8).abs() < 1e-9);
        assert!((f1 - 0.8).abs() < 1e-9);
        assert_eq!(prf(0, 0, 0), (0.0, 0.0, 0.0));
    }
}
