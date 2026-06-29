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

    /// 別の用語集のグループを連結した新しい用語集を返す。
    ///
    /// `flatten` はグループ順に連番でグループ ID を振り直すため、連結後も
    /// 「同グループ内=正例・異グループ間=負例」の不変条件は保たれる。ただし両者で
    /// 同じ語が出現するとラベルが壊れる(本来別グループの語が同一視される)ため、
    /// 連結する用語集どうしは語が重複しないこと(テストで保証する)。
    pub fn extended(mut self, other: &Glossary) -> Self {
        self.groups.extend(other.groups.iter().cloned());
        self
    }
}

/// クレートに同梱された既定の評価用用語集を返す。
pub fn default_glossary() -> Glossary {
    Glossary::parse(include_str!("../tests/data/glossary.txt"))
}

/// 連鎖暴走(over-merge)検証用のディストラクタ用語集を返す。
///
/// 全行が単独語(どれとも統合されない負例)。実データのタグ群に多い「ハブ語」
/// (`1`/`2B`/`3D` 等の汎用短トークンや単一文字)と、互いに非同義の無関係タグを
/// 多数含む。[`default_glossary`] と [`Glossary::extended`] で連結し、単連結
/// クラスタリングが巨大ゴミクラスタへ崩壊しないか(`largest_cluster_ratio` /
/// `groups_in_largest`)を測るためのストレスデータ。
pub fn default_distractors() -> Glossary {
    Glossary::parse(include_str!("../tests/data/distractors.txt"))
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
    /// 誤統合ペア数 (異グループの語を同一クラスタに入れてしまった数 = データ破壊)
    pub false_merges: usize,
    /// 誤統合の具体例 (異グループなのに統合された語ペア。表示用に一部のみ保持)
    pub false_merge_examples: Vec<(String, String)>,
    /// 最大予測クラスタの語数 (連鎖暴走の規模)
    pub largest_cluster_size: usize,
    /// 最大予測クラスタが全語に占める割合 (`largest_cluster_size / 総語数`)。
    /// 1.0 に近いほど「1 つの巨大ゴミクラスタが全体を飲み込んだ」= 連鎖暴走。
    pub largest_cluster_ratio: f64,
    /// 最大予測クラスタに巻き込まれた相異なる正解グループ数。
    /// 健全なら 1 (1 つの正解グループ = 1 クラスタ)、暴走すると多数の無関係グループを
    /// 1 クラスタが横断する。単連結クラスタリングの連鎖暴走を直接捉える核心指標。
    pub groups_in_largest: usize,
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

/// 閾値スイープの 1 行 (ある閾値での分類 F1 とクラスタ F1)
#[derive(Debug, Clone)]
pub struct SweepRow {
    /// この行の閾値
    pub threshold: f32,
    /// 分類 F1 (ペア単位、推移閉包なし)
    pub class_f1: f64,
    /// クラスタ F1 (`normalize` 相当の推移閉包込み)
    pub cluster_f1: f64,
    /// クラスタの適合率 (誤統合の少なさ)
    pub cluster_precision: f64,
    /// クラスタの再現率 (取りこぼしの少なさ)
    pub cluster_recall: f64,
    /// 誤統合ペア数 (推移閉包込みで同一クラスタにされた異グループ間ペア = データ破壊)
    pub false_merges: usize,
    /// 最大クラスタが全語に占める割合 (連鎖暴走の度合い。1.0 に近いほど暴走)
    pub largest_cluster_ratio: f64,
    /// 最大クラスタに巻き込まれた相異なる正解グループ数 (健全なら 1、暴走で多数)
    pub groups_in_largest: usize,
}

/// スコア付きの語ペア (表示用)。`(語 a, 語 b, スコア)`
pub type NamedPair = (String, String, f32);

/// 1 モデル分のベンチマーク結果 (共通スペック)
#[derive(Debug, Clone)]
pub struct Benchmark {
    /// 対象モデルのラベル
    pub model: String,
    pub classification: Classification,
    pub scan: ThresholdScan,
    pub clustering: Clustering,
    /// 複数閾値での挙動 (既定閾値以外も含む)
    pub sweep: Vec<SweepRow>,
    /// **最難負例**: 異グループなのにスコアが高い(= 最も誤統合しやすい)負例ペアの上位。
    /// 閾値非依存で、上位モデルほどここのスコアが低い。飽和したベンチマークでも
    /// 「適合率の境界」を直接見分けられる比較軸。スコア降順。
    pub hardest_negatives: Vec<NamedPair>,
    /// **最難正例**: 同グループなのにスコアが低い(= 最も取りこぼしやすい)正例ペアの下位。
    /// 閾値非依存で、上位モデルほどここのスコアが高い。「再現率の境界」を直接見分ける。
    /// スコア昇順。
    pub hardest_positives: Vec<NamedPair>,
    pub speed: Speed,
}

/// 最難ペアとして保持・表示する件数 (負例・正例それぞれ)
pub const HARDEST_PAIRS: usize = 10;

/// 既定でスイープする閾値 (0〜100 を 5 刻み)
pub const SWEEP_THRESHOLDS: &[f32] = &[
    0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0,
    80.0, 85.0, 90.0, 95.0, 100.0,
];

/// 1 ペア分のスコア情報 `(添字 i, 添字 j, スコア, 正例フラグ)`
///
/// 添字を保持することで、同じ埋め込みから任意の閾値でのクラスタリングを再計算できる。
type ScoredPair = (usize, usize, f32, bool);

/// 全ペアのスコア (添字・スコア・正例フラグ) を 1 度だけ計算する。
///
/// 同じ埋め込みから任意の閾値でのクラスタリングを再計算できるよう、添字付きで返す。
/// 併せて各語の正解グループ ID を返す (トポロジー指標の算出に使う)。
fn scored_pairs(n: &Narashi, glossary: &Glossary) -> Result<(usize, Vec<ScoredPair>, Vec<usize>)> {
    let (terms, group_ids) = glossary.flatten();
    let num = terms.len();
    let embeddings = n.embed_normalized(&terms)?;
    let mut scored: Vec<(usize, usize, f32, bool)> = Vec::with_capacity(num * num / 2);
    for i in 0..num {
        for j in (i + 1)..num {
            let s = n.score(dot(&embeddings[i], &embeddings[j]));
            scored.push((i, j, s, group_ids[i] == group_ids[j]));
        }
    }
    Ok((num, scored, group_ids))
}

/// 連結成分の根の配列から、最大クラスタの規模と巻き込んだ正解グループ数を求める。
///
/// 返り値 `(largest_cluster_size, groups_in_largest)`。`groups_in_largest` は最大クラスタに
/// 属する語の正解グループ ID の異なり数で、単連結クラスタリングの連鎖暴走を直接捉える
/// (健全なら 1、暴走すると無関係な多数グループを 1 クラスタが横断する)。
fn largest_cluster_topology(roots: &[usize], group_ids: &[usize]) -> (usize, usize) {
    // 各根のメンバ添字を集める。
    let mut members: std::collections::HashMap<usize, Vec<usize>> =
        std::collections::HashMap::new();
    for (i, &r) in roots.iter().enumerate() {
        members.entry(r).or_default().push(i);
    }
    members
        .values()
        .map(|idxs| {
            let groups: std::collections::HashSet<usize> =
                idxs.iter().map(|&i| group_ids[i]).collect();
            (idxs.len(), groups.len())
        })
        // 最大クラスタは語数で選ぶ (同数なら巻込グループ数が多い方)。
        .max_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)))
        .unwrap_or((0, 0))
}

/// 添字付きスコアから、各閾値の分類 F1 とクラスタ F1 (推移閉包込み) を算出する。
fn sweep_from_scored(
    num: usize,
    scored: &[ScoredPair],
    group_ids: &[usize],
    thresholds: &[f32],
) -> Vec<SweepRow> {
    thresholds
        .iter()
        .map(|&t| {
            let (mut stp, mut sfp, mut sfn) = (0usize, 0usize, 0usize);
            for &(_, _, s, positive) in scored {
                match (positive, s >= t) {
                    (true, true) => stp += 1,
                    (false, true) => sfp += 1,
                    (true, false) => sfn += 1,
                    (false, false) => {}
                }
            }
            let (_, _, class_f1) = prf(stp, sfp, sfn);

            let roots = connected_roots(num, scored, t);
            let (mut ctp, mut cfp, mut cfn) = (0usize, 0usize, 0usize);
            for &(i, j, _, positive) in scored {
                match (positive, roots[i] == roots[j]) {
                    (true, true) => ctp += 1,
                    (false, true) => cfp += 1,
                    (true, false) => cfn += 1,
                    (false, false) => {}
                }
            }
            let (cp, cr, cluster_f1) = prf(ctp, cfp, cfn);
            let (largest, groups_in_largest) = largest_cluster_topology(&roots, group_ids);
            SweepRow {
                threshold: t,
                class_f1,
                cluster_f1,
                cluster_precision: cp,
                cluster_recall: cr,
                false_merges: cfp,
                largest_cluster_ratio: if num == 0 {
                    0.0
                } else {
                    largest as f64 / num as f64
                },
                groups_in_largest,
            }
        })
        .collect()
}

/// 任意の閾値群でスイープし、各閾値の分類 F1・クラスタ F1・P・R を返す。
///
/// 埋め込みを 1 度だけ計算して使い回すため、細かい刻みのスイープでも高速。
/// 既定閾値以外でのモデル挙動 (ピークの位置・鋭さ) を検証するのに使う。
pub fn sweep(n: &Narashi, glossary: &Glossary, thresholds: &[f32]) -> Result<Vec<SweepRow>> {
    let (num, scored, group_ids) = scored_pairs(n, glossary)?;
    Ok(sweep_from_scored(num, &scored, &group_ids, thresholds))
}

/// 全ペアを `(語 a, 語 b, 正例フラグ, スコア)` で返す(難易度分析・ダンプ用)。
///
/// 「全モデルが正解してしまう簡単すぎるペア(= スコアの底上げ要因)」を横断的に洗い出す
/// ための生データ。埋め込みは 1 度だけ計算する。
pub fn all_scored_pairs(n: &Narashi, glossary: &Glossary) -> Result<Vec<NamedPairLabeled>> {
    let (terms, _) = glossary.flatten();
    let (_, scored, _) = scored_pairs(n, glossary)?;
    Ok(scored
        .into_iter()
        .map(|(i, j, s, positive)| (terms[i].clone(), terms[j].clone(), positive, s))
        .collect())
}

/// `(語 a, 語 b, 正例フラグ, スコア)`
pub type NamedPairLabeled = (String, String, bool, f32);

/// 全要素を独立成分とし、閾値以上のペアを連結した連結成分の代表を返す。
///
/// `normalize` の閾値統合と同じ分割を、再埋め込みせず添字付きスコアから再現する
/// (クラスタの分割のみを見るので代表語選定は不要)。
fn connected_roots(num: usize, edges: &[ScoredPair], threshold: f32) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..num).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while parent[r] != r {
            r = parent[r];
        }
        // 経路圧縮
        let mut cur = x;
        while parent[cur] != r {
            let next = parent[cur];
            parent[cur] = r;
            cur = next;
        }
        r
    }
    for &(i, j, s, _) in edges {
        if s >= threshold {
            let ri = find(&mut parent, i);
            let rj = find(&mut parent, j);
            if ri != rj {
                parent[ri] = rj;
            }
        }
    }
    (0..num).map(|x| find(&mut parent, x)).collect()
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
pub fn evaluate(
    n: &Narashi,
    glossary: &Glossary,
    threshold: f32,
    model: &str,
) -> Result<Benchmark> {
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

    // --- 全ペアの (添字, スコア, ラベル) ---
    // 添字を保持することで、任意の閾値でのクラスタリング (union-find) を
    // 再埋め込みせずに算出できる (閾値スイープ用)。
    let mut scored: Vec<(usize, usize, f32, bool)> = Vec::with_capacity(num * num / 2);
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
            scored.push((i, j, s, positive));
        }
    }

    // --- 固定閾値での分類精度 ---
    let (mut tp, mut fp, mut tn, mut fn_) = (0usize, 0usize, 0usize, 0usize);
    for &(_, _, s, positive) in &scored {
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
        for &(_, _, s, positive) in &scored {
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
    // 誤統合 (異グループなのに同一クラスタ) の具体例を先頭から数件保持する。
    const MAX_FALSE_MERGE_EXAMPLES: usize = 8;
    let mut false_merge_examples: Vec<(String, String)> = Vec::new();
    for i in 0..num {
        for j in (i + 1)..num {
            let same_pred = cluster_of.get(terms[i].as_str()) == cluster_of.get(terms[j].as_str());
            let same_true = group_ids[i] == group_ids[j];
            match (same_true, same_pred) {
                (true, true) => ctp += 1,
                (false, true) => {
                    cfp += 1;
                    if false_merge_examples.len() < MAX_FALSE_MERGE_EXAMPLES {
                        false_merge_examples.push((terms[i].clone(), terms[j].clone()));
                    }
                }
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
    // --- 連鎖暴走の検知: 最大予測クラスタの規模と巻き込んだ正解グループ数 ---
    // 単連結クラスタリングはハブ語を介して無関係なクラスタを 1 つの巨大成分へ連鎖させうる。
    // ペア単位の誤統合数では見えにくいため、最大クラスタが何語・何グループを飲み込んだかを測る。
    let true_gid_of: std::collections::HashMap<&str, usize> = terms
        .iter()
        .zip(group_ids.iter())
        .map(|(t, &g)| (t.as_str(), g))
        .collect();
    let (largest_cluster_size, groups_in_largest) = groups
        .iter()
        .map(|g| {
            let gids: std::collections::HashSet<usize> = g
                .members
                .iter()
                .filter_map(|m| true_gid_of.get(m.as_str()).copied())
                .collect();
            (g.members.len(), gids.len())
        })
        .max_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)))
        .unwrap_or((0, 0));
    let clustering = Clustering {
        threshold,
        pair_precision: cp,
        pair_recall: cr,
        pair_f1: cf1,
        exact_group_match,
        expected_groups: glossary.groups.len(),
        predicted_clusters: groups.len(),
        false_merges: cfp,
        false_merge_examples,
        largest_cluster_size,
        largest_cluster_ratio: if num == 0 {
            0.0
        } else {
            largest_cluster_size as f64 / num as f64
        },
        groups_in_largest,
    };

    // --- 閾値スイープ (既定閾値以外の挙動) ---
    // 各閾値で分類 F1 とクラスタ F1 (推移閉包込み) を再埋め込みせず算出する。
    let sweep = sweep_from_scored(num, &scored, &group_ids, SWEEP_THRESHOLDS);

    // --- 最難ペア (閾値非依存の境界可視化) ---
    // 負例: スコア降順の上位 = 最も誤統合しやすい(適合率の境界)。
    // 正例: スコア昇順の下位 = 最も取りこぼしやすい(再現率の境界)。
    // 飽和したベンチマークでも上位モデル同士の差がここに残るため、比較の主軸にする。
    let mut neg: Vec<&ScoredPair> = scored.iter().filter(|p| !p.3).collect();
    let mut pos: Vec<&ScoredPair> = scored.iter().filter(|p| p.3).collect();
    neg.sort_by(|a, b| b.2.total_cmp(&a.2));
    pos.sort_by(|a, b| a.2.total_cmp(&b.2));
    let to_named = |p: &&ScoredPair| (terms[p.0].clone(), terms[p.1].clone(), p.2);
    let hardest_negatives: Vec<NamedPair> = neg.iter().take(HARDEST_PAIRS).map(to_named).collect();
    let hardest_positives: Vec<NamedPair> = pos.iter().take(HARDEST_PAIRS).map(to_named).collect();

    Ok(Benchmark {
        model: model.to_string(),
        classification,
        scan,
        clustering,
        sweep,
        hardest_negatives,
        hardest_positives,
        speed: Speed {
            load_ms,
            embed_ms,
            per_text_ms: if num == 0 { 0.0 } else { embed_ms / num as f64 },
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
        writeln!(
            f,
            "-- クラスタ一致度 (normalize, 閾値 {:.1}) --",
            cl.threshold
        )?;
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
        writeln!(
            f,
            "-- 連鎖暴走の検知 (最大クラスタが飲み込んだ規模, 閾値 {:.1}) --",
            cl.threshold
        )?;
        writeln!(
            f,
            "  最大クラスタ={}語 (全体の{:.1}%)  巻込グループ数={} (健全=1, 多いほど暴走)",
            cl.largest_cluster_size,
            cl.largest_cluster_ratio * 100.0,
            cl.groups_in_largest
        )?;
        writeln!(
            f,
            "-- 誤統合 (データ破壊: 異グループの語を統合した数, 閾値 {:.1}) --",
            cl.threshold
        )?;
        write!(f, "  誤統合ペア数={}", cl.false_merges)?;
        if cl.false_merge_examples.is_empty() {
            writeln!(f, " (なし)")?;
        } else {
            let shown: Vec<String> = cl
                .false_merge_examples
                .iter()
                .map(|(a, b)| format!("{a}⇔{b}"))
                .collect();
            let more = cl.false_merges.saturating_sub(shown.len());
            let suffix = if more > 0 {
                format!(" ほか{more}件")
            } else {
                String::new()
            };
            writeln!(f, "  例: {}{}", shown.join(", "), suffix)?;
        }
        writeln!(
            f,
            "-- 閾値スイープ (分類F1 / クラスタF1 P R / 誤統合 / 最大率 巻込G) --"
        )?;
        writeln!(
            f,
            "  閾値 |  分類F1 | クラスタF1 (   P  /   R  ) | 誤統合 | 最大率 巻込G"
        )?;
        for row in &self.sweep {
            writeln!(
                f,
                "  {:>4.0} |  {:.3}  |   {:.3}   ( {:.3} / {:.3} ) | {:>5}  | {:>5.1}%  {:>4}",
                row.threshold,
                row.class_f1,
                row.cluster_f1,
                row.cluster_precision,
                row.cluster_recall,
                row.false_merges,
                row.largest_cluster_ratio * 100.0,
                row.groups_in_largest
            )?;
        }
        writeln!(
            f,
            "-- 最難負例 (誤統合しやすい順・閾値非依存。上位モデルほどスコアが低い) --"
        )?;
        if self.hardest_negatives.is_empty() {
            writeln!(f, "  (なし)")?;
        } else {
            for (a, b, s) in &self.hardest_negatives {
                writeln!(f, "  {s:>5.1}  {a} ⇔ {b}")?;
            }
        }
        writeln!(
            f,
            "-- 最難正例 (取りこぼしやすい順・閾値非依存。上位モデルほどスコアが高い) --"
        )?;
        if self.hardest_positives.is_empty() {
            writeln!(f, "  (なし)")?;
        } else {
            for (a, b, s) in &self.hardest_positives {
                writeln!(f, "  {s:>5.1}  {a} ⇔ {b}")?;
            }
        }
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
    fn default_glossary_has_no_duplicate_terms() {
        // 同じ語が複数グループに現れると正解ラベルが壊れる(本来統合すべきペアが
        // 別グループ扱いで負例化する、など)。データ拡張時の事故を検知する。
        let g = default_glossary();
        let mut seen: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for (gid, group) in g.groups.iter().enumerate() {
            for term in group {
                if let Some(prev) = seen.insert(term.as_str(), gid) {
                    panic!("用語 {term:?} がグループ {prev} と {gid} に重複しています");
                }
            }
        }
    }

    #[test]
    fn default_glossary_has_hard_cases() {
        // 飽和対策で導入した難所(難正例・難負例)が含まれていることを保証する。
        // 難負例の共下位語・対義語は単独グループ、難正例は多言語混在の同義グループ。
        let g = default_glossary();
        let flat: Vec<&str> = g.groups.iter().flatten().map(String::as_str).collect();
        // 難正例(表層が重ならない同義語)
        for t in ["便所", "restroom", "厕所"] {
            assert!(flat.contains(&t), "難正例 {t} が見つかりません");
        }
        // 難負例(共下位語・紛らわしい異義語・対義語)
        for t in ["春", "夏", "科学", "化学", "増加", "減少"] {
            assert!(flat.contains(&t), "難負例 {t} が見つかりません");
        }
        // 難負例は単独グループ(誤って同義グループに入れていないこと)
        let singletons: std::collections::HashSet<&str> = g
            .groups
            .iter()
            .filter(|grp| grp.len() == 1)
            .map(|grp| grp[0].as_str())
            .collect();
        for t in ["春", "科学", "増加"] {
            assert!(singletons.contains(&t), "{t} は単独語(難負例)であるべき");
        }
    }

    #[test]
    fn largest_cluster_topology_detects_chaining() {
        // 健全: 各正解グループが独立クラスタ → 巻込グループ数は 1。
        let roots = vec![0, 0, 2, 2, 4];
        let group_ids = vec![0, 0, 1, 1, 2];
        let (size, groups) = largest_cluster_topology(&roots, &group_ids);
        assert_eq!((size, groups), (2, 1));

        // 連鎖暴走: 1 つの根に複数の無関係グループが流れ込む。
        let roots = vec![9, 9, 9, 9, 5];
        let (size, groups) = largest_cluster_topology(&roots, &group_ids);
        assert_eq!(size, 4, "最大クラスタは 4 語");
        assert_eq!(groups, 2, "無関係な 2 グループを横断 = 暴走の検知");
    }

    #[test]
    fn sweep_detects_runaway_via_hub() {
        // 3 グループ {0,1}{2,3}{4}。ハブ的なペア (1,2) が別グループを橋渡しすると、
        // 低閾値で 0-1-2-3 が 1 クラスタへ連鎖する。トポロジー指標がこれを捉える。
        let group_ids = vec![0usize, 0, 1, 1, 2];
        let scored: Vec<ScoredPair> = vec![
            (0, 1, 100.0, true), // group0 内
            (2, 3, 100.0, true), // group1 内
            (1, 2, 90.0, false), // 橋(異グループ)
            (3, 4, 10.0, false), // 弱い無関係ペア
        ];
        let rows = sweep_from_scored(5, &scored, &group_ids, &[50.0, 95.0]);
        // 閾値 50: 橋が生きて 4 語が連鎖、2 グループを横断。
        assert_eq!(rows[0].groups_in_largest, 2);
        assert!((rows[0].largest_cluster_ratio - 0.8).abs() < 1e-9);
        // 閾値 95: 橋が切れ、各正解グループは独立 → 巻込は 1。
        assert_eq!(rows[1].groups_in_largest, 1);
        assert!((rows[1].largest_cluster_ratio - 0.4).abs() < 1e-9);
    }

    #[test]
    fn distractors_are_all_singletons_and_disjoint_from_glossary() {
        // ディストラクタは全行が単独語(負例の母集団)。グループ化されていたら誤り。
        let d = default_distractors();
        for g in &d.groups {
            assert_eq!(g.len(), 1, "ディストラクタ {g:?} は単独語であるべき");
        }
        // 用語集とディストラクタで語が重複するとラベルが壊れる(連結時に偽陰性化)。
        let g = default_glossary();
        let glossary: std::collections::HashSet<&str> =
            g.groups.iter().flatten().map(String::as_str).collect();
        for g in &d.groups {
            assert!(
                !glossary.contains(g[0].as_str()),
                "ディストラクタ {:?} が用語集と重複しています",
                g[0]
            );
        }
        // ディストラクタ内でも語は一意であること。
        let mut seen = std::collections::HashSet::new();
        for g in &d.groups {
            assert!(
                seen.insert(g[0].as_str()),
                "ディストラクタ内で {:?} が重複",
                g[0]
            );
        }
    }

    #[test]
    fn distractors_contain_hub_tags() {
        // 連鎖暴走の橋になりやすいハブ語(汎用短トークン)が含まれていることを保証する。
        let flat: std::collections::HashSet<String> = default_distractors()
            .groups
            .iter()
            .flatten()
            .cloned()
            .collect();
        for t in ["1", "2B", "3D"] {
            assert!(
                flat.contains(t),
                "ハブ語 {t} がディストラクタに見つかりません"
            );
        }
        // 規模も連鎖暴走の前提。ある程度の語数を確保しておく。
        assert!(
            flat.len() >= 200,
            "ディストラクタが少なすぎます(連鎖暴走の再現には規模が要る): {}",
            flat.len()
        );
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
