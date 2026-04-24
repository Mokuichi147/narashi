//! 多言語埋め込みを使って類似テキストを検出し、より汎用的な表記に統合することで
//! 表記ゆれを解消するライブラリ。
//!
//! # 仕組み
//!
//! - **類似度判定**: 多言語 E5 (`intfloat/multilingual-e5-small`) で各テキストを
//!   ベクトル化し、コサイン類似度を 0〜100 スコアに変換
//! - **汎用性判定**: `(トークン数, トークンID合計)` の辞書式比較で最小のものを
//!   代表 (canonical) として採用(短く・低IDなトークンで構成されるほど汎用的とみなす)
//! - **グルーピング**: 閾値以上のペアを union-find で連結成分化
//!
//! # 使い方
//!
//! ```no_run
//! use narashi::Narashi;
//!
//! let n = Narashi::new()?;
//! let score = n.similarity("白い背景", "白背景")?;
//! println!("{score:.1}");
//! # anyhow::Ok(())
//! ```
//!
//! 複数テキストをまとめて正規化:
//!
//! ```no_run
//! use narashi::Narashi;
//!
//! let n = Narashi::new()?;
//! let texts: Vec<String> = ["白い背景", "白背景", "漫画", "マンガ"]
//!     .iter().map(|s| s.to_string()).collect();
//! let groups = n.normalize(&texts, 95.0)?;
//! for g in &groups {
//!     println!("{} <- {:?}", g.canonical, g.members);
//! }
//! # anyhow::Ok(())
//! ```
//!
//! # キャッシュ
//!
//! モデル・トークナイザは初回実行時に自動ダウンロードされます(約 500MB)。
//! 保存先の優先順位は以下のとおり:
//!
//! 1. [`Options::with_cache_dir`] による明示指定
//! 2. 環境変数 [`CACHE_DIR_ENV`] (`NARASHI_CACHE_DIR`)
//! 3. `std::env::temp_dir().join("narashi")`

use anyhow::{Result, anyhow};
use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};
use hf_hub::api::sync::ApiBuilder;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

const MODEL_REPO: &str = "intfloat/multilingual-e5-small";

/// キャッシュ保存先を上書きするための環境変数名 (`NARASHI_CACHE_DIR`)
pub const CACHE_DIR_ENV: &str = "NARASHI_CACHE_DIR";

/// [`Narashi`] の初期化オプション
#[derive(Debug, Clone, Default)]
pub struct Options {
    cache_dir: Option<PathBuf>,
}

impl Options {
    /// デフォルト値でオプションを生成する
    pub fn new() -> Self {
        Self::default()
    }

    /// キャッシュディレクトリを明示指定する(環境変数・デフォルトより優先)
    pub fn with_cache_dir(mut self, dir: impl AsRef<Path>) -> Self {
        self.cache_dir = Some(dir.as_ref().to_path_buf());
        self
    }

    /// 実際に使用するキャッシュパスを解決する
    ///
    /// 優先順位: 明示指定 > [`CACHE_DIR_ENV`] > `std::env::temp_dir()/narashi`
    pub fn resolved_cache_dir(&self) -> PathBuf {
        self.cache_dir
            .clone()
            .or_else(|| std::env::var_os(CACHE_DIR_ENV).map(PathBuf::from))
            .unwrap_or_else(|| std::env::temp_dir().join("narashi"))
    }
}

/// 1つの統合グループ
///
/// `canonical` が代表となるテキスト、`members` はそれに統合された元のテキスト
/// すべて(canonical 自身も含む)。元のテキストがどの代表に統合されたかを
/// 追跡するためにこの構造体を返します。
#[derive(Debug, Clone)]
pub struct Group {
    /// グループの代表テキスト(最も汎用的)
    pub canonical: String,
    /// グループに含まれる全テキスト(canonical 自身も含む)
    pub members: Vec<String>,
}

/// 表記ゆれ解消の本体
///
/// 埋め込みモデルとトークナイザを保持し、類似度の計算とグルーピングを行います。
/// モデルの読み込みには時間がかかるため、複数回使う場合は同じインスタンスを
/// 再利用してください。
pub struct Narashi {
    embedder: TextEmbedding,
    tokenizer: Tokenizer,
}

impl Narashi {
    /// デフォルト設定で初期化する
    ///
    /// 必要に応じてモデル・トークナイザをダウンロードします(初回のみ、約 500MB)。
    pub fn new() -> Result<Self> {
        Self::with_options(Options::default())
    }

    /// 指定したオプションで初期化する
    pub fn with_options(opts: Options) -> Result<Self> {
        let cache_dir = opts.resolved_cache_dir();
        std::fs::create_dir_all(&cache_dir)?;

        let embedder = TextEmbedding::try_new(
            InitOptions::new(EmbeddingModel::MultilingualE5Small)
                .with_cache_dir(cache_dir.clone()),
        )?;

        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .build()
            .map_err(|e| anyhow!("hf-hub init failed: {e}"))?;
        let repo = api.model(MODEL_REPO.to_string());
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| anyhow!("tokenizer download failed: {e}"))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("tokenizer load failed: {e}"))?;
        Ok(Self { embedder, tokenizer })
    }

    /// 2つのテキストの類似度を 0〜100 で返す
    ///
    /// 100 に近いほど類似。コサイン類似度 `[-1, 1]` を `[0, 100]` に線形変換した値です。
    pub fn similarity(&self, a: &str, b: &str) -> Result<f32> {
        let inputs = vec![format!("query: {a}"), format!("query: {b}")];
        let embeddings = self.embedder.embed(inputs, None)?;
        Ok(cosine_to_score(cosine_similarity(
            &embeddings[0],
            &embeddings[1],
        )))
    }

    /// テキスト群を表記ゆれごとにグループ化し、代表(canonical)を選出する
    ///
    /// `threshold` (0〜100) 以上の類似度を持つペアは同じグループに統合されます。
    /// 各グループの代表は `(トークン数, トークンID合計)` の辞書式最小によって決定されます。
    pub fn normalize(&self, texts: &[String], threshold: f32) -> Result<Vec<Group>> {
        let n = texts.len();
        if n == 0 {
            return Ok(vec![]);
        }

        let inputs: Vec<String> = texts.iter().map(|t| format!("query: {t}")).collect();
        let embeddings = self.embedder.embed(inputs, None)?;

        let mut uf = UnionFind::new(n);
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_to_score(cosine_similarity(&embeddings[i], &embeddings[j]));
                if sim >= threshold {
                    uf.union(i, j);
                }
            }
        }

        let keys: Vec<(usize, u64)> = texts
            .iter()
            .map(|t| self.generality_key(t))
            .collect::<Result<Vec<_>>>()?;

        let mut buckets: HashMap<usize, Vec<usize>> = HashMap::new();
        for i in 0..n {
            buckets.entry(uf.find(i)).or_default().push(i);
        }

        let mut groups: Vec<Group> = buckets
            .into_values()
            .map(|indices| {
                let canonical_idx = *indices.iter().min_by_key(|&&i| keys[i]).unwrap();
                let mut members: Vec<String> =
                    indices.iter().map(|&i| texts[i].clone()).collect();
                members.sort();
                Group {
                    canonical: texts[canonical_idx].clone(),
                    members,
                }
            })
            .collect();

        groups.sort_by(|a, b| a.canonical.cmp(&b.canonical));
        Ok(groups)
    }

    fn generality_key(&self, text: &str) -> Result<(usize, u64)> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow!("tokenize failed: {e}"))?;
        let ids = encoding.get_ids();
        let count = ids.len();
        let sum: u64 = ids.iter().map(|&id| id as u64).sum();
        Ok((count, sum))
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

fn cosine_to_score(cos: f32) -> f32 {
    ((cos + 1.0) / 2.0 * 100.0).clamp(0.0, 100.0)
}

struct UnionFind {
    parent: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
        }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            let p = self.parent[x];
            self.parent[x] = self.find(p);
        }
        self.parent[x]
    }
    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx != ry {
            self.parent[rx] = ry;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let a = [1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn score_range() {
        assert!((cosine_to_score(1.0) - 100.0).abs() < 1e-4);
        assert!((cosine_to_score(0.0) - 50.0).abs() < 1e-4);
        assert!((cosine_to_score(-1.0) - 0.0).abs() < 1e-4);
    }

    #[test]
    fn options_resolution_precedence() {
        // SAFETY: mutates process env; single test avoids races with peers
        unsafe { std::env::remove_var(CACHE_DIR_ENV); }

        let resolved = Options::new().resolved_cache_dir();
        assert!(resolved.starts_with(std::env::temp_dir()));
        assert!(resolved.ends_with("narashi"));

        unsafe { std::env::set_var(CACHE_DIR_ENV, "/from/env"); }
        let resolved = Options::new().resolved_cache_dir();
        assert_eq!(resolved, PathBuf::from("/from/env"));

        let resolved = Options::new()
            .with_cache_dir("/explicit/path")
            .resolved_cache_dir();
        assert_eq!(resolved, PathBuf::from("/explicit/path"));

        unsafe { std::env::remove_var(CACHE_DIR_ENV); }
    }

    #[test]
    fn union_find_merges() {
        let mut uf = UnionFind::new(4);
        uf.union(0, 1);
        uf.union(1, 2);
        assert_eq!(uf.find(0), uf.find(2));
        assert_ne!(uf.find(0), uf.find(3));
    }

    #[test]
    #[ignore]
    fn real_similarity_high() {
        let n = Narashi::new().unwrap();
        let s = n.similarity("猫", "ネコ").unwrap();
        assert!(s > 70.0, "expected high similarity, got {s}");
    }

    #[test]
    #[ignore]
    fn real_normalize_groups() {
        let n = Narashi::new().unwrap();
        let texts: Vec<String> = ["猫", "ネコ", "犬", "イヌ", "自動車"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let groups = n.normalize(&texts, 80.0).unwrap();
        for g in &groups {
            println!("canonical={} members={:?}", g.canonical, g.members);
        }
        assert!(!groups.is_empty());
    }
}
