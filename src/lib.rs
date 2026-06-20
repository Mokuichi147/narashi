//! 多言語埋め込みを使って類似テキストを検出し、より汎用的な表記に統合することで
//! 表記ゆれを解消するライブラリ。
//!
//! # 仕組み
//!
//! - **類似度判定**: 多言語埋め込みモデル(既定: `Alibaba-NLP/gte-multilingual-base`)で
//!   各テキストをベクトル化し、コサイン類似度をモデル固有のベースライン基準で
//!   0〜100 スコアに校正(高帯域に偏るコサイン値を識別しやすいスケールへ展開)
//! - **汎用性判定**: `(トークン数, トークンID合計)` の辞書式比較で最小のものを
//!   代表 (canonical) として採用(短く・低IDなトークンで構成されるほど汎用的とみなす)
//! - **グルーピング**: 閾値以上のペアを union-find で連結成分化(ペア判定は並列化)
//! - **モデル選択**: [`Options::with_model`] で同規模の対称類似度モデル等へ切替可能
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
//! let groups = n.normalize(&texts, 70.0)?;
//! for g in &groups {
//!     println!("{} <- {:?}", g.canonical, g.members);
//! }
//! # anyhow::Ok(())
//! ```
//!
//! # キャッシュ
//!
//! モデル・トークナイザは初回実行時に自動ダウンロードされます(既定の gte は約 1.2GB、
//! `--model small` なら約 0.45GB)。保存先の優先順位は以下のとおり:
//!
//! 1. [`Options::with_cache_dir`] による明示指定
//! 2. 環境変数 [`CACHE_DIR_ENV`] (`NARASHI_CACHE_DIR`)
//! 3. `std::env::temp_dir().join("narashi")`

use anyhow::{Result, anyhow};
pub use fastembed::EmbeddingModel;
use fastembed::{
    InitOptions, InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use hf_hub::api::sync::ApiBuilder;
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// キャッシュ保存先を上書きするための環境変数名 (`NARASHI_CACHE_DIR`)
pub const CACHE_DIR_ENV: &str = "NARASHI_CACHE_DIR";

/// 既定の統合閾値 (0〜100)
///
/// 既定モデルでこの値以上のスコアを持つペアが統合されます。CLI の `--threshold`
/// 既定値および評価ベンチマークの基準として共有されます。
pub const DEFAULT_THRESHOLD: f32 = 70.0;

pub mod eval;

/// 既定の埋め込みモデル(gte-multilingual-base)
///
/// 用語集ベンチマーク(日本語・英語・中国語混在)で、実運用挙動を表す clusterF1 が
/// 全候補中で最高(ピーク 0.682)。しかもそのピークを既定閾値 [`DEFAULT_THRESHOLD`]
/// ちょうどで、高い適合率(P=0.964 ≒ 誤統合がほぼ無い)を保ったまま達成する。
/// 詳細な比較は `docs/benchmarks.md` を参照。
///
/// 速度・サイズ重視なら `--model small`(multilingual-e5-small, 約 0.45GB・最速級)へ、
/// 高再現率重視なら `--model paraphrase` へ切り替えられる。
pub const DEFAULT_MODEL: Model = Model::UserDefined(UserModel::GteMultilingualBase);

/// fastembed の組み込みカタログに無いユーザー定義モデル
///
/// fastembed の `try_new_from_user_defined` を使い、HF リポジトリの ONNX を
/// 直接読み込んで利用する。組み込みモデルと同じ指標で比較するために用意している。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UserModel {
    /// Alibaba-NLP/gte-multilingual-base
    ///
    /// 日本語を含む CJK に強い多言語モデル(768次元・CLS プーリング)。
    /// E5 系の対抗候補。ONNX は単一ファイル(約 1.2GB)で外部重みを持たない。
    GteMultilingualBase,
    /// distiluse-base-multilingual-cased-v2 (Xenova の ONNX)
    ///
    /// 50+ 言語対応の軽量多言語モデル(DistilBERT・Mean プーリング・768次元)。
    /// ONNX は単一ファイル(約 0.54GB)で外部重みを持たない。gte に次ぐ精度
    /// (clusterF1 ピーク 0.667 を既定閾値 70 で達成)を最小サイズ・最速級で得る
    /// 軽量代替。詳細は `docs/benchmarks.md` を参照。
    DistiluseMultilingualV2,
}

/// narashi が利用できる埋め込みモデルの選択
///
/// fastembed の組み込みモデル([`EmbeddingModel`])と、HF リポジトリから直接読み込む
/// [`UserModel`] の双方を統一的に扱う。`From<EmbeddingModel>` があるため
/// `with_model(EmbeddingModel::...)` のように組み込みモデルを直接渡せる。
#[derive(Debug, Clone)]
pub enum Model {
    /// fastembed の組み込みモデル
    Builtin(EmbeddingModel),
    /// HF リポジトリから読み込むユーザー定義モデル
    UserDefined(UserModel),
}

impl From<EmbeddingModel> for Model {
    fn from(m: EmbeddingModel) -> Self {
        Model::Builtin(m)
    }
}

impl From<UserModel> for Model {
    fn from(m: UserModel) -> Self {
        Model::UserDefined(m)
    }
}

/// モデルの読み込み方法
enum ModelSource {
    /// fastembed の組み込みモデル。fastembed 自身が ONNX を取得・管理する。
    Builtin(EmbeddingModel),
    /// ユーザー定義モデル。`ModelSpec::hf_repo` から ONNX を直接読み込む。
    UserDefined {
        /// リポジトリ内の ONNX ファイルパス(例: `"onnx/model.onnx"`)
        onnx_file: &'static str,
        /// プーリング方式(モデル依存。E5/MiniLM は Mean、BGE/GTE は CLS)
        pooling: Pooling,
    },
}

/// モデルごとの取り扱いを記述したメタ情報
///
/// モデルによって入力プレフィックスやコサイン類似度の分布が異なるため、
/// トークナイザ取得元・プレフィックス・スコア校正のベースライン・読み込み方法を
/// 切り替える。
struct ModelSpec {
    /// `tokenizer.json` を取得する Hugging Face リポジトリ ID
    /// (ユーザー定義モデルでは ONNX の取得元も兼ねる)
    hf_repo: &'static str,
    /// 埋め込み入力に付与するプレフィックス(E5 系は `"query: "`、対称モデルは空)
    query_prefix: &'static str,
    /// スコア校正の基準となるコサイン値(無関係な短文ペアの典型的な下限)
    ///
    /// この値を 0、コサイン 1.0 を 100 に写像してスコアの識別力を高める。
    cos_baseline: f32,
    /// ONNX の読み込み方法
    source: ModelSource,
}

/// 指定モデルの取り扱いメタ情報を返す
fn model_spec(model: &Model) -> ModelSpec {
    match model {
        Model::Builtin(m) => builtin_spec(m),
        Model::UserDefined(UserModel::GteMultilingualBase) => ModelSpec {
            hf_repo: "onnx-community/gte-multilingual-base",
            // GTE は STS/類似度用途では指示プレフィックス無しの対称利用。
            query_prefix: "",
            cos_baseline: 0.42,
            source: ModelSource::UserDefined {
                onnx_file: "onnx/model.onnx",
                pooling: Pooling::Cls,
            },
        },
        Model::UserDefined(UserModel::DistiluseMultilingualV2) => ModelSpec {
            hf_repo: "Xenova/distiluse-base-multilingual-cased-v2",
            query_prefix: "",
            // ピーク clusterF1 を既定閾値 70 に合わせる校正値(ベンチで決定)
            cos_baseline: 0.39,
            source: ModelSource::UserDefined {
                onnx_file: "onnx/model.onnx",
                pooling: Pooling::Mean,
            },
        },
    }
}

/// fastembed 組み込みモデルの取り扱いメタ情報を返す
fn builtin_spec(model: &EmbeddingModel) -> ModelSpec {
    let (hf_repo, query_prefix, cos_baseline) = match model {
        EmbeddingModel::MultilingualE5Small => ("intfloat/multilingual-e5-small", "query: ", 0.70),
        EmbeddingModel::MultilingualE5Base => ("intfloat/multilingual-e5-base", "query: ", 0.70),
        EmbeddingModel::MultilingualE5Large => ("intfloat/multilingual-e5-large", "query: ", 0.70),
        EmbeddingModel::ParaphraseMLMiniLML12V2 | EmbeddingModel::ParaphraseMLMiniLML12V2Q => {
            ("Xenova/paraphrase-multilingual-MiniLM-L12-v2", "", 0.30)
        }
        EmbeddingModel::ParaphraseMLMpnetBaseV2 => {
            ("Xenova/paraphrase-multilingual-mpnet-base-v2", "", 0.30)
        }
        // --- 別系統モデル (比較用)。多言語特化ではないため精度比較のベースライン ---
        // BGE 中国語特化 (BAAI 系)。中国語には強いが日本語は学習外。
        EmbeddingModel::BGESmallZHV15 => ("Xenova/bge-small-zh-v1.5", "", 0.30),
        // 英語 sentence-transformers (非多言語のベースライン)。
        EmbeddingModel::AllMiniLML6V2 => ("Qdrant/all-MiniLM-L6-v2-onnx", "", 0.0),
        // CLIP テキストエンコーダ (対照学習・全く別アーキテクチャ)。
        EmbeddingModel::ClipVitB32 => ("Qdrant/clip-ViT-B-32-text", "", 0.0),
        // 未対応モデルは E5 small 相当の保守的な既定で扱う
        _ => ("intfloat/multilingual-e5-small", "query: ", 0.70),
    };
    ModelSpec {
        hf_repo,
        query_prefix,
        cos_baseline,
        source: ModelSource::Builtin(model.clone()),
    }
}

/// [`Narashi`] の初期化オプション
#[derive(Debug, Clone, Default)]
pub struct Options {
    cache_dir: Option<PathBuf>,
    model: Option<Model>,
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

    /// 使用する埋め込みモデルを指定する(未指定時は [`DEFAULT_MODEL`])
    ///
    /// 組み込みモデル([`EmbeddingModel`])とユーザー定義モデル([`UserModel`])の
    /// どちらも `Into<Model>` 経由で渡せる。
    pub fn with_model(mut self, model: impl Into<Model>) -> Self {
        self.model = Some(model.into());
        self
    }

    /// 実際に使用する埋め込みモデルを解決する
    pub fn resolved_model(&self) -> Model {
        self.model.clone().unwrap_or(DEFAULT_MODEL)
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
    /// 埋め込み入力に付与するプレフィックス(モデル依存)
    query_prefix: &'static str,
    /// スコア校正の基準コサイン値(モデル依存)
    cos_baseline: f32,
}

impl Narashi {
    /// デフォルト設定で初期化する
    ///
    /// 必要に応じてモデル・トークナイザをダウンロードします(初回のみ、既定の gte は約 1.2GB)。
    pub fn new() -> Result<Self> {
        Self::with_options(Options::default())
    }

    /// 指定したオプションで初期化する
    pub fn with_options(opts: Options) -> Result<Self> {
        let cache_dir = opts.resolved_cache_dir();
        std::fs::create_dir_all(&cache_dir)?;

        let model = opts.resolved_model();
        let spec = model_spec(&model);

        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir.clone())
            .build()
            .map_err(|e| anyhow!("hf-hub init failed: {e}"))?;
        let repo = api.model(spec.hf_repo.to_string());

        // 埋め込みモデルを読み込む。組み込みは fastembed が ONNX を管理し、
        // ユーザー定義は hf_repo の ONNX を直接読み込む。
        let embedder = match &spec.source {
            ModelSource::Builtin(m) => TextEmbedding::try_new(
                InitOptions::new(m.clone()).with_cache_dir(cache_dir.clone()),
            )?,
            ModelSource::UserDefined { onnx_file, pooling } => {
                let fetch = |name: &str| -> Result<Vec<u8>> {
                    let path = repo
                        .get(name)
                        .map_err(|e| anyhow!("{name} download failed: {e}"))?;
                    Ok(std::fs::read(path)?)
                };
                let tokenizer_files = TokenizerFiles {
                    tokenizer_file: fetch("tokenizer.json")?,
                    config_file: fetch("config.json")?,
                    special_tokens_map_file: fetch("special_tokens_map.json")?,
                    tokenizer_config_file: fetch("tokenizer_config.json")?,
                };
                let user_model = UserDefinedEmbeddingModel::new(fetch(onnx_file)?, tokenizer_files)
                    .with_pooling(pooling.clone());
                TextEmbedding::try_new_from_user_defined(user_model, InitOptionsUserDefined::new())?
            }
        };

        // 代表選出のトークン数計算に使う独立したトークナイザ(全モデル共通で hf_repo から)
        let tokenizer_path = repo
            .get("tokenizer.json")
            .map_err(|e| anyhow!("tokenizer download failed: {e}"))?;
        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("tokenizer load failed: {e}"))?;
        Ok(Self {
            embedder,
            tokenizer,
            query_prefix: spec.query_prefix,
            cos_baseline: spec.cos_baseline,
        })
    }

    /// 2つのテキストの類似度を 0〜100 で返す
    ///
    /// 100 に近いほど類似。コサイン類似度をモデル固有のベースライン(無関係な
    /// 短文ペアの典型的な下限)を 0、完全一致 1.0 を 100 として校正した値です。
    /// これにより高帯域に偏りがちなコサイン値を識別しやすいスケールへ広げます。
    pub fn similarity(&self, a: &str, b: &str) -> Result<f32> {
        let embeddings = self.embed_normalized(&[a.to_string(), b.to_string()])?;
        Ok(self.score(dot(&embeddings[0], &embeddings[1])))
    }

    /// 入力テキスト群をプレフィックス付きで埋め込み、各ベクトルを L2 正規化して返す
    ///
    /// 正規化済みベクトル同士のコサイン類似度はドット積に一致するため、
    /// 以降のペア比較では平方根を伴う norm 再計算を省ける。
    pub(crate) fn embed_normalized(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inputs: Vec<String> = texts
            .iter()
            .map(|t| format!("{}{}", self.query_prefix, t))
            .collect();
        let mut embeddings = self.embedder.embed(inputs, None)?;
        embeddings.par_iter_mut().for_each(|v| normalize_l2(v));
        Ok(embeddings)
    }

    /// コサイン類似度を 0〜100 のスコアへ校正する(モデル依存のベースライン基準)
    pub(crate) fn score(&self, cos: f32) -> f32 {
        ((cos - self.cos_baseline) / (1.0 - self.cos_baseline)).clamp(0.0, 1.0) * 100.0
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

        let embeddings = self.embed_normalized(texts)?;
        let embeddings = &embeddings;

        // O(n²) のペア判定を rayon で並列化し、閾値以上のペアだけ収集する。
        // union 自体は安価なため、収集後に逐次でまとめる。
        let pairs: Vec<(usize, usize)> = (0..n)
            .into_par_iter()
            .flat_map_iter(|i| {
                ((i + 1)..n).filter_map(move |j| {
                    let sim = self.score(dot(&embeddings[i], &embeddings[j]));
                    (sim >= threshold).then_some((i, j))
                })
            })
            .collect();

        let mut uf = UnionFind::new(n);
        for (i, j) in pairs {
            uf.union(i, j);
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
                let mut members: Vec<String> = indices.iter().map(|&i| texts[i].clone()).collect();
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

/// ベクトルを L2 ノルムで正規化する(ゼロベクトルは変更しない)
fn normalize_l2(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// 内積を返す。L2 正規化済みベクトル同士ではコサイン類似度に一致する。
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
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
    fn normalized_dot_identical() {
        let mut a = [3.0, 0.0, 0.0];
        normalize_l2(&mut a);
        assert!((dot(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalized_dot_orthogonal() {
        let mut a = [1.0, 0.0, 0.0];
        let mut b = [0.0, 2.0, 0.0];
        normalize_l2(&mut a);
        normalize_l2(&mut b);
        assert!(dot(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn normalize_l2_zero_vector() {
        let mut z = [0.0, 0.0, 0.0];
        normalize_l2(&mut z);
        assert_eq!(z, [0.0, 0.0, 0.0]);
    }

    /// スコア校正: コサイン=baseline で 0、=1.0 で 100、baseline 未満は 0 にクランプ。
    fn score_with(baseline: f32, cos: f32) -> f32 {
        ((cos - baseline) / (1.0 - baseline)).clamp(0.0, 1.0) * 100.0
    }

    #[test]
    fn score_calibration_range() {
        let base = 0.70;
        assert!((score_with(base, 1.0) - 100.0).abs() < 1e-4);
        assert!((score_with(base, base) - 0.0).abs() < 1e-4);
        // baseline 未満は下限クランプ
        assert!((score_with(base, 0.5) - 0.0).abs() < 1e-4);
        // 中間点(baseline と 1.0 の中央)は 50
        assert!((score_with(base, (base + 1.0) / 2.0) - 50.0).abs() < 1e-4);
    }

    #[test]
    fn options_resolution_precedence() {
        // SAFETY: mutates process env; single test avoids races with peers
        unsafe {
            std::env::remove_var(CACHE_DIR_ENV);
        }

        let resolved = Options::new().resolved_cache_dir();
        assert!(resolved.starts_with(std::env::temp_dir()));
        assert!(resolved.ends_with("narashi"));

        unsafe {
            std::env::set_var(CACHE_DIR_ENV, "/from/env");
        }
        let resolved = Options::new().resolved_cache_dir();
        assert_eq!(resolved, PathBuf::from("/from/env"));

        let resolved = Options::new()
            .with_cache_dir("/explicit/path")
            .resolved_cache_dir();
        assert_eq!(resolved, PathBuf::from("/explicit/path"));

        unsafe {
            std::env::remove_var(CACHE_DIR_ENV);
        }
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
        // 校正後スケール: 関連の強い表記ゆれは無関係ペアより明確に高くなる。
        // 既定モデル gte では「猫」⇔「ネコ」≒73、「猫」⇔「自動車」≒28 と大きく分離する。
        let related = n.similarity("猫", "ネコ").unwrap();
        let unrelated = n.similarity("猫", "自動車").unwrap();
        assert!(
            related > 60.0 && related > unrelated + 20.0,
            "expected related ({related}) clearly above unrelated ({unrelated})"
        );
    }

    #[test]
    #[ignore]
    fn real_normalize_groups() {
        let n = Narashi::new().unwrap();
        let texts: Vec<String> = ["猫", "ネコ", "犬", "イヌ", "自動車"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let groups = n.normalize(&texts, 70.0).unwrap();
        for g in &groups {
            println!("canonical={} members={:?}", g.canonical, g.members);
        }
        assert!(!groups.is_empty());
    }
}
