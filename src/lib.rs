//! 多言語埋め込みを使って類似テキストを検出し、より汎用的な表記に統合することで
//! 表記ゆれを解消するライブラリ。
//!
//! # 仕組み
//!
//! - **類似度判定**: 多言語埋め込みモデル(既定: `BAAI/bge-m3`)で
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
//! モデル・トークナイザは初回実行時に自動ダウンロードされます(既定の bge-m3 は約 1.06GB、
//! `--model small` なら約 0.45GB)。保存先の優先順位は以下のとおり:
//!
//! 1. [`Options::with_cache_dir`] による明示指定
//! 2. 環境変数 [`CACHE_DIR_ENV`] (`NARASHI_CACHE_DIR`)
//! 3. `std::env::temp_dir().join("narashi")`

use anyhow::{Result, anyhow};
#[cfg(feature = "onnx")]
pub use fastembed::EmbeddingModel;
#[cfg(feature = "onnx")]
use fastembed::{
    InitOptions, InitOptionsUserDefined, Pooling, TextEmbedding, TokenizerFiles,
    UserDefinedEmbeddingModel,
};
use hf_hub::api::sync::{ApiBuilder, ApiRepo};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

#[cfg(feature = "candle")]
mod candle_backend;

#[cfg(not(any(feature = "onnx", feature = "candle")))]
compile_error!("少なくとも 1 つのバックエンド機能(`onnx` または `candle`)を有効にしてください");

/// キャッシュ保存先を上書きするための環境変数名 (`NARASHI_CACHE_DIR`)
pub const CACHE_DIR_ENV: &str = "NARASHI_CACHE_DIR";

/// 既定の統合閾値 (0〜100)
///
/// 既定モデルでこの値以上のスコアを持つペアが統合されます。CLI の `--threshold`
/// 既定値および評価ベンチマークの基準として共有されます。
pub const DEFAULT_THRESHOLD: f32 = 70.0;

pub mod eval;

/// 既定の埋め込みモデル(bge-m3)
///
/// 用語集ベンチマーク v2(日本語・英語・中国語混在・難正例/難負例を含む)で **ONNX 勢の
/// clusterF1 真ピーク 0.699 が最高**、かつピーク時適合率 P=0.939・誤統合わずか 7 件と**最も
/// 誤統合が少ない**。精度(clusterF1)・誤統合の少なさ・サイズ(約 1.06GB)のすべてで従来既定の
/// gte(0.657 / 誤統合 27 件 / 1.2GB)を上回る。校正(`cos_baseline=0.072`)により
/// clusterF1 真ピークを既定閾値 [`DEFAULT_THRESHOLD`] ちょうどで達成する。
///
/// 唯一の弱点は推論速度で、1024次元 + fp16(CPU)のため gte の約 3 倍(≈16ms/語)。
/// バッチ用途では問題にならないが、速度重視なら `--model gte`(約 1/3 の推論時間・
/// clusterF1 0.657)へ、軽量重視なら `--model distiluse`(約 0.54GB)/ `--model small`
/// (約 0.45GB)へ切り替えられる。詳細は `docs/benchmarks.md` を参照。
pub const DEFAULT_MODEL: Model = Model::UserDefined(UserModel::BgeM3);

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
    /// (clusterF1 ピーク 0.662 を既定閾値 70 で達成)を最小サイズ・最速級で得る
    /// 軽量代替。詳細は `docs/benchmarks.md` を参照。
    DistiluseMultilingualV2,
    /// ibm-granite/granite-embedding-97m-multilingual-r2
    ///
    /// IBM Granite Embedding R2(最新世代)の軽量版(384次元・CLS プーリング)。
    /// 200+ 言語対応で日本語は明示的な学習対象。ONNX は単一ファイル(約 0.39GB)。
    GraniteMultilingual97mR2,
    /// ibm-granite/granite-embedding-107m-multilingual
    ///
    /// IBM Granite Embedding R1 の軽量版(768次元・CLS プーリング)。
    /// XLM-RoBERTa 系。ONNX は単一ファイル(約 0.43GB)。
    GraniteMultilingual107m,
    /// ibm-granite/granite-embedding-278m-multilingual
    ///
    /// IBM Granite Embedding R1(768次元・CLS プーリング)。
    /// XLM-RoBERTa 系。ONNX は単一ファイル(約 1.1GB)。
    GraniteMultilingual278m,
    /// ibm-granite/granite-embedding-311m-multilingual-r2
    ///
    /// IBM Granite Embedding R2(最新世代)の標準版(768次元・CLS プーリング)。
    /// 200+ 言語対応で日本語は明示的な学習対象。ONNX は単一ファイル(約 1.25GB)。
    GraniteMultilingual311mR2,
    /// BAAI/bge-m3 (Xenova の fp16 ONNX)— **既定モデル** ([`DEFAULT_MODEL`])
    ///
    /// XLM-RoBERTa-large 系の高性能多言語モデル(1024次元・CLS プーリング・MIT)。
    /// 100+ 言語対応で日本語も対象。外部重み付き fp32 は単一ファイル化できないが、
    /// fp16 版は単一ファイル(約 1.06GB)で `commit_from_memory` 経由で読める。
    /// clusterF1 真ピーク 0.699(ONNX 勢で最高)を誤統合 7 件・P=0.939 で達成し、
    /// 精度・誤統合の少なさ・サイズで gte を上回る。推論は gte の約 3 倍(≈16ms/語)。
    BgeM3,
    /// Snowflake/snowflake-arctic-embed-l-v2.0 (fp16 ONNX)
    ///
    /// BGE-M3 系を検索向けに再学習した大型モデル(1024次元・CLS プーリング・Apache 2.0)。
    /// 74 言語対応。fp16 版が単一ファイル(約 1.06GB)。検索特化のため対称類似度では
    /// 同系 arctic-m-v2.0 同様に振るわない可能性があるが、大型枠の確認として評価する。
    ArcticEmbedLV2,
    /// intfloat/multilingual-e5-large-instruct(**Candle バックエンド専用**)
    ///
    /// XLM-RoBERTa-large 系の指示対応 E5(1024次元・Mean プーリング・MIT)。
    /// 100 言語対応で日本語も対象。**この HF リポジトリの ONNX は外部重み付き
    /// (`model.onnx_data`)で fastembed の単一ファイル経路では読めず、従来の ONNX
    /// バックエンドでは利用できなかった**が、Candle バックエンドが safetensors を直接
    /// 読み込むことで利用可能になった。指示プレフィックス(`"Instruct: ... \nQuery: "`)を
    /// 付与して対称類似度に用いる。`cos_baseline=0.67` に校正し clusterF1 真ピークを既定
    /// 閾値 70 に合わせている。ただし真ピークは 0.645(gte 0.657 未満)で、真ピーク時の
    /// 誤統合 75 件と全モデル中最多・推論も Candle CPU で遅い(≈220–270ms/語)ため既定には
    /// 採用せず、Candle バックエンドの実例兼 ONNX 非依存環境の選択肢に留める
    /// (詳細は `docs/benchmarks.md`)。
    E5LargeInstruct,
    /// Qwen/Qwen3-Embedding-0.6B(**Candle バックエンド専用**)
    ///
    /// Qwen3(デコーダ専用 Transformer)ベースの埋め込みモデル(1024次元・**LastToken
    /// プーリング**・Apache 2.0)。100+ 言語対応。**last-token プーリングは fastembed
    /// (Cls/Mean のみ)が非対応で従来は評価できなかった**が、Candle バックエンドが
    /// safetensors を直接読み、末尾(EOS)トークンの隠れ状態を取り出すことで利用可能に
    /// なった。指示プレフィックス(`"Instruct: ...\nQuery:"`)を付与して対称類似度に用いる。
    /// `cos_baseline=0.69` に校正し clusterF1 真ピークを既定閾値 70 付近に合わせている。
    /// **clusterF1 真ピーク 0.764(bge-m3 0.699 超)** を P=0.926・
    /// 誤統合 10 件と高精度・高安全性で達成する。ただしデコーダ系で重く、推論は Candle CPU で
    /// 低速(≈200ms/語・bge-m3 の約 12 倍)かつ Candle 専用のため、**既定は速度重視で
    /// bge-m3 のまま**とし、精度最優先/ONNX 非依存環境向けの選択肢として提供する
    /// (詳細は `docs/benchmarks.md`)。
    Qwen3Embedding0_6B,
    /// Qwen/Qwen3-Embedding-4B(**Candle バックエンド専用・精度上限枠**)
    ///
    /// Qwen3-Embedding の 4B 版(2560次元・36層・LastToken・Apache 2.0)。0.6B と同一アーキ。
    /// safetensors は分割保存で f16 約 8GB(CPU の matmul が bf16 非対応のため f16 で読む)。
    /// 用語集ベンチ v2 で **clusterF1 真ピーク 0.956(P=0.963・R=0.948・誤統合 7 件)** と全モデル
    /// 中で最高精度を既定閾値 70 で達成(v1 では P=1.000・誤統合 0 と「完璧」だったが、難化した v2 で
    /// `保証⇔保障` などを誤統合するようになり頭打ちが解消)。ただし推論は Candle CPU で ≈3.9 秒/語と
    /// 極めて低速・約 8GB RAM 必須のため、**GPU・バッチ・オフライン等で速度を許容できる精度
    /// 最優先用途向け**(`--model qwen3-4b`、詳細は `docs/benchmarks.md`)。
    Qwen3Embedding4B,
    /// Qwen/Qwen3-Embedding-8B(**Candle バックエンド専用・eval 用**)
    ///
    /// Qwen3-Embedding の 8B 版(4096次元・36層・LastToken・Apache 2.0)。4B と同一の読み込み
    /// 経路(分割 safetensors・f16)。f16 でも約 16GB RAM を要し推論はさらに低速。十分な RAM の
    /// 環境向けの検証用(詳細は `docs/benchmarks.md`)。
    Qwen3Embedding8B,
}

/// narashi が利用できる埋め込みモデルの選択
///
/// fastembed の組み込みモデル([`EmbeddingModel`])と、HF リポジトリから直接読み込む
/// [`UserModel`] の双方を統一的に扱う。`From<EmbeddingModel>` があるため
/// `with_model(EmbeddingModel::...)` のように組み込みモデルを直接渡せる。
#[derive(Debug, Clone)]
pub enum Model {
    /// fastembed の組み込みモデル(`onnx` 機能が必要)
    #[cfg(feature = "onnx")]
    Builtin(EmbeddingModel),
    /// HF リポジトリから読み込むユーザー定義モデル
    UserDefined(UserModel),
}

#[cfg(feature = "onnx")]
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

/// プーリング方式(バックエンド非依存。E5/MiniLM は Mean、BGE/GTE は CLS、
/// Qwen3 等のデコーダ系埋め込みは末尾(EOS)トークンの LastToken)
#[derive(Debug, Clone, Copy)]
enum Pool {
    /// 先頭(CLS)トークンの隠れ状態を用いる
    Cls,
    /// attention mask による加重平均を用いる
    Mean,
    /// 末尾(最後の実トークン=EOS)の隠れ状態を用いる(デコーダ系埋め込み)
    LastToken,
}

/// モデルの実行バックエンドと重みの所在
enum BackendKind {
    /// fastembed の組み込みモデル。fastembed 自身が ONNX を取得・管理する。
    #[cfg(feature = "onnx")]
    Builtin(EmbeddingModel),
    /// `ModelSpec::hf_repo` の ONNX を fastembed で直接読み込む(例: `"onnx/model.onnx"`)
    Onnx { weights_file: &'static str },
    /// `ModelSpec::hf_repo` の safetensors を Candle で直接読み込む(例: `"model.safetensors"`)
    Candle { weights_file: &'static str },
}

/// モデルごとの取り扱いを記述したメタ情報
///
/// モデルによって入力プレフィックスやコサイン類似度の分布が異なるため、
/// トークナイザ取得元・プレフィックス・スコア校正のベースライン・プーリング・
/// 実行バックエンドを切り替える。
struct ModelSpec {
    /// `tokenizer.json`(および重み)を取得する Hugging Face リポジトリ ID
    hf_repo: &'static str,
    /// 埋め込み入力に付与するプレフィックス(E5 系は `"query: "`、対称モデルは空)
    query_prefix: &'static str,
    /// スコア校正の基準となるコサイン値(無関係な短文ペアの典型的な下限)
    ///
    /// この値を 0、コサイン 1.0 を 100 に写像してスコアの識別力を高める。
    cos_baseline: f32,
    /// プーリング方式(`Builtin` は fastembed が内部で決めるため未使用)
    pooling: Pool,
    /// 実行バックエンドと重みファイル
    backend: BackendKind,
}

/// 指定モデルの取り扱いメタ情報を返す
fn model_spec(model: &Model) -> ModelSpec {
    match model {
        #[cfg(feature = "onnx")]
        Model::Builtin(m) => builtin_spec(m),
        Model::UserDefined(UserModel::GteMultilingualBase) => ModelSpec {
            hf_repo: "onnx-community/gte-multilingual-base",
            // GTE は STS/類似度用途では指示プレフィックス無しの対称利用。
            query_prefix: "",
            cos_baseline: 0.42,
            pooling: Pool::Cls,
            backend: BackendKind::Onnx {
                weights_file: "onnx/model.onnx",
            },
        },
        Model::UserDefined(UserModel::DistiluseMultilingualV2) => ModelSpec {
            hf_repo: "Xenova/distiluse-base-multilingual-cased-v2",
            query_prefix: "",
            // ピーク clusterF1 を既定閾値 70 に合わせる校正値(ベンチで決定)
            cos_baseline: 0.39,
            pooling: Pool::Mean,
            backend: BackendKind::Onnx {
                weights_file: "onnx/model.onnx",
            },
        },
        // IBM Granite Embedding 系(Apache 2.0・CLS プーリング・プレフィックス無し)。
        // cos_baseline は暫定。採用時にベンチで最適閾値が既定 70 に来るよう再校正する。
        Model::UserDefined(UserModel::GraniteMultilingual97mR2) => ModelSpec {
            hf_repo: "ibm-granite/granite-embedding-97m-multilingual-r2",
            query_prefix: "",
            cos_baseline: 0.42,
            pooling: Pool::Cls,
            backend: BackendKind::Onnx {
                weights_file: "onnx/model.onnx",
            },
        },
        Model::UserDefined(UserModel::GraniteMultilingual107m) => ModelSpec {
            hf_repo: "ibm-granite/granite-embedding-107m-multilingual",
            query_prefix: "",
            cos_baseline: 0.42,
            pooling: Pool::Cls,
            backend: BackendKind::Onnx {
                weights_file: "model.onnx",
            },
        },
        Model::UserDefined(UserModel::GraniteMultilingual278m) => ModelSpec {
            hf_repo: "ibm-granite/granite-embedding-278m-multilingual",
            query_prefix: "",
            // clusterF1 真ピークが既定閾値 70 に来るよう校正(ベンチで決定)
            cos_baseline: 0.44,
            pooling: Pool::Cls,
            backend: BackendKind::Onnx {
                weights_file: "model.onnx",
            },
        },
        Model::UserDefined(UserModel::GraniteMultilingual311mR2) => ModelSpec {
            hf_repo: "ibm-granite/granite-embedding-311m-multilingual-r2",
            query_prefix: "",
            cos_baseline: 0.42,
            pooling: Pool::Cls,
            backend: BackendKind::Onnx {
                weights_file: "onnx/model.onnx",
            },
        },
        // 大型・高精度候補(fp16 単一ファイル ONNX・CLS プーリング・プレフィックス無し)。
        // cos_baseline は暫定。採用時にベンチで最適閾値が既定 70 に来るよう再校正する。
        Model::UserDefined(UserModel::BgeM3) => ModelSpec {
            hf_repo: "Xenova/bge-m3",
            query_prefix: "",
            // clusterF1 真ピーク(cos≈0.72)が既定閾値 70 に来るよう校正(ベンチで決定)
            cos_baseline: 0.072,
            pooling: Pool::Cls,
            backend: BackendKind::Onnx {
                weights_file: "onnx/model_fp16.onnx",
            },
        },
        Model::UserDefined(UserModel::ArcticEmbedLV2) => ModelSpec {
            hf_repo: "Snowflake/snowflake-arctic-embed-l-v2.0",
            query_prefix: "",
            cos_baseline: 0.42,
            pooling: Pool::Cls,
            backend: BackendKind::Onnx {
                weights_file: "onnx/model_fp16.onnx",
            },
        },
        // Candle バックエンド専用。ONNX 変換が無い safetensors モデルを直接読み込む。
        Model::UserDefined(UserModel::E5LargeInstruct) => ModelSpec {
            hf_repo: "intfloat/multilingual-e5-large-instruct",
            // 指示対応 E5。対称類似度では同一の指示を両テキストに付与する。
            query_prefix: "Instruct: Retrieve semantically similar text.\nQuery: ",
            // ベンチで clusterF1 真ピーク(cos≈0.90)が既定閾値 70 に来るよう校正。
            // 真ピークは 0.645 と gte(0.657)未満かつ真ピーク時の誤統合 75 件と多いため
            // 既定には採用せず eval 用の選択肢に留める(詳細は docs/benchmarks.md)。
            cos_baseline: 0.67,
            pooling: Pool::Mean,
            backend: BackendKind::Candle {
                weights_file: "model.safetensors",
            },
        },
        // Candle バックエンド専用。Qwen3 デコーダ + last-token プーリング。
        Model::UserDefined(UserModel::Qwen3Embedding0_6B) => ModelSpec {
            hf_repo: "Qwen/Qwen3-Embedding-0.6B",
            // Qwen3-Embedding 公式のクエリ書式。対称類似度では両テキストに同一指示を付与。
            query_prefix: "Instruct: Retrieve semantically similar text.\nQuery:",
            // ベンチで clusterF1 真ピーク(cos≈0.91)が既定閾値 70 に来るよう校正。
            cos_baseline: 0.69,
            pooling: Pool::LastToken,
            backend: BackendKind::Candle {
                weights_file: "model.safetensors",
            },
        },
        // Candle バックエンド専用・精度上限枠。4B 版(分割 safetensors・f16・約 8GB)。
        Model::UserDefined(UserModel::Qwen3Embedding4B) => ModelSpec {
            hf_repo: "Qwen/Qwen3-Embedding-4B",
            query_prefix: "Instruct: Retrieve semantically similar text.\nQuery:",
            // ベンチ v2 で clusterF1 真ピーク 0.956 が既定閾値 70 に来ることを確認済み。
            cos_baseline: 0.69,
            pooling: Pool::LastToken,
            backend: BackendKind::Candle {
                // 4B は分割保存。単一 `model.safetensors` が無ければ index.json から
                // 全シャードを解決する(resolve_candle_weights)。
                weights_file: "model.safetensors",
            },
        },
        // Candle バックエンド専用・eval 用。8B 版(分割 safetensors・f16・約 16GB RAM)。
        Model::UserDefined(UserModel::Qwen3Embedding8B) => ModelSpec {
            hf_repo: "Qwen/Qwen3-Embedding-8B",
            query_prefix: "Instruct: Retrieve semantically similar text.\nQuery:",
            // 暫定。十分な RAM の環境でベンチして既定閾値 70 に来るよう再校正する。
            cos_baseline: 0.69,
            pooling: Pool::LastToken,
            backend: BackendKind::Candle {
                weights_file: "model.safetensors",
            },
        },
    }
}

/// fastembed 組み込みモデルの取り扱いメタ情報を返す
#[cfg(feature = "onnx")]
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
        // 組み込みモデルのプーリングは fastembed が内部で決めるため未使用。
        pooling: Pool::Mean,
        backend: BackendKind::Builtin(model.clone()),
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
    embedder: Embedder,
    tokenizer: Tokenizer,
    /// 埋め込み入力に付与するプレフィックス(モデル依存)
    query_prefix: &'static str,
    /// スコア校正の基準コサイン値(モデル依存)
    cos_baseline: f32,
}

/// 実行時の埋め込み器(バックエンドごとの実体)
enum Embedder {
    /// fastembed(ONNX Runtime)
    #[cfg(feature = "onnx")]
    Onnx(TextEmbedding),
    /// Candle(ピュア Rust)
    #[cfg(feature = "candle")]
    Candle(candle_backend::CandleEmbedder),
}

impl Embedder {
    /// プレフィックス付与済みのテキスト群を埋め込む
    fn embed(&self, inputs: Vec<String>) -> Result<Vec<Vec<f32>>> {
        match self {
            #[cfg(feature = "onnx")]
            Embedder::Onnx(e) => Ok(e.embed(inputs, None)?),
            #[cfg(feature = "candle")]
            Embedder::Candle(e) => e.embed(&inputs),
        }
    }
}

/// ユーザー定義 ONNX モデルを fastembed で読み込む
#[cfg(feature = "onnx")]
fn build_onnx_embedder(repo: &ApiRepo, weights_file: &str, pool: Pool) -> Result<Embedder> {
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
    let pooling = match pool {
        Pool::Cls => Pooling::Cls,
        Pool::Mean => Pooling::Mean,
        // fastembed(ONNX)は last-token プーリング非対応。LastToken を要するモデルは
        // Candle バックエンド側で扱うため、ここに到達することはない。
        Pool::LastToken => {
            return Err(anyhow!(
                "last-token プーリングは ONNX バックエンドでは非対応です(Candle が必要)"
            ));
        }
    };
    let user_model =
        UserDefinedEmbeddingModel::new(fetch(weights_file)?, tokenizer_files).with_pooling(pooling);
    Ok(Embedder::Onnx(TextEmbedding::try_new_from_user_defined(
        user_model,
        InitOptionsUserDefined::new(),
    )?))
}

/// safetensors モデルを Candle で読み込む
#[cfg(feature = "candle")]
fn build_candle_embedder(repo: &ApiRepo, weights_file: &str, pool: Pool) -> Result<Embedder> {
    let fetch = |name: &str| -> Result<Vec<u8>> {
        let path = repo
            .get(name)
            .map_err(|e| anyhow!("{name} download failed: {e}"))?;
        Ok(std::fs::read(path)?)
    };
    Ok(Embedder::Candle(candle_backend::CandleEmbedder::new(
        &fetch("config.json")?,
        &fetch("tokenizer.json")?,
        &resolve_candle_weights(repo, weights_file)?,
        pool,
    )?))
}

/// Candle 用の safetensors の重みパスを解決する(単一ファイル or 分割シャード)
///
/// まず単一ファイル(`weights_file`)を試し、無ければ `model.safetensors.index.json` を
/// 読んで全シャードを取得する(4B/8B のような大型モデルは分割保存される)。
#[cfg(feature = "candle")]
fn resolve_candle_weights(repo: &ApiRepo, weights_file: &str) -> Result<Vec<PathBuf>> {
    if let Ok(p) = repo.get(weights_file) {
        return Ok(vec![p]);
    }
    let index = repo
        .get("model.safetensors.index.json")
        .map_err(|e| anyhow!("{weights_file} も index.json も取得できません: {e}"))?;
    let json: serde_json::Value = serde_json::from_slice(&std::fs::read(index)?)?;
    let map = json
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| anyhow!("index.json に weight_map がありません"))?;
    // シャード名の重複を除いて順に取得する。
    let mut shards: Vec<String> = map
        .values()
        .filter_map(|v| v.as_str().map(str::to_string))
        .collect();
    shards.sort();
    shards.dedup();
    shards
        .iter()
        .map(|name| {
            repo.get(name)
                .map_err(|e| anyhow!("{name} download failed: {e}"))
        })
        .collect()
}

impl Narashi {
    /// デフォルト設定で初期化する
    ///
    /// 必要に応じてモデル・トークナイザをダウンロードします(初回のみ、既定の bge-m3 は約 1.06GB)。
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

        // 埋め込みモデルを読み込む。バックエンドに応じて取得元・読み込み方法を切り替える。
        // 必要な機能が無効な場合はその旨を伝えて中断する。
        let embedder = match spec.backend {
            #[cfg(feature = "onnx")]
            BackendKind::Builtin(m) => Embedder::Onnx(TextEmbedding::try_new(
                InitOptions::new(m).with_cache_dir(cache_dir.clone()),
            )?),
            BackendKind::Onnx { weights_file } => {
                #[cfg(feature = "onnx")]
                {
                    build_onnx_embedder(&repo, weights_file, spec.pooling)?
                }
                #[cfg(not(feature = "onnx"))]
                {
                    let _ = weights_file;
                    return Err(anyhow!(
                        "モデル '{}' は ONNX バックエンドが必要です(`--features onnx` を有効にして再ビルドしてください)",
                        spec.hf_repo
                    ));
                }
            }
            BackendKind::Candle { weights_file } => {
                #[cfg(feature = "candle")]
                {
                    build_candle_embedder(&repo, weights_file, spec.pooling)?
                }
                #[cfg(not(feature = "candle"))]
                {
                    let _ = weights_file;
                    return Err(anyhow!(
                        "モデル '{}' は Candle バックエンドが必要です(`--features candle` を有効にして再ビルドしてください)",
                        spec.hf_repo
                    ));
                }
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
        let mut embeddings = self.embedder.embed(inputs)?;
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
        // 既定モデル bge-m3 では「猫」⇔「ネコ」≒65、「猫」⇔「自動車」≒46 と分離する
        // (bge-m3 はコサインが高帯域に圧縮されるぶん gte より絶対差は小さいが順序は明確)。
        let related = n.similarity("猫", "ネコ").unwrap();
        let unrelated = n.similarity("猫", "自動車").unwrap();
        assert!(
            related > 60.0 && related > unrelated + 15.0,
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

    /// Candle バックエンドが safetensors から埋め込みを生成し、類義語ペアを
    /// 非類義ペアより明確に高くスコアリングすることを確認する。
    ///
    /// ネットワーク不要。ローカルに `config.json` / `tokenizer.json` /
    /// `model.safetensors` を置き、`NARASHI_TEST_MODEL_DIR` でそのディレクトリを
    /// 指定して `cargo test --features candle -- --ignored` で実行する。
    #[cfg(feature = "candle")]
    #[test]
    #[ignore]
    fn candle_embedder_separates_synonyms() {
        let dir = std::env::var("NARASHI_TEST_MODEL_DIR")
            .expect("NARASHI_TEST_MODEL_DIR を safetensors モデルのディレクトリに設定してください");
        let p = Path::new(&dir);
        let cfg = std::fs::read(p.join("config.json")).unwrap();
        let tok = std::fs::read(p.join("tokenizer.json")).unwrap();
        let emb = crate::candle_backend::CandleEmbedder::new(
            &cfg,
            &tok,
            &[p.join("model.safetensors")],
            Pool::Mean,
        )
        .unwrap();
        let prefix = "Instruct: Retrieve semantically similar text.\nQuery: ";
        let mk = |s: &str| format!("{prefix}{s}");
        let texts = vec![mk("白い背景"), mk("白背景"), mk("頬紅"), mk("照れ")];
        let mut v = emb.embed(&texts).unwrap();
        for x in v.iter_mut() {
            normalize_l2(x);
        }
        let synonym = dot(&v[0], &v[1]);
        let unrelated = dot(&v[0], &v[2]);
        assert!(
            synonym > unrelated + 0.05,
            "synonym pair ({synonym}) should clearly exceed unrelated ({unrelated})"
        );
    }
}
