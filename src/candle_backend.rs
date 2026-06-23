//! Candle(ピュア Rust)による埋め込みバックエンド。
//!
//! ONNX Runtime を介さず、Hugging Face の safetensors 重みを直接読み込んで埋め込み
//! モデルを実行する。ネイティブの ONNX Runtime バイナリを取得できない環境でも動作し、
//! ONNX 変換版が公開されていない/配布形式が ONNX で扱えないモデルも利用可能にする。
//!
//! `config.json` の `model_type` でアーキテクチャを判定して読み分ける:
//! - `xlm-roberta`: エンコーダ。CLS/Mean プーリング(例: [`crate::UserModel::E5LargeInstruct`])
//! - `qwen3`: デコーダ。末尾(EOS)トークンの last-token プーリング
//!   (例: [`crate::UserModel::Qwen3Embedding0_6B`])

use crate::Pool;
use anyhow::{Result, anyhow, bail};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3::{Config as Qwen3Config, Model as Qwen3Model};
use candle_transformers::models::xlm_roberta::{Config as XlmConfig, XLMRobertaModel};
use std::collections::HashMap;
use tokenizers::Tokenizer;

/// Qwen3 の計算精度をモデルサイズで選ぶ。
///
/// candle の CPU バックエンドは **bf16 の matmul 非対応**なので bf16 は使えない。
/// 小型(0.6B・hidden 1024)は **f32** で高速に回す。大型(4B:hidden 2560・8B:4096)は
/// f32 だと RAM を超える(4B で約 16GB・8B で約 32GB)ため **f16** で読む(約半分の 8GB/16GB)。
/// 0.6B では f32 と f16 で clusterF1 が一致することを確認済み(f16 は CPU で約 3 倍遅いだけ)。
fn qwen3_dtype(cfg: &Qwen3Config) -> DType {
    if cfg.hidden_size <= 1024 {
        DType::F32
    } else {
        DType::F16
    }
}

/// 対応するモデルアーキテクチャと、その重みの保持方法
enum Backend {
    /// XLM-RoBERTa(エンコーダ)。重みは初期化時に常駐させる。
    XlmRoberta(XLMRobertaModel),
    /// Qwen3(デコーダ)。`Model` の KvCache はリセット API が非公開で `forward` を
    /// 繰り返すとキャッシュが連結されてしまう一方、candle 0.9 の CPU バックエンドは
    /// バッチ(>1)+因果マスクの broadcast で添字エラーになる。そこで **1 件ずつ
    /// (バッチ=1)** 処理し、各件でモデルを作り直して(=まっさらなキャッシュで 1 回だけ
    /// 前向き計算)回避する。重みは初回に [`qwen3_dtype`] の精度へ変換・`model.` 接頭辞
    /// 付けして常駐させ、都度の作り直しは Arc 共有のため安価(I/O や変換なし)。
    Qwen3 {
        cfg: Qwen3Config,
        tensors: HashMap<String, Tensor>,
    },
}

/// safetensors の埋め込みモデルを Candle で実行する埋め込み器
pub(crate) struct CandleEmbedder {
    backend: Backend,
    tokenizer: Tokenizer,
    device: Device,
    pool: Pool,
}

impl CandleEmbedder {
    /// `config.json` / `tokenizer.json` のバイト列と safetensors のパス群から初期化する
    ///
    /// `config.json` の `model_type` でアーキテクチャを判定する。`weights` は分割保存
    /// (シャード)に対応するためパスの slice。XLM-RoBERTa は f32 へ昇格、Qwen3 は
    /// [`qwen3_dtype`] が選ぶ精度(0.6B は f32、4B/8B は RAM 削減のため f16)で読む。
    pub(crate) fn new(
        config_json: &[u8],
        tokenizer_json: &[u8],
        weights: &[std::path::PathBuf],
        pool: Pool,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| anyhow!("tokenizer load failed: {e}"))?;
        // `model_type` でアーキテクチャを判定する(serde derive を増やさず Value で読む)。
        let probe: serde_json::Value = serde_json::from_slice(config_json)
            .map_err(|e| anyhow!("config.json parse failed: {e}"))?;
        let model_type = probe
            .get("model_type")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("config.json に model_type がありません"))?;

        let backend = match model_type {
            "xlm-roberta" => {
                let cfg: XlmConfig = serde_json::from_slice(config_json)
                    .map_err(|e| anyhow!("config.json parse failed: {e}"))?;
                let vb =
                    unsafe { VarBuilder::from_mmaped_safetensors(weights, DType::F32, &device)? };
                Backend::XlmRoberta(XLMRobertaModel::new(&cfg, vb)?)
            }
            "qwen3" => {
                let cfg: Qwen3Config = serde_json::from_slice(config_json)
                    .map_err(|e| anyhow!("config.json parse failed: {e}"))?;
                // Qwen3-Embedding の safetensors は `embed_tokens` / `layers` / `norm` のように
                // `model.` 接頭辞無しで重みを保存しているが、candle の `Model::new` は `model.`
                // 接頭辞を前提とする。読み込み後にキーへ `model.` を付けて常駐。重みは複数
                // シャード(4B/8B)に分かれることがあるので全シャードをマージする。
                let dtype = qwen3_dtype(&cfg);
                let mut tensors = HashMap::new();
                for shard in weights {
                    for (k, v) in candle_core::safetensors::load(shard, &device)? {
                        tensors.insert(format!("model.{k}"), v.to_dtype(dtype)?);
                    }
                }
                Backend::Qwen3 { cfg, tensors }
            }
            other => {
                bail!("未対応のアーキテクチャです: model_type={other}(対応: xlm-roberta / qwen3)")
            }
        };
        Ok(Self {
            backend,
            tokenizer,
            device,
            pool,
        })
    }

    /// テキスト群を埋め込む(プーリング済み・L2 正規化前)
    ///
    /// 正規化は呼び出し側([`crate::Narashi::embed_normalized`])で行う。
    pub(crate) fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match &self.backend {
            Backend::XlmRoberta(model) => texts.iter().map(|t| self.embed_xlm(model, t)).collect(),
            Backend::Qwen3 { cfg, tensors } => texts
                .iter()
                .map(|t| self.embed_qwen3(cfg, tensors, t))
                .collect(),
        }
    }

    /// XLM-RoBERTa(エンコーダ)を 1 件ずつ埋め込む(CLS / Mean プーリング)
    fn embed_xlm(&self, model: &XLMRobertaModel, text: &str) -> Result<Vec<f32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenize failed: {e}"))?;
        let ids = enc.get_ids().to_vec();
        let mask = enc.get_attention_mask().to_vec();
        let n = ids.len();
        let input_ids = Tensor::from_vec(ids, (1, n), &self.device)?;
        let attn = Tensor::from_vec(mask.clone(), (1, n), &self.device)?;
        // XLM-RoBERTa の token_type は常に 0(単一系列)。
        let ttype = Tensor::zeros((1, n), DType::U32, &self.device)?;
        // 最終層の隠れ状態 [1, n, hidden]
        let hs = model.forward(&input_ids, &attn, &ttype, None, None, None)?;
        let pooled = match self.pool {
            // CLS プーリング: 先頭トークンの隠れ状態
            Pool::Cls => hs.i((0, 0))?,
            // Mean プーリング: attention mask で加重平均
            Pool::Mean => {
                let maskf = Tensor::from_vec(
                    mask.iter().map(|&m| m as f32).collect::<Vec<_>>(),
                    (1, n, 1),
                    &self.device,
                )?;
                let summed = hs.broadcast_mul(&maskf)?.sum(1)?; // [1, hidden]
                let cnt = maskf.sum(1)?; // [1, 1]
                summed.broadcast_div(&cnt)?.squeeze(0)? // [hidden]
            }
            Pool::LastToken => bail!("XLM-RoBERTa では last-token プーリングは未使用です"),
        };
        Ok(pooled.to_vec1()?)
    }

    /// Qwen3(デコーダ)を 1 件埋め込む(last-token プーリング)
    ///
    /// バッチ=1。tokenizer が末尾へ付与する EOS の隠れ状態を埋め込みとする。
    /// 常駐済みの重み(Arc 共有)からモデルを作り直すため、まっさらな KvCache で 1 度だけ
    /// 前向き計算でき、キャッシュ連結も candle のバッチ broadcast バグも避けられる。
    fn embed_qwen3(
        &self,
        cfg: &Qwen3Config,
        tensors: &HashMap<String, Tensor>,
        text: &str,
    ) -> Result<Vec<f32>> {
        debug_assert!(matches!(self.pool, Pool::LastToken));
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenize failed: {e}"))?;
        let ids = enc.get_ids().to_vec();
        let n = ids.len();
        let input = Tensor::from_vec(ids, (1, n), &self.device)?;

        // Arc 共有のクローン(I/O・変換なし)。新しい Model = まっさらな KvCache。
        let vb = VarBuilder::from_tensors(tensors.clone(), qwen3_dtype(cfg), &self.device);
        let mut model = Qwen3Model::new(cfg, vb)?;
        // 最終層の隠れ状態 [1, n, hidden] の末尾トークン(= EOS)。
        // 計算は f16(CPU の matmul が bf16 非対応のため)なので f32 へ戻して返す。
        let hs = model.forward(&input, 0)?;
        let pooled = hs.i((0, n - 1))?.to_dtype(DType::F32)?;
        Ok(pooled.to_vec1()?)
    }
}
