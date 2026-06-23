//! Candle(ピュア Rust)による埋め込みバックエンド。
//!
//! ONNX Runtime を介さず、Hugging Face の safetensors 重みを直接読み込んで
//! XLM-RoBERTa 系の埋め込みモデルを実行する。ネイティブの ONNX Runtime バイナリを
//! 取得できない環境でも動作し、ONNX 変換版が公開されていないモデル
//! (例: [`crate::UserModel::E5LargeInstruct`])も利用可能にする。

use crate::Pool;
use anyhow::{Result, anyhow};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::xlm_roberta::{Config, XLMRobertaModel};
use std::path::Path;
use tokenizers::Tokenizer;

/// safetensors の XLM-RoBERTa を Candle で実行する埋め込み器
pub(crate) struct CandleEmbedder {
    model: XLMRobertaModel,
    tokenizer: Tokenizer,
    device: Device,
    pool: Pool,
}

impl CandleEmbedder {
    /// `config.json` / `tokenizer.json` のバイト列と safetensors のパスから初期化する
    ///
    /// 重みは fp16/fp32 いずれでも mmap して f32 として読み込む。
    pub(crate) fn new(
        config_json: &[u8],
        tokenizer_json: &[u8],
        weights: &Path,
        pool: Pool,
    ) -> Result<Self> {
        let device = Device::Cpu;
        let cfg: Config = serde_json::from_slice(config_json)
            .map_err(|e| anyhow!("config.json parse failed: {e}"))?;
        let tokenizer = Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| anyhow!("tokenizer load failed: {e}"))?;
        // safetensors を mmap して f32 で読み込む(fp16 重みは自動で昇格)。
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)? };
        let model = XLMRobertaModel::new(&cfg, vb)?;
        Ok(Self {
            model,
            tokenizer,
            device,
            pool,
        })
    }

    /// テキスト群を埋め込む(プーリング済み・L2 正規化前)
    ///
    /// 正規化は呼び出し側([`crate::Narashi::embed_normalized`])で行うため、
    /// ここではプーリング結果をそのまま返す。
    pub(crate) fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed_one(t)).collect()
    }

    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
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
        let hs = self
            .model
            .forward(&input_ids, &attn, &ttype, None, None, None)?;
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
        };
        Ok(pooled.to_vec1()?)
    }
}
