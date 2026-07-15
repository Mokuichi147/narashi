//! OpenAI Embeddings API バックエンド。
//!
//! ローカル推論を行わず、`/v1/embeddings` エンドポイントへ HTTPS でテキストを送って
//! 埋め込みを得る。モデルの重みダウンロードが不要な代わりにネットワーク接続が必要で、
//! テキストは接続先へ送信される点に注意。`OPENAI_BASE_URL`([`BASE_URL_ENV`])を設定すれば
//! OpenAI 互換 API(Azure OpenAI 互換ゲートウェイ・Ollama/LM Studio 等のローカルサーバ)にも
//! 向けられる。**既定の OpenAI 本家エンドポイントへ接続するときのみ API キー
//! ([`API_KEY_ENV`])を必須とする**。`OPENAI_BASE_URL` でエンドポイントを差し替えた
//! ときはキー無しでも初期化でき、指定されていれば従来どおり `Authorization` ヘッダに
//! 載せる(キー不要なローカルサーバー向け)。

use anyhow::{Result, anyhow, bail};

/// OpenAI API キーを渡す環境変数名(`Options::with_openai_api_key` が優先)
pub const API_KEY_ENV: &str = "OPENAI_API_KEY";

/// API のベース URL を上書きする環境変数名(既定: `https://api.openai.com/v1`)
///
/// OpenAI 互換の埋め込み API(プロキシ・ローカルサーバ等)へ向けるときに使う。
/// 末尾の `/` は無視され、`{BASE_URL}/embeddings` へ POST する。
pub const BASE_URL_ENV: &str = "OPENAI_BASE_URL";

const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// 1 リクエストに載せる最大テキスト数。API 上限(2048 入力)より安全側に取り、
/// 1 リクエストの巨大化(タイムアウト・トークン上限)も避ける。
const BATCH_SIZE: usize = 512;

/// OpenAI Embeddings API(または互換 API)を呼び出す埋め込み器
pub(crate) struct OpenAiEmbedder {
    agent: ureq::Agent,
    /// `{base_url}/embeddings` まで解決済みのエンドポイント
    endpoint: String,
    /// 未設定なら `Authorization` ヘッダを送らない(キー不要なローカルサーバー向け)
    api_key: Option<String>,
    /// API に渡すモデル名(例: `text-embedding-3-small`)
    model: String,
}

impl OpenAiEmbedder {
    /// API キー・エンドポイントを解決して初期化する
    ///
    /// キーは 明示指定 > 環境変数 [`API_KEY_ENV`] の順、エンドポイントは
    /// 明示指定 > 環境変数 [`BASE_URL_ENV`] > 既定(OpenAI 本家)の順で解決する。
    /// **既定の OpenAI 本家エンドポイントに接続する場合のみキーを必須とする**。
    /// 別エンドポイント(ローカルサーバー等)に向けている場合はキー無しでも初期化でき、
    /// 指定されていればそのまま `Authorization: Bearer` ヘッダへ載せる。
    pub(crate) fn new(
        model: String,
        api_key: Option<String>,
        base_url: Option<String>,
    ) -> Result<Self> {
        let api_key = api_key
            .or_else(|| std::env::var(API_KEY_ENV).ok())
            .filter(|k| !k.trim().is_empty());
        let base_url = base_url
            .filter(|s| !s.trim().is_empty())
            .or_else(|| std::env::var(BASE_URL_ENV).ok())
            .filter(|s| !s.trim().is_empty());
        let is_default_endpoint = base_url.is_none();
        let base_url = base_url.unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        if is_default_endpoint && api_key.is_none() {
            return Err(anyhow!(
                "OpenAI API キーが見つかりません(環境変数 {API_KEY_ENV} を設定するか、\
                 Options::with_openai_api_key で指定してください)。\
                 キー不要なローカルサーバー等を使う場合は {BASE_URL_ENV} でエンドポイントを指定してください"
            ));
        }
        Ok(Self {
            agent: ureq::AgentBuilder::new().build(),
            endpoint: format!("{}/embeddings", base_url.trim_end_matches('/')),
            api_key,
            model,
        })
    }

    /// テキスト群を埋め込む(API 上限に収まるようバッチ分割して逐次リクエスト)
    pub(crate) fn embed(&self, inputs: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(inputs.len());
        for chunk in inputs.chunks(BATCH_SIZE) {
            out.extend(self.embed_batch(chunk)?);
        }
        Ok(out)
    }

    /// 1 バッチ分を `/embeddings` へ POST し、入力順の埋め込みを返す
    fn embed_batch(&self, chunk: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut req = self.agent.post(&self.endpoint);
        if let Some(key) = &self.api_key {
            req = req.set("Authorization", &format!("Bearer {key}"));
        }
        let resp = req
            .send_json(serde_json::json!({
                "model": &self.model,
                "input": chunk,
            }))
            .map_err(|e| match e {
                // エラー応答の本文(エラー理由の JSON)を含めて返す
                ureq::Error::Status(code, resp) => {
                    let body = resp.into_string().unwrap_or_default();
                    anyhow!("OpenAI API エラー (HTTP {code}): {body}")
                }
                e => anyhow!("OpenAI API リクエストに失敗しました: {e}"),
            })?;
        let json: serde_json::Value = resp.into_json()?;
        let data = json
            .get("data")
            .and_then(|v| v.as_array())
            .ok_or_else(|| anyhow!("OpenAI API 応答に data 配列がありません"))?;
        if data.len() != chunk.len() {
            bail!(
                "OpenAI API 応答の埋め込み数({})が入力数({})と一致しません",
                data.len(),
                chunk.len()
            );
        }
        // 仕様上は入力順で返るが、各要素の `index` を尊重して並べ直す。
        let mut rows: Vec<Vec<f32>> = vec![Vec::new(); chunk.len()];
        for item in data {
            let idx = item
                .get("index")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow!("OpenAI API 応答の要素に index がありません"))?
                as usize;
            let embedding = item
                .get("embedding")
                .and_then(|v| v.as_array())
                .ok_or_else(|| anyhow!("OpenAI API 応答の要素に embedding がありません"))?
                .iter()
                .map(|x| {
                    x.as_f64()
                        .map(|f| f as f32)
                        .ok_or_else(|| anyhow!("embedding に数値でない要素があります"))
                })
                .collect::<Result<Vec<f32>>>()?;
            let slot = rows
                .get_mut(idx)
                .ok_or_else(|| anyhow!("OpenAI API 応答の index {idx} が入力範囲外です"))?;
            *slot = embedding;
        }
        Ok(rows)
    }
}
