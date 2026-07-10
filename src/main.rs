use anyhow::{Result, anyhow};
use clap::{Parser, ValueEnum};
#[cfg(feature = "onnx")]
use narashi::EmbeddingModel;
use narashi::{DEFAULT_THRESHOLD, Group, Language, Model, Narashi, Options, UserModel};
use std::path::PathBuf;

/// `--prefer-lang` の各要素を [`Language`] に変換する(`ja` / `zh` / `ko` / `en` など)
fn parse_language(code: &str) -> Result<Language, String> {
    Language::from_code(code)
        .ok_or_else(|| format!("不明な言語コード: '{code}'(ja / zh / ko / en を指定してください)"))
}

/// CLI から選択できる埋め込みモデル
///
/// ONNX バックエンドのモデルは `onnx` 機能、Candle バックエンドのモデルは `candle` 機能が
/// 有効なビルドでのみ選択できる(既定ビルドは両方有効)。
#[derive(Copy, Clone, Debug, ValueEnum)]
enum ModelArg {
    /// bge-m3 (既定・ONNX勢でclusterF1最高0.699・誤統合も最小7件・1024次元・約1.06GB・約3倍低速)
    #[cfg(feature = "onnx")]
    BgeM3,
    /// gte-multilingual-base (高適合率・CJKに強い・速度重視の代替・768次元・約1.2GB)
    #[cfg(feature = "onnx")]
    Gte,
    /// granite-278m-multilingual (clusterF1高め0.682だが誤統合が多め28件・日本語明示学習・768次元・約1.1GB)
    #[cfg(feature = "onnx")]
    Granite,
    /// distiluse-multilingual-v2 (軽量代替・高適合率・約0.54GB)
    #[cfg(feature = "onnx")]
    Distiluse,
    /// 多言語 E5 small (高適合率かつ最速級・軽量 約0.45GB)
    #[cfg(feature = "onnx")]
    Small,
    /// 多言語 E5 large (E5系の上限・約8倍低速)
    #[cfg(feature = "onnx")]
    Large,
    /// 多言語 E5 base (small に劣後・非推奨)
    #[cfg(feature = "onnx")]
    Base,
    /// paraphrase-multilingual-MiniLM-L12-v2 (高再現率・要 高め閾値)
    #[cfg(feature = "onnx")]
    Paraphrase,
    /// paraphrase-multilingual-mpnet-base-v2 (再現率最優先・要 高め閾値)
    #[cfg(feature = "onnx")]
    Mpnet,
    /// paraphrase-multilingual-MiniLM-L12-v2 量子化版 (高速)
    #[cfg(feature = "onnx")]
    ParaphraseQ,
    /// multilingual-e5-large-instruct (Candle・ONNX非依存環境向け・clusterF1 0.645で誤統合最多75件・低速・1024次元)
    #[cfg(feature = "candle")]
    E5Instruct,
    /// Qwen3-Embedding-0.6B (Candle・clusterF1 0.764でbge-m3超だが暴走オンセット94で安全運用点が無く軽量枠・1024次元)
    #[cfg(feature = "candle")]
    Qwen3,
    /// Qwen3-Embedding-4B (Candle単独ビルドの既定・clusterF1 0.956・暴走オンセット82で@83 R≈0.75・f16でGPU推奨・2560次元)
    #[cfg(feature = "candle")]
    #[value(name = "qwen3-4b")]
    Qwen34b,
    /// Qwen3-Embedding-8B (Candle・eval用・約16GB RAM 必須・さらに低速)
    #[cfg(feature = "candle")]
    #[value(name = "qwen3-8b")]
    Qwen38b,
    /// OpenAI text-embedding-3-small (API・要 OPENAI_API_KEY・1536次元・テキストはOpenAIへ送信)
    #[cfg(feature = "openai")]
    #[value(name = "openai-small")]
    OpenaiSmall,
    /// OpenAI text-embedding-3-large (API・要 OPENAI_API_KEY・3072次元・テキストはOpenAIへ送信)
    #[cfg(feature = "openai")]
    #[value(name = "openai-large")]
    OpenaiLarge,
}

/// 既定モデル名(`--model` の既定値文字列)。ONNX が有効なら bge-m3、Candle のみなら
/// Qwen3-Embedding-4B(Candle 勢では暴走オンセット 82・安全運用点 @83 で R≈0.75 と
/// 堅牢性 × 再現率のバランスが最良)。値は `ModelArg` の `value_enum` 名(kebab-case)と一致させる。
#[cfg(feature = "onnx")]
const DEFAULT_MODEL_NAME: &str = "bge-m3";
#[cfg(all(not(feature = "onnx"), feature = "candle"))]
const DEFAULT_MODEL_NAME: &str = "qwen3-4b";
// OpenAI 単独ビルドではローカル推論が無いため API モデルを既定にする。
#[cfg(all(not(feature = "onnx"), not(feature = "candle"), feature = "openai"))]
const DEFAULT_MODEL_NAME: &str = "openai-small";

impl From<ModelArg> for Model {
    fn from(m: ModelArg) -> Self {
        match m {
            #[cfg(feature = "onnx")]
            ModelArg::Small => EmbeddingModel::MultilingualE5Small.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Base => EmbeddingModel::MultilingualE5Base.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Large => EmbeddingModel::MultilingualE5Large.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Paraphrase => EmbeddingModel::ParaphraseMLMiniLML12V2.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Mpnet => EmbeddingModel::ParaphraseMLMpnetBaseV2.into(),
            #[cfg(feature = "onnx")]
            ModelArg::ParaphraseQ => EmbeddingModel::ParaphraseMLMiniLML12V2Q.into(),
            #[cfg(feature = "onnx")]
            ModelArg::BgeM3 => UserModel::BgeM3.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Gte => UserModel::GteMultilingualBase.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Granite => UserModel::GraniteMultilingual278m.into(),
            #[cfg(feature = "onnx")]
            ModelArg::Distiluse => UserModel::DistiluseMultilingualV2.into(),
            #[cfg(feature = "candle")]
            ModelArg::E5Instruct => UserModel::E5LargeInstruct.into(),
            #[cfg(feature = "candle")]
            ModelArg::Qwen3 => UserModel::Qwen3Embedding0_6B.into(),
            #[cfg(feature = "candle")]
            ModelArg::Qwen34b => UserModel::Qwen3Embedding4B.into(),
            #[cfg(feature = "candle")]
            ModelArg::Qwen38b => UserModel::Qwen3Embedding8B.into(),
            #[cfg(feature = "openai")]
            ModelArg::OpenaiSmall => UserModel::OpenAiTextEmbedding3Small.into(),
            #[cfg(feature = "openai")]
            ModelArg::OpenaiLarge => UserModel::OpenAiTextEmbedding3Large.into(),
        }
    }
}

#[derive(Parser)]
#[command(name = "narashi", about = "表記ゆれ解消ツール")]
struct Cli {
    /// 類似度の閾値 (0-100)
    #[arg(short, long, default_value_t = DEFAULT_THRESHOLD)]
    threshold: f32,

    /// 使用する埋め込みモデル
    ///
    /// 既定はカタログ名(bge-m3 / gte / ... / openai-small / openai-large、選択肢は
    /// `--help` 参照)から選ぶ。**`--openai-base-url` を指定した場合は、この値を
    /// カタログ名としてではなく OpenAI 互換 API にそのまま渡すモデル名として使う**
    /// (ローカルサーバーが独自名で提供するモデルを指定できる。ローカル推論やモデルの
    /// ダウンロードは行わない)。
    #[arg(long, default_value = DEFAULT_MODEL_NAME)]
    model: String,

    /// モデルキャッシュの保存先 (既定: OSのTEMPフォルダ下)
    #[arg(long, env = "NARASHI_CACHE_DIR")]
    cache_dir: Option<PathBuf>,

    /// 代表として優先して残す言語の順位 (カンマ区切り: ja,zh,ko,en)
    ///
    /// 例: `--prefer-lang ja,zh` なら、異言語が統合されたとき日本語→中国語の順で代表を残す。
    /// 未指定なら言語優先なし(従来どおり汎用性スコアのみで代表を決める)。
    #[arg(long, value_delimiter = ',', value_parser = parse_language)]
    prefer_lang: Vec<Language>,

    /// OpenAI API キー(`--model openai-small` / `openai-large` 使用時。既定の OpenAI
    /// エンドポイントに接続する場合のみ必須。`--openai-base-url` でローカルサーバー等に
    /// 向ける場合は不要)
    #[cfg(feature = "openai")]
    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,

    /// OpenAI 互換 API のエンドポイント(既定: `https://api.openai.com/v1`)。
    /// キー不要なローカルサーバー(Ollama / LM Studio 等)へ向けるときに指定する
    #[cfg(feature = "openai")]
    #[arg(long, env = "OPENAI_BASE_URL")]
    openai_base_url: Option<String>,

    /// 比較するテキスト (2つ以上)
    #[arg(required = true, num_args = 2..)]
    texts: Vec<String>,
}

/// `--model` の値を解決する。
///
/// `openai_mode`(`--openai-base-url` 指定時)なら値をカタログ名としてではなく、
/// OpenAI 互換 API へそのまま渡すモデル名として扱う(ローカルサーバーが独自名で
/// 提供するモデルを指定できる。ローカル推論やモデルのダウンロードは行わない)。
/// 利便性のため、カタログの `openai-small` / `openai-large` だけは実際の OpenAI
/// API モデル名へ変換する。`openai_mode` が偽ならカタログ名として解釈する。
fn resolve_model(name: &str, openai_mode: bool) -> Result<Model, String> {
    #[cfg(feature = "openai")]
    if openai_mode {
        let literal = match name {
            "openai-small" => "text-embedding-3-small",
            "openai-large" => "text-embedding-3-large",
            other => other,
        };
        return Ok(Model::OpenAi(literal.to_string()));
    }
    #[cfg(not(feature = "openai"))]
    let _ = openai_mode;

    ModelArg::from_str(name, false)
        .map(Model::from)
        .map_err(|_| {
            let choices: Vec<String> = ModelArg::value_variants()
                .iter()
                .filter_map(|v| v.to_possible_value())
                .map(|pv| pv.get_name().to_string())
                .collect();
            format!(
                "不明なモデル: '{name}'(選択肢: {}。または --openai-base-url を指定すると \
                 任意のモデル名を OpenAI 互換 API へ直接渡せます)",
                choices.join(", ")
            )
        })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    #[cfg(feature = "openai")]
    let openai_mode = cli.openai_base_url.is_some();
    #[cfg(not(feature = "openai"))]
    let openai_mode = false;
    let model = resolve_model(&cli.model, openai_mode).map_err(|e| anyhow!(e))?;
    let mut opts = Options::new().with_model(model);
    if let Some(dir) = cli.cache_dir {
        opts = opts.with_cache_dir(dir);
    }
    if !cli.prefer_lang.is_empty() {
        opts = opts.with_language_priority(cli.prefer_lang);
    }
    #[cfg(feature = "openai")]
    {
        if let Some(key) = cli.openai_api_key {
            opts = opts.with_openai_api_key(key);
        }
        if let Some(base_url) = cli.openai_base_url {
            opts = opts.with_openai_base_url(base_url);
        }
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
