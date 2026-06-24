# narashi

![Crates.io License](https://img.shields.io/crates/l/narashi?cacheSeconds=0)
![Crates.io Version](https://img.shields.io/crates/v/narashi?cacheSeconds=0)

2つのテキストの類似度を数値化し、類似している場合はより汎用的な表記に統合することで表記ゆれを解消する Rust 製ツール・ライブラリ。

## 仕組み

- **類似度判定**: 多言語埋め込みモデル(既定: `bge-m3`)でベクトル化し、コサイン類似度を**モデル固有のベースライン基準で 0〜100 スコアに校正**(高帯域に偏りがちなコサイン値を識別しやすいスケールへ展開)
- **汎用性判定**: トークナイザーの ID 値で判定 (早く語彙化されたトークン = ID が小さい = 汎用的)
- **統合優先度**: `(トークン数, トークンID合計)` の辞書式比較で最小のものを代表として採用
- **グルーピング**: 閾値以上のペアを union-find で連結成分化(ペア比較は埋め込みを一度だけ正規化し、内積で並列計算)
- **モデル選択**: 既定は用語集ベンチマーク(日英中混在・難正例/難負例を含む v2 データセット)で ONNX 勢の clusterF1 が最高(0.699)かつ誤統合も最小(7 件)の `bge-m3`。速度を優先するなら `--model gte`(約 1/3 の推論時間)、軽量・最速を優先するなら他モデルへ `--model` で切替可能(選択肢は下の[オプション](#オプション)表、詳細な比較は [`docs/benchmarks.md`](docs/benchmarks.md))

## インストール

[crates.io](https://crates.io/crates/narashi) からインストール:

```sh
cargo install narashi
```

## CLI 使用方法

### 2つのテキストを比較

```sh
$ narashi "白い背景" "白背景"
白い背景 ⇔ 白背景: 99.4
→ 「白背景」に統合
```

閾値未満の場合は統合されません:

```sh
$ narashi "頬紅" "照れ"
頬紅 ⇔ 照れ: 51.5
(閾値 70.0 未満のため統合なし)
```

### 複数テキストを一括で正規化

```sh
$ narashi "白い背景" "白背景" "漫画" "マンガ" "頬紅" "照れ"
[統合] 漫画 ← マンガ
[単独] 照れ
[統合] 白背景 ← 白い背景
[単独] 頬紅
```

### オプション

| フラグ | 説明 | デフォルト |
| --- | --- | --- |
| `-t, --threshold <N>` | 類似度の閾値 (0〜100) | `70.0` |
| `--model <MODEL>` | 埋め込みモデル (`bge-m3` / `gte` / `granite` / `distiluse` / `small` / `large` / `base` / `paraphrase` / `mpnet` / `paraphrase-q` / `e5-instruct` / `qwen3` / `qwen3-4b` / `qwen3-8b`) | `bge-m3` |
| `--cache-dir <PATH>` | モデルキャッシュの保存先 | OS の TEMP フォルダ下 / `narashi` |
| `-h, --help` | ヘルプ表示 | |

`--model` の選択肢(詳細な比較は [`docs/benchmarks.md`](docs/benchmarks.md)):

| 値 | モデル | 次元 | 特徴 |
| --- | --- | ---: | --- |
| `bge-m3` | bge-m3 | 1024 | **既定**。ONNX 勢で clusterF1 最高(0.699)かつ誤統合も最小(7 件)。約 1.06GB・推論は gte の約 3 倍 |
| `gte` | gte-multilingual-base | 768 | 速度重視の代替。CJK に強い・約 1.2GB(bge-m3 の約 1/3 の推論時間)。clusterF1 0.657・誤統合はやや多め(27 件) |
| `granite` | granite-embedding-278m-multilingual | 768 | clusterF1 高め(0.682)・日本語明示学習だが誤統合は多め(28 件)。約 1.1GB |
| `distiluse` | distiluse-base-multilingual-cased-v2 | 768 | 軽量代替の第一候補。clusterF1 0.662・高適合率・約 0.54GB・最速級 |
| `small` | multilingual-e5-small | 384 | 最軽量・最保守(clusterF1 0.561・高適合率・誤統合最少級 14 件・約 0.45GB)。サイズ/速度を最優先する場合 |
| `large` | multilingual-e5-large | 1024 | E5 系で最高精度(0.638)だが低速で gte に劣後 |
| `base` | multilingual-e5-base | 768 | small に劣後(0.514)・非推奨 |
| `paraphrase` | paraphrase-multilingual-MiniLM-L12-v2 | 384 | clusterF1 0.591。要 高め閾値(~89) |
| `mpnet` | paraphrase-multilingual-mpnet-base-v2 | 768 | clusterF1 0.567。要 高め閾値(~86) |
| `paraphrase-q` | paraphrase MiniLM 量子化版 | 384 | 高速だが現環境の ONNX Runtime では実行時エラー |
| `e5-instruct` | multilingual-e5-large-instruct | 1024 | **Candle バックエンド**。外部重み付き ONNX で従来は利用できなかった指示対応 E5。clusterF1 真ピーク 0.645(gte 未満)・誤統合 75 件と最多で Candle CPU で低速のため既定には非推奨だが、ONNX 非依存環境向けの選択肢。約 1.1GB |
| `qwen3` | Qwen3-Embedding-0.6B | 1024 | **Candle バックエンド**(last-token プーリング)。clusterF1 0.764・誤統合 10 件で bge-m3 を精度で上回る。デコーダ系で Candle CPU の推論が bge-m3 の約 12 倍と低速のため既定にはせず、精度重視/ONNX 非依存環境向け。約 1.2GB |
| `qwen3-4b` | Qwen3-Embedding-4B | 2560 | **Candle バックエンド**(last-token・f16)。**clusterF1 0.956(P=0.963・誤統合 7 件)で全モデル中最高精度**。ただし推論が ≈3.9 秒/語と極めて低速・約 8GB RAM 必須のため、GPU・バッチ・オフライン等で速度を許容できる精度最優先用途向け。約 8GB |
| `qwen3-8b` | Qwen3-Embedding-8B | 4096 | **Candle バックエンド**(eval 用)。4B と同経路。約 16GB RAM 必須でさらに低速。十分な RAM の環境での検証用 |

### 実行バックエンド(フィーチャ)

埋め込みの推論バックエンドは Cargo のフィーチャで選べます(既定は両方有効)。

| フィーチャ | バックエンド | 対象モデル | 備考 |
| --- | --- | --- | --- |
| `onnx`(既定) | ONNX Runtime(`fastembed`) | 上表の `e5-instruct` / `qwen3` 以外すべて | ネイティブの ONNX Runtime バイナリを取得・リンクする |
| `candle`(既定) | Candle(ピュア Rust) | `e5-instruct`(XLM-RoBERTa)・`qwen3` / `qwen3-4b` / `qwen3-8b`(Qwen3 デコーダ) | ONNX で扱えないモデルも HF の safetensors から直接読み込む(`config.json` の `model_type` で判定。分割保存・f16 にも対応) |

- 既定の `cargo install narashi` は両バックエンドを含み、従来モデルに加えて `e5-instruct` も使えます。
- `candle` のみでビルドすると、**ネイティブ ONNX Runtime バイナリを取得できない環境**(オフライン・制限ネットワーク等)でも動作します:

  ```sh
  cargo install narashi --no-default-features --features candle,cli
  # 既定モデルは qwen3(Qwen3-Embedding-0.6B)に自動で切り替わります
  ```

ライブラリとして使うだけなら `cli` を外して `clap` 依存を省けます(例: `default-features = false, features = ["candle"]`)。

### GPU(Metal / CUDA)で Candle を実行する

Candle モデル(`qwen3` / `qwen3-4b` / `qwen3-8b` / `e5-instruct`)は GPU で実行できます。GPU は既定では無効で、`metal`(Apple GPU)または `cuda`(NVIDIA GPU)フィーチャを **opt-in** で有効化します。GPU では f16 の matmul が使えるため、CPU で f16 が遅い Qwen3 4B/8B で特に効きます。

| フィーチャ | バックエンド | 対象 | 前提 |
| --- | --- | --- | --- |
| `metal` | Apple GPU(Metal) | Apple Silicon の macOS | macOS 専用。MLX は candle にバックエンドが無いため非対応(Apple GPU は Metal 経由) |
| `cuda` | NVIDIA GPU(CUDA) | NVIDIA GPU 搭載の Linux / Windows | CUDA Toolkit が必要。macOS では使えない |

```sh
# Apple Silicon(Metal)で GPU を使う
cargo install narashi --features metal

# NVIDIA GPU(CUDA)で GPU を使う
cargo install narashi --features cuda
```

ソースからビルドする場合も同様に `--features` で指定します:

```sh
cargo build --release --features metal   # または cuda
cargo run --release --features metal -- "白い背景" "白背景"
```

- GPU フィーチャは内部で `candle` を含むため、別途 `candle` を指定する必要はありません。
- 実行時に GPU デバイスの取得に失敗した場合は、自動的に CPU へフォールバックします(`select_device`)。
- ONNX バックエンド(`onnx` 既定)で動くモデルは GPU フィーチャの影響を受けません。GPU は Candle 経路のモデルにのみ作用します。

> Apple Silicon での計測では、Metal は Candle CPU 並列の約 1.8 倍速。バッチ化は candle 側の制約により未対応です。

キャッシュディレクトリは環境変数 `NARASHI_CACHE_DIR` でも指定できます:

```sh
$ export NARASHI_CACHE_DIR=/path/to/cache
$ narashi "テキスト1" "テキスト2"
```

優先順位は **コマンドラインフラグ > 環境変数 > デフォルト(TEMP)** の順です。

### ダウンロードしたモデルの削除

モデルはキャッシュディレクトリ以下に保存されます(初回利用時に Hugging Face から自動ダウンロード)。
不要になったら、そのディレクトリを削除すれば再ダウンロード前の状態に戻せます。

```sh
# デフォルト(OS の TEMP フォルダ下)の場合 — すべてのモデルを削除
$ rm -rf "$(dirname "$(mktemp -u)")/narashi"

# キャッシュ位置を指定している場合は、そのディレクトリを削除
$ rm -rf "$NARASHI_CACHE_DIR"          # 環境変数で指定したとき
$ rm -rf /path/to/cache                # --cache-dir で指定したとき
```

キャッシュ内はモデルごとにサブフォルダ(`models--…` 形式)に分かれているため、特定のモデルだけ削除したい場合は
該当サブフォルダのみを消すこともできます。

## クレート経由での使用

`Cargo.toml` に追加:

```toml
[dependencies]
narashi = "0.4"
```

### 類似度の算出

```rust
use narashi::Narashi;

let n = Narashi::new()?;
let score = n.similarity("白い背景", "白背景")?;
println!("{score:.1}"); // => 99.4
```

### 表記ゆれ解消

```rust
use narashi::{Narashi, Group};

let n = Narashi::new()?;
let texts: Vec<String> = ["白い背景", "白背景", "漫画", "マンガ"]
    .iter()
    .map(|s| s.to_string())
    .collect();

let groups: Vec<Group> = n.normalize(&texts, 70.0)?;
for g in &groups {
    println!("canonical={} members={:?}", g.canonical, g.members);
}
// => canonical=漫画 members=["マンガ", "漫画"]
// => canonical=白背景 members=["白い背景", "白背景"]
```

`Group` は元のテキストがどの代表に統合されたかを完全に追跡します。

### モデルの指定

```rust
use narashi::{EmbeddingModel, Narashi, Options};

// 既定は bge-m3(clusterF1 最高・誤統合最小)。軽量・最速の E5 small へ切り替える例:
let opts = Options::new().with_model(EmbeddingModel::MultilingualE5Small);
let n = Narashi::with_options(opts)?;
```

### キャッシュディレクトリの指定

```rust
use narashi::{Narashi, Options};

let opts = Options::new().with_cache_dir("/path/to/cache");
let n = Narashi::with_options(opts)?;
```

明示指定がない場合は環境変数 `NARASHI_CACHE_DIR`、それも無ければ OS の TEMP フォルダ下 (`narashi` サブディレクトリ) が使われます。

## ライセンス

以下のいずれかを選択:

- MIT License ([LICENSE-MIT](LICENSE-MIT))
- Apache License 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

### 埋め込みモデルのライセンス

narashi はモデルの重みを同梱せず、実行時に Hugging Face から各モデルをダウンロードします。各モデルはそれぞれの提供元のライセンス(いずれも Apache 2.0 または MIT の寛容なライセンス)に従います。利用にあたっては原典のライセンスをご確認ください。

| `--model` | 原典 | ライセンス |
| --- | --- | --- |
| `bge-m3`(既定) | [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)([Xenova/bge-m3](https://huggingface.co/Xenova/bge-m3) の fp16 ONNX) | MIT |
| `gte` | [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) | Apache 2.0 |
| `granite` | [ibm-granite/granite-embedding-278m-multilingual](https://huggingface.co/ibm-granite/granite-embedding-278m-multilingual) | Apache 2.0 |
| `distiluse` | [sentence-transformers/distiluse-base-multilingual-cased-v2](https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2) | Apache 2.0 |
| `small` | [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) | MIT |
| `base` | [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) | MIT |
| `large` | [intfloat/multilingual-e5-large](https://huggingface.co/intfloat/multilingual-e5-large) | MIT |
| `paraphrase` / `paraphrase-q` | [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) | Apache 2.0 |
| `mpnet` | [sentence-transformers/paraphrase-multilingual-mpnet-base-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2) | Apache 2.0 |
| `e5-instruct` | [intfloat/multilingual-e5-large-instruct](https://huggingface.co/intfloat/multilingual-e5-large-instruct)(safetensors を Candle で直接読込) | MIT |
| `qwen3` / `qwen3-4b` / `qwen3-8b` | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) / [4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) / [8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)(safetensors を Candle で直接読込) | Apache 2.0 |
