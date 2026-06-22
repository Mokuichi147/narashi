# narashi

2つのテキストの類似度を数値化し、類似している場合はより汎用的な表記に統合することで表記ゆれを解消する Rust 製ツール・ライブラリ。

## 仕組み

- **類似度判定**: 多言語埋め込みモデル(既定: `bge-m3`)でベクトル化し、コサイン類似度を**モデル固有のベースライン基準で 0〜100 スコアに校正**(高帯域に偏りがちなコサイン値を識別しやすいスケールへ展開)
- **汎用性判定**: トークナイザーの ID 値で判定 (早く語彙化されたトークン = ID が小さい = 汎用的)
- **統合優先度**: `(トークン数, トークンID合計)` の辞書式比較で最小のものを代表として採用
- **グルーピング**: 閾値以上のペアを union-find で連結成分化(ペア比較は埋め込みを一度だけ正規化し、内積で並列計算)
- **モデル選択**: 既定は用語集ベンチマーク(日英中混在)で clusterF1 が最高(0.725)かつ誤統合も最小の `bge-m3`。速度を優先するなら `--model gte`(約 1/3 の推論時間)、軽量・最速を優先するなら他モデルへ `--model` で切替可能(選択肢は下の[オプション](#オプション)表、詳細な比較は [`docs/benchmarks.md`](docs/benchmarks.md))

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
| `--model <MODEL>` | 埋め込みモデル (`bge-m3` / `gte` / `granite` / `distiluse` / `small` / `large` / `base` / `paraphrase` / `mpnet` / `paraphrase-q`) | `bge-m3` |
| `--cache-dir <PATH>` | モデルキャッシュの保存先 | OS の TEMP フォルダ下 / `narashi` |
| `-h, --help` | ヘルプ表示 | |

`--model` の選択肢(詳細な比較は [`docs/benchmarks.md`](docs/benchmarks.md)):

| 値 | モデル | 次元 | 特徴 |
| --- | --- | ---: | --- |
| `bge-m3` | bge-m3 | 1024 | **既定**。clusterF1 最高(0.725)かつ誤統合も最小(1 件)。約 1.06GB・推論は gte の約 3 倍 |
| `gte` | gte-multilingual-base | 768 | 速度重視の代替。高適合率・CJK に強い・約 1.2GB(bge-m3 の約 1/3 の推論時間) |
| `granite` | granite-embedding-278m-multilingual | 768 | clusterF1 高め(0.705)・日本語明示学習だが誤統合は多め。約 1.1GB |
| `distiluse` | distiluse-base-multilingual-cased-v2 | 768 | 軽量代替の第一候補。高適合率・約 0.54GB・最速級 |
| `small` | multilingual-e5-small | 384 | 最軽量・最保守(高適合率・約 0.45GB)。サイズ/速度を最優先する場合 |
| `large` | multilingual-e5-large | 1024 | E5 系で最高精度だが低速で gte に劣後 |
| `base` | multilingual-e5-base | 768 | small に劣後・非推奨 |
| `paraphrase` | paraphrase-multilingual-MiniLM-L12-v2 | 384 | 高再現率。要 高め閾値(~88) |
| `mpnet` | paraphrase-multilingual-mpnet-base-v2 | 768 | 再現率最優先。要 高め閾値(~87) |
| `paraphrase-q` | paraphrase MiniLM 量子化版 | 384 | 高速だが現環境の ONNX Runtime では実行時エラー |

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
narashi = "0.2"
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
