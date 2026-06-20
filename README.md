# narashi

2つのテキストの類似度を数値化し、類似している場合はより汎用的な表記に統合することで表記ゆれを解消する Rust 製ツール・ライブラリ。

## 仕組み

- **類似度判定**: 多言語埋め込みモデル(既定: `gte-multilingual-base`)でベクトル化し、コサイン類似度を**モデル固有のベースライン基準で 0〜100 スコアに校正**(高帯域に偏りがちなコサイン値を識別しやすいスケールへ展開)
- **汎用性判定**: トークナイザーの ID 値で判定 (早く語彙化されたトークン = ID が小さい = 汎用的)
- **統合優先度**: `(トークン数, トークンID合計)` の辞書式比較で最小のものを代表として採用
- **グルーピング**: 閾値以上のペアを union-find で連結成分化(ペア比較は埋め込みを一度だけ正規化し、内積で並列計算)
- **モデル選択**: 既定の gte-multilingual-base は用語集ベンチマーク(日英中混在)で実運用挙動の clusterF1 が全候補中で最高(ピーク 0.682)かつ既定閾値ちょうどで高適合率(誤統合がほぼ無い)。軽量・最速を優先する場合は `--model small`、その他も `--model` で切替可能(モデル比較は [`docs/benchmarks.md`](docs/benchmarks.md))

## インストール

```sh
cargo install --path .
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
| `--model <MODEL>` | 埋め込みモデル (`gte` / `small` / `large` / `base` / `paraphrase` / `mpnet` / `paraphrase-q`) | `gte` |
| `--cache-dir <PATH>` | モデルキャッシュの保存先 | OS の TEMP フォルダ下 / `narashi` |
| `-h, --help` | ヘルプ表示 | |

`--model` の選択肢(詳細な比較は [`docs/benchmarks.md`](docs/benchmarks.md)):

| 値 | モデル | 次元 | 特徴 |
| --- | --- | ---: | --- |
| `gte` | gte-multilingual-base | 768 | **既定**。精度最良(clusterF1 トップ 0.682・CJK に強い)。約 3 倍低速・1.2GB |
| `small` | multilingual-e5-small | 384 | 高適合率・最速級・軽量(約 0.45GB)。速度/サイズ重視向け |
| `large` | multilingual-e5-large | 1024 | E5 系の上限(0.644)。約 8 倍低速で gte に劣後 |
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

## クレート経由での使用

`Cargo.toml` に追加:

```toml
[dependencies]
narashi = { git = "https://github.com/mokuichi147/narashi" }
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

// 既定は gte-multilingual-base(精度最良)。軽量・最速の E5 small へ切り替える例:
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
