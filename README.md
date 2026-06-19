# narashi

2つのテキストの類似度を数値化し、類似している場合はより汎用的な表記に統合することで表記ゆれを解消する Rust 製ツール・ライブラリ。

## 仕組み

- **類似度判定**: 多言語埋め込みモデル(既定: `paraphrase-multilingual-MiniLM-L12-v2`)でベクトル化し、コサイン類似度を**モデル固有のベースライン基準で 0〜100 スコアに校正**(高帯域に偏りがちなコサイン値を識別しやすいスケールへ展開)
- **汎用性判定**: トークナイザーの ID 値で判定 (早く語彙化されたトークン = ID が小さい = 汎用的)
- **統合優先度**: `(トークン数, トークンID合計)` の辞書式比較で最小のものを代表として採用
- **グルーピング**: 閾値以上のペアを union-find で連結成分化(ペア比較は埋め込みを一度だけ正規化し、内積で並列計算)
- **モデル選択**: 既定の paraphrase モデルは同規模(384次元/12層)の E5 small より日本語の短い表記ゆれ(例:「猫」⇔「ネコ」)で大きく高精度。`--model` で切替も可能

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
| `--model <MODEL>` | 埋め込みモデル (`paraphrase` / `paraphrase-q` / `small` / `base`) | `paraphrase` |
| `--cache-dir <PATH>` | モデルキャッシュの保存先 | OS の TEMP フォルダ下 / `narashi` |
| `-h, --help` | ヘルプ表示 | |

`--model` の選択肢:

| 値 | モデル | 特徴 |
| --- | --- | --- |
| `paraphrase` | paraphrase-multilingual-MiniLM-L12-v2 | 既定。対称類似度向けで表記ゆれに最も高精度 |
| `paraphrase-q` | 同上(量子化版) | 精度をほぼ保ちつつ高速・省メモリ |
| `small` | multilingual-e5-small | 検索向けチューニング(同規模) |
| `base` | multilingual-e5-base | 高次元(768)・低速 |

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

// 既定は paraphrase-multilingual-MiniLM-L12-v2。E5 small へ切り替える例:
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
