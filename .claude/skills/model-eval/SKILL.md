---
name: model-eval
description: narashi の表記ゆれ統合に使う埋め込みモデルを評価・比較し、既定モデルや閾値を選定するワークフロー。新しいモデルを検討するとき、既定モデルを変えるとき、用語集を更新して再計測するときに使う。
---

# 埋め込みモデルの評価・比較ワークフロー

narashi はコサイン類似度で表記ゆれを統合するため、埋め込みモデルの良し悪しが精度を左右する。
モデル候補を「同じ物差し」で横並び比較し、既定モデル(`DEFAULT_MODEL`)と閾値(`DEFAULT_THRESHOLD`)を
データに基づいて選ぶための手順。結果は必ず `docs/benchmarks.md` に追記して履歴を残す。

## 全体像

0. 候補モデルのライセンスを確認する(寛容なライセンスのみ採用可)
1. 用語集(正解データ)を必要に応じて拡張する
2. 候補モデルをベンチマークで計測する(キャッシュに無いモデルは事前取得が必要)
3. 指標を解釈して用途に合うモデルを選ぶ
4. `docs/benchmarks.md` を更新し、必要なら `src/lib.rs` の既定を変更する

## 0. ライセンスの確認 (候補に入れる前の必須チェック)

narashi 本体は **MIT / Apache-2.0 のデュアルライセンス**。新しい候補モデルを評価対象に入れる前に、
**原典の Hugging Face モデルカードでライセンスを確認**し、寛容なライセンス(Apache 2.0 / MIT)で
**商用利用可・コピーレフトや非商用(NC)制限なし**のものだけを採用する。CC-BY-NC、GPL 系、独自の
利用規約付き(Llama 系など)、ライセンス不明のモデルは原則採用しない。

- narashi はモデル重みを**同梱せず実行時に Hugging Face からダウンロード**するため、リポジトリでの
  重み再配布は発生しない。ただし利用者がそのモデルを使うことになるので、利用者にとって安全な
  ライセンスであることを保証する。
- ONNX 変換リポジトリ(`onnx-community/*`, `Xenova/*`, `Qdrant/*` 等)はモデルカードにライセンス
  タグが無いことがある。その場合は**原典モデル**のライセンスを根拠とする(機械的変換は原典の
  ライセンスが及ぶ)。原典が不明なものは採用しない。
- 採用したら、原典リポジトリとライセンスを **`README.md` のライセンス節のモデル表**に追記する。

現行モデルのライセンス(いずれも寛容):

| `--model` | 原典 | ライセンス |
| --- | --- | --- |
| `bge-m3`(既定) | BAAI/bge-m3(`Xenova/bge-m3` の fp16 ONNX) | MIT |
| `gte` | Alibaba-NLP/gte-multilingual-base | Apache 2.0 |
| `granite` | ibm-granite/granite-embedding-278m-multilingual | Apache 2.0 |
| `qwen3` / `qwen3-4b` / `qwen3-8b`(Candle) | Qwen/Qwen3-Embedding-0.6B / 4B / 8B(safetensors を Candle で直読・4B/8B は分割保存・f16) | Apache 2.0 |
| `e5-instruct`(Candle・見送り) | intfloat/multilingual-e5-large-instruct(safetensors を Candle で直読) | MIT |
| `arctic-l`(eval用) | Snowflake/snowflake-arctic-embed-l-v2.0 | Apache 2.0 |
| `granite-107m` / `granite-278m` / `granite-311m-r2` / `granite-97m-r2`(eval用) | ibm-granite/granite-embedding-* | Apache 2.0 |
| `distiluse` | sentence-transformers/distiluse-base-multilingual-cased-v2 | Apache 2.0 |
| `small` / `base` / `large` | intfloat/multilingual-e5-* | MIT |
| `paraphrase` / `paraphrase-q` | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 | Apache 2.0 |
| `mpnet` | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 | Apache 2.0 |
| `bge-zh`(比較用) | BAAI/bge-small-zh-v1.5 | MIT |
| `all-minilm`(比較用) | sentence-transformers/all-MiniLM-L6-v2 | Apache 2.0 |
| `clip`(比較用) | OpenAI CLIP (sentence-transformers/clip-ViT-B-32) | MIT |

## 1. 用語集 (正解データ)

`tests/data/glossary.txt` が唯一の正解データ。`src/eval.rs` が `include_str!` で取り込む。

- 1 行 = 1 つの同義語グループ、メンバはカンマ区切り。`#` 始まりと空行は無視。
- **同一グループ内ペア = 統合すべき(正例)**、**異グループ間ペア = 統合すべきでない(負例)**。
- メンバ 1 つの行 = 単独語(ディストラクタ)。どれとも統合されない。難しい負例(字面が近い等)を入れる。
- 多言語埋め込みなので、日本語の表記ゆれ(かな/カナ/漢字/送り仮名)に加え、**同義の英語・中国語を同じ
  グループに入れる**とクロス言語統合能力も測れる(例: `猫, ネコ, ねこ, cat, 貓`)。
- 整合性チェック: クロス言語の同義語は必ず**同じ**グループへ。別グループの語同士が実は同義だと
  正解ラベルが壊れる(偽陰性化する)。単独語は互いに、またどのグループとも同義でないこと。
  `src/eval.rs` の `default_glossary_has_no_duplicate_terms` テストが語の重複(=ラベル破壊)を検知する。
- **飽和対策(重要)**: 上位モデルが容易に満点へ達するとモデル間を分離できない。`glossary.txt` には
  易しい例だけでなく**難所**を多く入れる(詳細は `docs/benchmarks.md` の「ベンチマークの飽和対策」)。
  - **難正例**(再現率の難所): 表層が重ならない多言語同義語(略語/外来語/漢語/英中混在。例
    `便所, トイレ, お手洗い, restroom, 厕所`)。
  - **難負例**(適合率の難所): 埋め込み空間で近接するが別語。共下位語(季節・色・曜日・方角・兄弟・飲料)、
    紛らわしい異義語(`科学`/`化学`)、対義語(`増加`/`減少`)を**それぞれ単独グループ**で入れる。
  - 難所を足したら `default_glossary_has_hard_cases` テストの代表語も必要に応じて更新する。

## 2. 計測

計測には 2 つの example を使い分ける。

- **`benchmark`** — 1 モデルの素性能サマリ(best-F1 / best-thr / margin / per_text)と **5 刻み(0〜100)
  閾値スイープ**。各モデルの概形と速度・分離度を一望する。
- **`fine_sweep`** — **1 刻み(0〜100)閾値スイープ**で clusterF1 の真ピーク(値・最適閾値・ピークの鋭さ)を
  確定する。**モデル間比較はこの真ピークで行う**(後述「3. 指標の読み方」)。

```sh
export ORT_LIB_LOCATION=/tmp/ortlib/onnxruntime \
       NARASHI_CACHE_DIR=/tmp/narashi_cache HF_HOME=/tmp/narashi_cache HF_HUB_OFFLINE=1

# 素性能サマリ + 5 刻みスイープ
cargo run --example benchmark                  # 既定 (bge-m3)
cargo run --example benchmark -- bge-m3        # bge-m3(既定・ONNX勢でclusterF1最高0.699・誤統合最小7件・fp16単一ONNX)
cargo run --example benchmark -- gte           # gte-multilingual-base(速度重視の代替・高適合率)
cargo run --example benchmark -- granite-278m  # granite-278m(clusterF1高め・誤統合多め)
cargo run --example benchmark -- arctic-l      # snowflake-arctic-embed-l-v2.0(検索特化・低調・eval用)
cargo run --example benchmark -- qwen3         # Qwen3-Embedding-0.6B(Candle・last-token・clusterF1 0.764・低速)
cargo run --example benchmark -- qwen3-4b      # Qwen3-Embedding-4B(Candle・f16・clusterF1最高0.956・超低速≈3.9秒/語・約8GB RAM)
cargo run --example benchmark -- qwen3-8b      # Qwen3-Embedding-8B(Candle・f16・約16GB RAM 必須・要十分なRAM)
cargo run --example benchmark -- e5-instruct   # multilingual-e5-large-instruct(Candle・見送り・誤統合多め)
cargo run --example benchmark -- granite-107m  # granite-107m(R1 軽量・eval用)
cargo run --example benchmark -- granite-311m-r2  # granite-311m R2(eval用)
cargo run --example benchmark -- granite-97m-r2   # granite-97m R2(eval用)
cargo run --example benchmark -- distiluse     # distiluse-base-multilingual-cased-v2(軽量代替)
cargo run --example benchmark -- small         # multilingual-e5-small
cargo run --example benchmark -- base          # multilingual-e5-base
cargo run --example benchmark -- large         # multilingual-e5-large
cargo run --example benchmark -- paraphrase    # paraphrase-multilingual-MiniLM-L12-v2
cargo run --example benchmark -- mpnet         # paraphrase-multilingual-mpnet-base-v2
cargo run --example benchmark -- paraphrase-q  # 量子化版(※現環境の ORT では実行時エラー)
cargo run --example benchmark -- large 75      # 第2引数で閾値を指定
# 別系統(多言語特化でないベースライン。下限確認用)
cargo run --example benchmark -- bge-zh        # BGE 中国語特化
cargo run --example benchmark -- all-minilm    # 英語 SentenceTransformers
cargo run --example benchmark -- clip          # CLIP テキストエンコーダ

# clusterF1 の真ピーク(1 刻み)— モデル間比較の主データ
cargo run --example fine_sweep -- gte
cargo run --example fine_sweep -- distiluse

# 連鎖暴走(over-merge)の堅牢性 — 用語集 + ディストラクタ(ハブ語+無関係タグ)で暴走オンセットを測る
cargo run --example robustness -- bge-m3
cargo run --example robustness -- qwen3   # 実データで崩壊したモデルは必ず測る(Candle は --features candle)
```

各モデルを順に流し、出力をそのまま比較する。`load` 時間はサンドボックスのコールドスタート依存なので、
速度は `per_text`(1 用語あたり)で比較する。

**Candle バックエンドのモデル**(`qwen3` / `e5-instruct`)は ONNX Runtime を介さず HF の safetensors を
直接読む(`config.json` の `model_type` でアーキ判定。対応: xlm-roberta / qwen3)。ORT バイナリを取得
できない本環境では **`--no-default-features --features candle`** でビルドし、ORT 非依存の `candle_eval`
example で計測する(`benchmark`/`fine_sweep` は `onnx` 必須でビルドに ORT が要るため使えない):

```sh
# release で(デバッグビルドは Candle が桁違いに遅い。0.6B でも数十分かかる)
cargo run --release --no-default-features --features candle --example candle_eval -- qwen3
cargo run --release --no-default-features --features candle --example candle_eval -- qwen3 70 fine  # 第3引数 fine で1刻み真ピークも算出
cargo run --release --no-default-features --features candle --example candle_eval -- qwen3-4b
```

事前取得は ONNX 勢と同様に `curl` でキャッシュへ:`config.json` / `tokenizer.json` / `model.safetensors`
(必要なら `tokenizer_config.json` 等)を置く。Candle 勢の per_text は ONNX 勢と別系統のため直接比較不可
(参考値)。新アーキを足すときは `src/candle_backend.rs` の `match model_type` に分岐を追加する
(各アーキで `Config` 型・コンストラクタ・`forward` 引数・プーリングが異なる)。

**固定閾値(既定 70)での横並び比較はしない**。校正は全モデル共通([`SCORE_BASELINE`]=0.5)だが、
各モデルの余弦分布が違うため**適切な動作点(閾値)はモデルごとに異なる**。`benchmark` の 5 刻みスイープで
概形をつかみ、**`fine_sweep` の 1 刻み真ピーク(各モデルの最適閾値での clusterF1)でモデル間を比較**する。
モデルにより clusterF1 のピーク閾値は大きく異なる(共通校正下では総じて高め)が、ピーク閾値の位置自体は
優劣ではない。比較は真ピークの **値** で行い、運用閾値は別途 `robustness` の暴走オンセットを見て選ぶ。

新しい候補を検討するときは、e5/paraphrase 系の本命に加えて**別系統(BGE/英語ST/CLIP 等)も 1〜2 個**
混ぜて下限を可視化すると、多言語特化が効いているかが定量的に示せる。

### キャッシュに無いモデルの事前取得 (重要な落とし穴)

この環境のネットワークは TLS 傍受プロキシ経由で、Rust 側 (hf-hub + rustls の組合せ) は
プロキシの CA を信頼せず `invalid peer certificate: UnknownIssuer` でダウンロードに失敗する。
一方 `curl` はシステム CA バンドルでプロキシ CA を信頼するため成功する。よって**未キャッシュの
モデルは `curl` で hf-hub のキャッシュ構造に手動配置**してから `HF_HUB_OFFLINE=1` で実行する。

キャッシュ構造(プレーンなファイル。blob シンボリックリンクは不要):

```
$NARASHI_CACHE_DIR/models--<org>--<name>/refs/main          # 中身は snapshot ディレクトリ名 (任意の文字列)
$NARASHI_CACHE_DIR/models--<org>--<name>/snapshots/<name>/   # ここに実ファイルを置く
```

取得が必要なファイル(`fastembed` のモデル定義 `model_file` + `load_tokenizer_hf_hub`):

- ONNX 本体: `model_file`(モデルにより `onnx/model.onnx` / `model.onnx` / `model_optimized.onnx`)。
  外部重みを持つモデル(e5-large 等)は `additional_files`(例: `model.onnx_data`)も必要。
- トークナイザ: `tokenizer.json`, `config.json`, `special_tokens_map.json`, `tokenizer_config.json`。
- 注意: narashi は `src/lib.rs` の `model_spec().hf_repo` から別途 `tokenizer.json` を取得する。これが
  fastembed の `model_code` リポジトリと**異なる**モデル(e5-large, paraphrase-q)では、両方のリポジトリの
  キャッシュを用意する必要がある。

`model_file` / `additional_files` / `model_code` の正値は fastembed のソースで確認:
`~/.cargo/registry/src/*/fastembed-*/src/models/text_embedding.rs`。

取得スニペット(例: e5-base):

```sh
cd "$NARASHI_CACHE_DIR"
repo="intfloat/multilingual-e5-base"; dir="models--${repo//\//--}"; snap="$dir/snapshots/snap"
mkdir -p "$snap/onnx"
for f in tokenizer.json config.json special_tokens_map.json tokenizer_config.json onnx/model.onnx; do
  curl -fsSL "https://huggingface.co/$repo/resolve/main/$f" -o "$snap/$f"
done
mkdir -p "$dir/refs"; echo -n snap > "$dir/refs/main"
```

## 3. 指標の読み方

`src/eval.rs` が算出。詳細と最新の数値は `docs/benchmarks.md` を参照。

- **clusterF1 真ピーク** = 各モデルの最適閾値(`fine_sweep` の 1 刻みピーク)での `normalize` 実出力
  (推移閉包込み)と正解の一致度 = 実運用の挙動。**モデル間比較はこの値そのもので行う**(固定閾値 70 の
  値は使わない)。ただし**上位モデルでは飽和・希釈で頭打ちになりやすい**ので、下の最難ペア指標と併用する。
- **最難負例 / 最難正例(閾値非依存・飽和対策の主軸)**: `benchmark` / `candle_eval` 出力末尾の 2 リスト。
  最難負例 = 誤統合しやすい順の負例ペア(上位モデルほどスコアが**低い**)、最難正例 = 取りこぼしやすい順の
  正例ペア(上位モデルほどスコアが**高い**)。**clusterF1 が飽和しても分離が続く**ため、最上位モデル同士の
  優劣はこのスコア(最難負例の最高スコア・最難正例の最低スコア)で判定する。
- **ピーク時 P / R**: 真ピーク閾値での適合率/再現率。表記ゆれ統合では**誤統合(低 Precision)はデータ
  破壊**で取りこぼし(低 Recall)より痛い。同程度の clusterF1 なら Precision の高いモデルを選ぶ。
- **誤統合(真ピーク時)**: 各モデルの**真ピーク閾値**での誤統合ペア数(`fine_sweep` の真ピーク行の値)。
  clusterF1 と並ぶ採用判断の主要指標。**固定閾値 70 での誤統合は比較に使わない**——モデルごとにピークが
  異なり、ピーク外では再現率が落ちて誤統合が見かけ上減る(例: arctic-l は真ピーク 51 で誤統合 13 件だが
  @70 では 0 件)ため、固定閾値の比較は無意味。clusterF1 を真ピークで比べるのと同じ理由で誤統合も真ピーク
  で比べる。
- **best-F1 / best-thr**: 素ペア分類 F1 の上限と、その最適閾値。
- **margin**: `正例min − 負例max`。0 に近いほど正例・負例が分離している。
- **最適閾値の位置は優劣ではない**: best-thr / clusterF1 真ピークの閾値が 70 から外れても、それは
  各モデルの余弦分布の違い(共通校正 [`SCORE_BASELINE`]=0.5 のもとでの相対位置)であって、モデルの
  良し悪しの根拠にしない。比較はあくまで真ピークの **値** で行う。運用閾値はモデルごとに `fine_sweep`
  (品質ピーク)と `robustness`(暴走オンセット)を見て選ぶ。

## 3'. 連鎖暴走(over-merge)対策【既定モデル/閾値の採用判断で最優先】

`normalize` は単連結クラスタリング(union-find の推移閉包)なので、`1`/`2B`/`3D` のような**汎用短トークン
(ハブ語)**が無関係なクラスタの「橋」になると、全体が 1 つの巨大ゴミクラスタへ連鎖崩壊する(**連鎖暴走**)。
実データでは qwen3-4b・閾値 70 で 9166 タグ中 7663 が単一クラスタへ崩壊した。**clusterF1・誤統合(ペア数)では
これを検知できない**(トポロジー指標が必要)。クリーン用語集では規模不足で再現もできない。

- **`robustness` example で測る**: 用語集 + `tests/data/distractors.txt`(ハブ語 + 多ドメイン無関係タグ)を連結し、
  1 刻みスイープで次の 2 指標を見る(`src/eval.rs` の `SweepRow` / `Clustering`):
  - **`largest_cluster_ratio`** = 最大予測クラスタ / 全語数。1.0 に近いほど暴走。
  - **`groups_in_largest`** = 最大クラスタに巻き込まれた相異なる正解グループ数。健全なら 1、暴走で多数。
- **暴走オンセット**: ratio ≥ 10% または groups ≥ 5 になる最高閾値。example が末尾に出力する。
- **運用閾値は clusterF1 のピークではなく、暴走オンセットより安全側(高い)で選ぶ**。clusterF1 ピークが
  暴走域内に来るモデルが多く、ピークに合わせると実データで崩壊する(example が ⚠ で警告する)。
- **校正は単一定数 [`SCORE_BASELINE`]=0.5(モデル非依存)**(`score = (cos-0.5)/(1-0.5)*100`、全モデルで
  閾値 70 = 余弦 0.85)。**モデル間の余弦分布差は、校正ではなく「モデルごとに運用閾値を選ぶ」ことで吸収する**。
  だから**モデルを替えたら必ず閾値も選び直す**。実測例: granite-278m は @74–78 で安全(巻込2・R≈0.55–0.58)、
  bge-m3 は @72–74、distiluse は @90 まで上げないと安全にならない。e5-small は @70 で崩壊する。
  詳細表は `docs/benchmarks.md`「堅牢性: 連鎖暴走対策」。
- **新モデル採用時は必ず robustness を回す**。clusterF1 が高くても、暴走オンセットより安全側に十分な再現率を
  残せる運用閾値が取れなければ採用しない。Candle 勢も `--features candle` で測る(per_text が遅いので時間に
  注意。数百語 × 低速)。

## 4. 反映

- 結果表と所見を **`docs/benchmarks.md` に更新**(素性能の表・5 刻みスイープ表・**1 刻み真ピーク表
  〔モデル間比較の主表〕**・**堅牢性=暴走オンセット表**の 4 つを揃える)。日付見出しの追加検証
  セクションは作らず、恒常セクションへ統合する(数値・採用判断と記述整理を混ぜない)。
- 既定を変えるなら `src/lib.rs` の `DEFAULT_MODEL` を変更し、doc コメントの根拠も更新。
  **既定の採用は clusterF1 だけでなく `robustness` の暴走オンセットが運用閾値より安全側にあることを必須条件にする**。
- 新しい選択肢を CLI に出すなら `src/main.rs` の `ModelArg`、`examples/benchmark.rs`・`examples/fine_sweep.rs`・
  `examples/robustness.rs` の match 分岐、`README.md` のモデル表を揃える。
- 新しいモデルを採用したら、原典リポジトリとライセンスを **`README.md` のライセンス節のモデル表**にも
  追記する(セクション 0 の確認結果を反映)。
- スコア校正は全モデル共通の単一定数。変えるなら `src/lib.rs` の [`SCORE_BASELINE`] を更新する
  (モデル間の差は運用閾値をモデルごとに選んで吸収する)。

## これまでの結論

最新の数値・採用判断は `docs/benchmarks.md` を一次情報とする(本節は要約のみ)。比較は **clusterF1 真
ピーク**(各モデルの最適閾値での値)で行う。**運用閾値はモデルごとに `robustness` の暴走オンセットを見て
選ぶ**:
- **精度の絶対最高は Qwen3-Embedding-8B**(Candle・Apache 2.0・last-token・f16)。clusterF1 真ピーク
  **0.974(P=0.984・R=0.964・誤統合 3 件)**で全モデル中最高、堅牢性ベンチでも暴走耐性が最良(@86–89 で
  巻込1・R≈0.87–0.91)。RTX 3090(24GB VRAM・CUDA)で計測。GPU 実行なら per_text≈26ms と高速だが要 GPU
  (`--model qwen3-8b`、`--features candle,cuda`)。
- **次が Qwen3-Embedding-4B**(Candle・Apache 2.0・last-token・f16)。clusterF1 真ピーク
  **0.956(P=0.963・R=0.948・誤統合 7 件)**(v1 では P=1.000・誤統合 0 と「完璧」だったが、
  難化した v2 で `保証⇔保障` などを誤統合し頭打ちが解消)。弱点は速度/RAM(Candle CPU・f16・≈3.9 秒/語・約 8GB RAM。
  GPU なら大幅高速)。GPU・バッチ・オフライン等で速度を許容できる精度最優先用途向け(`--model qwen3-4b`)。
- **その中間が Qwen3-Embedding-0.6B**(Candle・Apache 2.0・last-token)。clusterF1 真ピーク 0.764・P=0.926・
  誤統合 10 件で bge-m3(0.699)を精度で上回る。≈200ms/語と 4B よりは軽い。ONNX 非依存環境向け(`--model qwen3`・
  Candle 単独ビルドでは既定)。
- **総合の既定は bge-m3**(精度・速度・安全性のバランス)。clusterF1 真ピーク **0.699 は ONNX 勢で最高**、ピーク時
  P=0.939・**誤統合わずか 7 件は強モデル中で最小**で、ONNX により Qwen3 の約 12 倍高速(≈16ms/語)。
  「高 clusterF1 は誤統合増と引き換え」(granite-278m は 0.682 だが誤統合 28 件)というトレードオフを破り、
  精度・安全性・サイズ(1.06GB)で gte(0.657 / 誤統合 27 件)を上回る。`BAAI/bge-m3` の **fp16 単一 ONNX**
  (`Xenova/bge-m3`)をユーザー定義モデルとして読み込む。
- **速度重視の代替は gte-multilingual-base**。真ピーク 0.657・推論は bge-m3 の約 1/3(≈5ms/語)だが誤統合 27 件と
  多め。約 1.2GB。
- **clusterF1/再現率を最優先するなら granite-278m-multilingual**(IBM Granite Embedding R1・Apache 2.0)。
  clusterF1 真ピーク 0.682(bge-m3 に次ぐ)・日本語は明示学習。ただし真ピーク時の誤統合 28 件で「猫=犬」型の
  データ破壊が増えるため選択肢に留める(`--model granite`)。
- **軽量代替の第一候補は distiluse-multilingual-v2**(真ピーク 0.662・誤統合 12 件・最小サイズ 0.54GB・最速級)。
  v2 では clusterF1・誤統合とも gte を上回る。ただし単漢字かな(`猫⇔ねこ`)の難正例は取りこぼす。
- **e5-small はなお保守枠**(最軽量・最速・ピーク時 P=0.853・誤統合 14 件)。e5-large は 0.638 だが約 8 倍遅い。
  e5-base は劣後(0.514)。paraphrase 系は真ピーク 0.59/0.57 でピークが高閾値側・誤統合多め。量子化版は現環境で実行不可。
- granite の 278m 以外(311m-r2:0.652 / 107m:0.652 / 97m-r2:0.537)は誤統合が多く不採用(eval から選択可)。
- **大型枠の突破口**: fp16 単一 ONNX(≤2GB)を使えば外部重み付き大型モデルも評価可能。bge-m3 はこの方法で
  評価し採用した。一方 snowflake-arctic-embed-l-v2.0(fp16・真ピーク 0.495)は検索特化で低調・見送り(eval から選択可)。
- snowflake-arctic-embed-m-v2.0【非推奨】・LaBSE【見送り】はコードに組み込まず結果のみ記録。
- 別系統(BGE-zh / 英語 MiniLM / CLIP)は本データで大きく劣り、多言語特化が必須と確認。
- **Candle バックエンドで評価できるようになったモデル**: e5-large-instruct(XLM-RoBERTa・外部重み付き ONNX
  しか無く従来不可→ Candle で評価→ clusterF1 0.645・誤統合 75 件で**見送り**)、Qwen3-Embedding(last-token
  プーリングで fastembed 非対応→ Candle に実装。**0.6B で 0.764、4B(分割保存・f16・約 8GB)で 0.956、
  8B(f16)で clusterF1 0.974・P=0.984・誤統合 3 件と全モデル中最高**→ いずれも**採用**。8B は RTX 3090
  (24GB VRAM・CUDA)で計測した)。
- 評価対象外(現状): jina-v3(CC-BY-NC・ライセンス非寛容)・ruri-base(BertJapaneseTokenizer が MeCab/unidic
  依存で `tokenizer.json` 非配布。BertModel は Candle で読めるが純 Rust では形態素解析器が別途必要)・
  gte-Qwen2-7B / e5-mistral-7b 等(last-token だが 7〜8B 級で Candle CPU では非現実的)。詳細は benchmarks 参照。
