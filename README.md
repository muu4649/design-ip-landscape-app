# デザインランドスケープ — Gemini 無料API版

J-PlatPat の意匠検索結果（PDF / CSV）を Google Gemini で自動分析し、  
デザイン空間の可視化と戦略レポートを生成する Streamlit アプリ。

---

## 機能概要

| タブ | 内容 |
|------|------|
| 📥 データ取込 | J-PlatPat PDF / CSV をアップロード、サムネイル確認 |
| 🤖 Gemini 分析 | 意匠を形状・素材・スタイル・革新性スコアなどでバッチ分類 |
| 🗺️ ランドスケープ | 分類結果を PCA 2D に変換して Plotly インタラクティブ散布図 |
| 📊 クラスター | クラスター別サムネイルギャラリー・集計・AI要約 |
| 📝 戦略レポート | VOYAGER 3フェーズ分析による Markdown 戦略レポート生成 |

---

## 使用技術

- **LLM**: Google Gemini 2.5 Flash（無料ティア）
- **PDF 解析**: PyMuPDF (fitz)
- **次元削減**: PCA 2D (scikit-learn)
- **クラスタリング**: KMeans (scikit-learn)
- **可視化**: Plotly Express
- **UI**: Streamlit

---

## セットアップ

### 1. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

### 2. Gemini API キーの取得

[Google AI Studio](https://aistudio.google.com) で無料 API キーを発行する。

無料ティアの制限:
- 15 リクエスト / 分（RPM）
- 1,500 リクエスト / 日（RPD）
- 1,000,000 トークン / 分

### 3. アプリ起動

```bash
streamlit run app.py
```

---

## 使い方

1. サイドバーに Gemini API キーを入力して「検証」
2. **📥 データ取込** タブで J-PlatPat の意匠検索結果 PDF または CSV をアップロード
3. **🤖 Gemini 分析** タブで「分析開始」
4. **🗺️ ランドスケープ** タブで 2D 散布図を確認
5. **📝 戦略レポート** タブで戦略レポートを生成・ダウンロード

---

## J-PlatPat からのデータ取得

1. [J-PlatPat](https://www.j-platpat.inpit.go.jp/) にアクセス
2. 意匠検索で対象を絞り込む
3. 検索結果を **PDF** または **CSV** でダウンロード
4. 本アプリにアップロード

---

## VOYAGER 3フェーズ レポート

戦略レポートは以下の 3 フェーズで生成される。

```
Phase 1: クラスター別アナリスト
         各クラスターの特徴・競合状況を個別分析

Phase 2: クロスクラスター統合
         全体のデザイントレンド・空白領域を統合分析

Phase 3: 戦略レポート
         経営層向け Markdown レポート（[[Evidence X]] 引用付き）
```

---

## ファイル構成

```
.
├── app.py             # Streamlit メインアプリ (5タブ)
├── gemini_client.py   # Gemini API クライアント + VOYAGER 3フェーズ
├── pdf_parser.py      # J-PlatPat PDF / CSV パーサー
├── requirements.txt
└── .streamlit/
    └── config.toml
```

---

## 注意事項

- 無料ティアの RPD (1,500/日) を消費するため、大量データの繰り返し分析には注意
- J-PlatPat PDF のハーグ協定経由登録案件は出願人名が取得できない場合がある
- 分析結果はあくまで参考情報であり、法的判断の根拠には使用しないこと
