"""
Design IP Landscape AI Assistant
Streamlit + Claude API — 全フェーズ対応アシスタント
"""

from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

import anthropic
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
try:
    import japanize_matplotlib  # noqa: F401
except ImportError:
    pass

# ─────────────────────────────────────────────
# 定数
# ─────────────────────────────────────────────
MODEL = "claude-opus-4-5"
MAX_TOKENS = 8096

SHAPE_CATEGORIES = [
    "バスケット/円筒型",
    "円錐型",
    "台形/扇型",
    "フレーム/ワイヤー型",
    "フラットボトム/ウェーブ型",
    "ドーム/半球型",
    "ダイヤモンド/多面体型",
    "一体型",
    "ドリップバッグ/袋型",
]

SCORE_AXES = ["競合の少なさ", "IP先行余地", "素材空白", "市場成長性", "商標余地"]

# フェーズ定義（番号・名前・色・検出キーワード）
PHASES: list[dict] = [
    {"num": 1, "en": "SCOPE",    "ja": "スコープ設定",       "color": "#2471A3",
     "keywords": ["スコープ", "目的", "物品", "カテゴリ", "競合定義"]},
    {"num": 2, "en": "GATHER",   "ja": "データ収集",          "color": "#1A5276",
     "keywords": ["J-PlatPat", "CSV", "ダウンロード", "データ収集", "取得"]},
    {"num": 3, "en": "READ",     "ja": "意匠ランドスケープ",  "color": "#1A5276",
     "keywords": ["出願人", "ランキング", "年別", "推移", "overview", "timeline", "partial"]},
    {"num": 4, "en": "CLASSIFY", "ja": "形状タクソノミー",    "color": "#7D6608",
     "keywords": ["形状分類", "タクソノミー", "classify", "shape"]},
    {"num": 5, "en": "MATRIX",   "ja": "形状×素材マトリクス", "color": "#B7770D",
     "keywords": ["マトリクス", "matrix", "素材", "白地図"]},
    {"num": 6, "en": "SPOT",     "ja": "白地図の特定",        "color": "#B7770D",
     "keywords": ["白地図", "未充填", "チャンス", "SPOT"]},
    {"num": 7, "en": "SCORE",    "ja": "5軸スコアリング",     "color": "#6C3483",
     "keywords": ["スコア", "5軸", "レーダー", "評価"]},
    {"num": 8, "en": "NAME",     "ja": "商標ランドスケープ",  "color": "#1E8449",
     "keywords": ["商標", "trademark", "命名", "区分", "Drip"]},
    {"num": 9, "en": "BUILD",    "ja": "コンセプト具現化",    "color": "#922B21",
     "keywords": ["スケッチ", "コンセプト", "寸法", "BUILD", "デザイン案"]},
]

# フェーズごとの次アクション候補（フォールバック用）
PHASE_NEXT_ACTIONS: dict[int, list[dict]] = {
    0: [  # 開始前
        {"label": "Ph1 スコープを設定する",
         "prompt": "コーヒードリッパーを対象にIPランドスケープ分析を始めます。Ph1 SCOPEとして、分析スコープ・目的・競合定義を整理してください。"},
        {"label": "J-PlatPatのデータ取得ガイドを見る",
         "prompt": "Ph2 GATHERとして、J-PlatPatからコーヒードリッパーの意匠データと商標データを取得する手順を教えてください。"},
    ],
    1: [
        {"label": "Ph2 データ収集ガイドへ",
         "prompt": "Ph2 GATHERとして、J-PlatPatから意匠CSVと意匠画像PDFを取得する具体的な手順を教えてください。"},
    ],
    2: [
        {"label": "Ph3 出願人ランキングを分析",
         "prompt": "Ph3 READとして、意匠データの出願人ランキングTop15を分析してください。"},
        {"label": "Ph3 年別出願推移を確認",
         "prompt": "Ph3 READとして、意匠出願の年別推移グラフを生成してください。"},
    ],
    3: [
        {"label": "Ph3 意匠種別（部分/関連）を確認",
         "prompt": "部分意匠・関連意匠の割合を分析してください。"},
        {"label": "Ph4 形状タクソノミーへ進む",
         "prompt": "Ph4 CLASSIFYとして、データを9形状カテゴリに分類してください。"},
    ],
    4: [
        {"label": "Ph5 形状×素材マトリクスを生成",
         "prompt": "Ph5 MATRIXとして、形状カテゴリ×素材のマトリクスを生成し白地図を可視化してください。"},
    ],
    5: [
        {"label": "Ph6 白地図候補を整理",
         "prompt": "Ph6 SPOTとして、マトリクスの白地図（未充填領域）を論点整理してください。どの形状×素材の組合せが最も有望ですか？"},
        {"label": "Ph7 5軸スコアリングへ進む",
         "prompt": "Ph7 SCOREとして、上位候補コンセプトを5軸（競合の少なさ・IP先行余地・素材空白・市場成長性・商標余地）でスコアリングしてください。"},
    ],
    6: [
        {"label": "Ph7 スコアリング実行",
         "prompt": "Ph7 SCOREとして、白地図候補の上位3コンセプトを5軸スコアリングしてレーダーチャートを生成してください。"},
    ],
    7: [
        {"label": "Ph8 商標ランドスケープを分析",
         "prompt": "Ph8 NAMEとして、区分11・21の商標白地図を分析し、Drip系命名の白地図候補を抽出してください。"},
    ],
    8: [
        {"label": "Ph8 命名提案（Top1コンセプト）",
         "prompt": "最もスコアの高いコンセプトに対して、商標白地図から3案の命名提案をしてください。"},
        {"label": "Ph9 デザインスケッチを生成",
         "prompt": "Ph9 BUILDとして、最優先コンセプトの3Dデザインスケッチ（5面図）を生成してください。"},
    ],
    9: [
        {"label": "分析レポートをまとめる",
         "prompt": "今回の分析結果を総括してください。スコープ・白地図・採択コンセプト・商標提案を一覧で整理してください。"},
        {"label": "別コンセプトでスケッチ生成",
         "prompt": "別の形状・素材の組合せでデザインスケッチを生成してください。"},
    ],
}

# ─────────────────────────────────────────────
# システムプロンプト
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """あなたはデザイン意匠IPランドスケープ専門のAIアシスタントです。
以下の9フェーズフレームワークに従って、コーヒードリッパーや生活雑貨製品の
意匠・商標分析を全フェーズ主体的にリードします。

## フレームワーク概要

Ph 1 SCOPE    — スコープ設定（物品カテゴリ、分析目的、競合定義）
Ph 2 GATHER   — データ収集（J-PlatPat CSV・画像PDF取得ガイド）
Ph 3 READ     — 意匠ランドスケープ（出願人分析・時系列・関連/部分意匠）
Ph 4 CLASSIFY — 形状タクソノミー（ルールベース形状分類・9カテゴリ）
Ph 5 MATRIX   — 形状×素材マトリクス（白地図候補の可視化）
Ph 6 SPOT     — 白地図の特定（未充填領域の論点整理）
Ph 7 SCORE    — 5軸スコアリング（競合・IP余地・素材・市場・商標の5軸）
Ph 8 NAME     — 商標ランドスケープ（区分11/21 商標白地図・命名提案）
Ph 9 BUILD    — コンセプト具現化（設計仕様・3Dスケッチ生成）

## 行動指針
- 各フェーズが完了したら、次フェーズへの移行を能動的に提案する
- ユーザーを待たず、「次はXXXを実行します」と宣言してツールを呼び出す
- データがない場合でも分析方法・手順を説明してリードする
- 数値根拠を明示（件数・割合・スコア）
- 回答は日本語、結論ファースト

## ツール使用方針
- 意匠CSVがあれば analyze_design_patents を即座に活用
- 商標CSVがあれば analyze_trademarks を即座に活用
- 形状×素材マトリクスは generate_shape_matrix で生成
- スコアリングは score_concepts で5軸レーダーチャートを生成
- 商標提案は propose_trademarks で実行
- デザインスケッチは generate_sketch で生成

## 必須出力形式
回答の末尾に必ず以下のブロックを含めること（UIのボタン生成に使用）：

<next_actions>
[
  {"label": "次のアクションのボタンラベル（20文字以内）", "prompt": "ボタンを押したときにAIに送るプロンプト"},
  {"label": "別の選択肢", "prompt": "対応するプロンプト"}
]
</next_actions>

next_actionsには常に2〜3件の具体的な次ステップを含めること。
"""

# ─────────────────────────────────────────────
# ツール定義
# ─────────────────────────────────────────────
TOOLS: list[dict] = [
    {
        "name": "analyze_design_patents",
        "description": (
            "意匠CSVデータを分析し、出願人ランキング・年別出願推移・形状分類分布・"
            "関連/部分意匠の割合などを可視化する。"
            "analysis_type: 'overview'|'timeline'|'shape'|'partial'"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "analysis_type": {
                    "type": "string",
                    "enum": ["overview", "timeline", "shape", "partial"],
                    "description": "実行する分析の種類",
                },
                "top_n": {
                    "type": "integer",
                    "description": "上位N件を表示（デフォルト15）",
                    "default": 15,
                },
            },
            "required": ["analysis_type"],
        },
    },
    {
        "name": "analyze_trademarks",
        "description": (
            "商標CSVを分析し、区分別の出願件数・既登録商標の修飾語リスト・"
            "白地図候補（未使用のXXX Drip / Drip XXX 形式）を抽出する。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_classes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "分析する区分番号リスト（例: [11, 21]）",
                },
                "keyword_filter": {
                    "type": "string",
                    "description": "商標テキストフィルタキーワード（任意）",
                },
            },
            "required": [],
        },
    },
    {
        "name": "generate_shape_matrix",
        "description": (
            "形状カテゴリ × 素材のマトリクスを生成し、既存権利の密度と"
            "白地図（未充填領域）を赤破線でハイライトした図を返す。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "materials": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "素材リスト（省略時はデフォルトセットを使用）",
                },
                "highlight_white_space": {
                    "type": "boolean",
                    "description": "白地図ハイライト表示（デフォルトtrue）",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "score_concepts",
        "description": (
            "複数コンセプト（形状×素材の組合せ）を5軸（競合の少なさ・IP先行余地・"
            "素材空白・市場成長性・商標余地）でスコアリングし、"
            "レーダーチャートとスコア一覧を返す。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "concepts": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "shape": {"type": "string"},
                            "material": {"type": "string"},
                            "scores": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "5軸スコア [競合, IP余地, 素材, 市場, 商標] 各1-10",
                            },
                        },
                        "required": ["name", "shape", "scores"],
                    },
                    "description": "スコアリングするコンセプトのリスト",
                }
            },
            "required": ["concepts"],
        },
    },
    {
        "name": "propose_trademarks",
        "description": (
            "コンセプトの説明を受け取り、商標白地図から最適な商標名を3案提案する。"
            "前置型（XXX Drip）・後置型（Drip XXX）・造語型の3パターンを検討する。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "concept_description": {
                    "type": "string",
                    "description": "コンセプトの特徴・形状・素材・ターゲット市場の説明",
                },
                "nice_classes": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "出願を検討する区分（デフォルト [11, 21]）",
                },
                "style_preference": {
                    "type": "string",
                    "enum": ["prefix", "suffix", "both"],
                    "description": "商標スタイル: prefix=XXX Drip, suffix=Drip XXX, both=両方",
                    "default": "both",
                },
            },
            "required": ["concept_description"],
        },
    },
    {
        "name": "generate_sketch",
        "description": (
            "形状・素材・寸法仕様を受け取り、Python matplotlibで3Dデザインスケッチを生成する。"
            "物理ベースのライティングを使用したフォトリアリスティックな意匠スケッチ。"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "concept_name": {
                    "type": "string",
                    "description": "コンセプト名（例: Concept C）",
                },
                "shape_type": {
                    "type": "string",
                    "enum": [
                        "cone",
                        "flat_bottom",
                        "diamond",
                        "cylinder",
                        "trapezoid",
                        "frame",
                        "dome",
                    ],
                    "description": "形状タイプ",
                },
                "material": {
                    "type": "string",
                    "description": "素材名（例: SUS304, チタン, セラミック）",
                },
                "dimensions": {
                    "type": "object",
                    "properties": {
                        "diameter_top": {"type": "number", "description": "上部直径 mm"},
                        "diameter_bottom": {"type": "number", "description": "下部直径 mm"},
                        "height": {"type": "number", "description": "高さ mm"},
                    },
                    "description": "主要寸法",
                },
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "特徴リスト（例: ['ウェーブリブ8条', '抽出穴3×φ3mm']）",
                },
            },
            "required": ["concept_name", "shape_type"],
        },
    },
]

# ─────────────────────────────────────────────
# 形状分類ユーティリティ
# ─────────────────────────────────────────────
APPLICANT_RULES: dict[str, str] = {
    "HARIO": "円錐型",
    "ＨＡＲＩＯ": "円錐型",
    "ハリオ": "円錐型",
    "kalita": "フラットボトム/ウェーブ型",
    "Kalita": "フラットボトム/ウェーブ型",
    "カリタ": "フラットボトム/ウェーブ型",
    "melitta": "台形/扇型",
    "Melitta": "台形/扇型",
    "メリタ": "台形/扇型",
    "KINTO": "円錐型",
    "キントー": "円錐型",
    "コーノ": "円錐型",
    "KONO": "円錐型",
    "origami": "台形/扇型",
    "Origami": "台形/扇型",
    "オリガミ": "台形/扇型",
    "YETI": "フレーム/ワイヤー型",
    "Pi-Design": "一体型",
    "片岡物産": "フレーム/ワイヤー型",
    "大日本印刷": "ドリップバッグ/袋型",
    "ユーシーシー": "ドリップバッグ/袋型",
    "UCC": "ドリップバッグ/袋型",
    "key coffee": "ドリップバッグ/袋型",
    "KEY COFFEE": "ドリップバッグ/袋型",
    "キーコーヒー": "ドリップバッグ/袋型",
}

ITEM_KEYWORDS: dict[str, str] = {
    "ドリップバッグ": "ドリップバッグ/袋型",
    "ドリップパック": "ドリップバッグ/袋型",
    "コーヒーバッグ": "ドリップバッグ/袋型",
    "カプセル": "その他",
    "ポッド": "その他",
    "フィルター": "台形/扇型",
    "ペーパーフィルター": "台形/扇型",
}


def classify_shape(applicant: str, item_name: str) -> str:
    """出願人名・物品名から形状カテゴリを推定する。"""
    for key, shape in APPLICANT_RULES.items():
        if key.lower() in str(applicant).lower():
            return shape
    for key, shape in ITEM_KEYWORDS.items():
        if key in str(item_name):
            return shape
    return "バスケット/円筒型"


def macro_cat(item_name: str) -> str:
    """ドリッパー本体か周辺品かを大分類する。"""
    dripper_kws = ["ドリッパー", "dripper", "ドリップ器", "コーヒードリッパー"]
    bag_kws = ["バッグ", "パック", "袋"]
    capsule_kws = ["カプセル", "ポッド", "capsule", "pod"]
    n = str(item_name).lower()
    if any(k in n for k in capsule_kws):
        return "カプセル/ポッド型"
    if any(k in n for k in bag_kws):
        return "ドリップバッグ/袋型"
    if any(k in n for k in dripper_kws):
        return "ドリッパー本体"
    return "周辺品/その他"


# ─────────────────────────────────────────────
# ツール実装
# ─────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure) -> str:
    """matplotlibのFigureをbase64 PNG文字列に変換する。"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def tool_analyze_design_patents(analysis_type: str, top_n: int = 15) -> dict:
    """意匠CSVを分析してチャートを生成する。"""
    df: pd.DataFrame | None = st.session_state.get("design_df")
    if df is None or df.empty:
        return {"text": "意匠CSVがアップロードされていません。サイドバーからCSVをアップロードしてください。"}

    BG = "#0F1923"
    TEXT = "#E8EDF2"
    ACC = "#2471A3"

    if analysis_type == "overview":
        # 出願人ランキング
        col_applicant = _find_col(df, ["出願人", "権利者", "出願人/権利者", "applicant"])
        if col_applicant is None:
            return {"text": f"出願人列が見つかりません。列名: {list(df.columns)}"}

        counts = df[col_applicant].value_counts().head(top_n)
        fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
        ax.set_facecolor(BG)
        bars = ax.barh(counts.index[::-1], counts.values[::-1], color=ACC, edgecolor="none")
        for bar, val in zip(bars, counts.values[::-1]):
            ax.text(val + 0.3, bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", color=TEXT, fontsize=9)
        ax.set_title(f"出願人ランキング Top{top_n}", color=TEXT, fontsize=13, pad=10)
        ax.set_xlabel("件数", color=TEXT)
        ax.tick_params(colors=TEXT, labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334455")
        text_summary = (
            f"総レコード数: {len(df)}件\n"
            f"ユニーク出願人数: {df[col_applicant].nunique()}社\n"
            f"Top1出願人: {counts.index[0]}（{counts.iloc[0]}件）"
        )
        return {"text": text_summary, "chart_b64": _fig_to_b64(fig), "chart_title": "出願人ランキング"}

    elif analysis_type == "timeline":
        col_date = _find_col(df, ["出願日", "登録日", "公告日", "date"])
        col_applicant = _find_col(df, ["出願人", "権利者", "出願人/権利者"])
        if col_date is None:
            return {"text": f"日付列が見つかりません。列名: {list(df.columns)}"}

        df2 = df.copy()
        df2["year"] = pd.to_datetime(df2[col_date], errors="coerce").dt.year
        yearly = df2.groupby("year").size().dropna()
        yearly = yearly[yearly.index >= 2000]

        fig, ax = plt.subplots(figsize=(10, 5), facecolor=BG)
        ax.set_facecolor(BG)
        ax.fill_between(yearly.index.astype(int), yearly.values, alpha=0.3, color=ACC)
        ax.plot(yearly.index.astype(int), yearly.values, color=ACC, linewidth=2, marker="o", markersize=4)
        ax.set_title("年別出願件数推移", color=TEXT, fontsize=13, pad=10)
        ax.set_xlabel("年", color=TEXT)
        ax.set_ylabel("件数", color=TEXT)
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334455")
        peak_year = int(yearly.idxmax())
        peak_val = int(yearly.max())
        return {
            "text": f"ピーク年: {peak_year}年（{peak_val}件）\n2020年以降の件数: {int(yearly[yearly.index >= 2020].sum())}件",
            "chart_b64": _fig_to_b64(fig),
            "chart_title": "年別出願件数推移",
        }

    elif analysis_type == "shape":
        col_applicant = _find_col(df, ["出願人", "権利者", "出願人/権利者"])
        col_item = _find_col(df, ["意匠に係る物品", "物品", "商品"])
        if col_applicant is None or col_item is None:
            return {"text": f"必要な列が見つかりません。列名: {list(df.columns)}"}

        df2 = df.copy()
        df2["macro"] = df2[col_item].apply(macro_cat)
        df2 = df2[df2["macro"] == "ドリッパー本体"].copy()
        df2["shape"] = df2.apply(
            lambda r: classify_shape(r[col_applicant], r[col_item]), axis=1
        )

        shape_counts = df2["shape"].value_counts()
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(shape_counts)))[::-1]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
        for ax in axes:
            ax.set_facecolor(BG)

        # 棒グラフ
        axes[0].barh(shape_counts.index[::-1], shape_counts.values[::-1],
                     color=colors[::-1], edgecolor="none")
        axes[0].set_title("形状カテゴリ別件数", color=TEXT, fontsize=12, pad=8)
        axes[0].tick_params(colors=TEXT, labelsize=8)
        for spine in axes[0].spines.values():
            spine.set_edgecolor("#334455")

        # 円グラフ
        wedges, texts, autotexts = axes[1].pie(
            shape_counts.values,
            labels=shape_counts.index,
            autopct="%1.1f%%",
            colors=colors,
            textprops={"color": TEXT, "fontsize": 8},
        )
        for at in autotexts:
            at.set_fontsize(7)
        axes[1].set_title("形状構成比", color=TEXT, fontsize=12, pad=8)

        fig.suptitle("ドリッパー本体 形状分類", color=TEXT, fontsize=14, y=1.01)

        st.session_state.setdefault("analysis_results", {})
        st.session_state["analysis_results"]["shape_df"] = df2

        top_shape = shape_counts.index[0]
        return {
            "text": (
                f"ドリッパー本体: {len(df2)}件\n"
                f"最多形状: {top_shape}（{shape_counts.iloc[0]}件, "
                f"{shape_counts.iloc[0]/len(df2)*100:.1f}%）\n"
                f"形状カテゴリ数: {len(shape_counts)}種"
            ),
            "chart_b64": _fig_to_b64(fig),
            "chart_title": "形状分類分布",
        }

    elif analysis_type == "partial":
        col_type = _find_col(df, ["その他種別", "種別", "type"])
        if col_type is None:
            return {"text": f"種別列が見つかりません。列名: {list(df.columns)}"}

        df2 = df.copy()
        df2[col_type] = df2[col_type].fillna("通常意匠")
        type_counts = df2[col_type].value_counts()

        fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)
        ax.set_facecolor(BG)
        colors = [ACC if "部分" in str(t) else "#1E8449" if "関連" in str(t) else "#7D6608"
                  for t in type_counts.index]
        ax.bar(range(len(type_counts)), type_counts.values, color=colors, edgecolor="none")
        ax.set_xticks(range(len(type_counts)))
        ax.set_xticklabels(type_counts.index, rotation=20, ha="right", color=TEXT, fontsize=9)
        ax.set_title("意匠種別（部分意匠・関連意匠）", color=TEXT, fontsize=12, pad=8)
        ax.set_ylabel("件数", color=TEXT)
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334455")

        partial = sum(type_counts[i] for i in type_counts.index if "部分" in str(i))
        related = sum(type_counts[i] for i in type_counts.index if "関連" in str(i))
        total = len(df)
        return {
            "text": (
                f"部分意匠: {partial}件（{partial/total*100:.1f}%）\n"
                f"関連意匠: {related}件（{related/total*100:.1f}%）\n"
                f"通常意匠: {total-partial-related}件"
            ),
            "chart_b64": _fig_to_b64(fig),
            "chart_title": "意匠種別分布",
        }

    return {"text": f"未知の analysis_type: {analysis_type}"}


def tool_analyze_trademarks(
    target_classes: list[int] | None = None, keyword_filter: str | None = None
) -> dict:
    """商標CSVを分析する。"""
    df: pd.DataFrame | None = st.session_state.get("trademark_df")
    if df is None or df.empty:
        return {"text": "商標CSVがアップロードされていません。サイドバーからCSVをアップロードしてください。"}

    BG = "#0F1923"
    TEXT = "#E8EDF2"

    col_class = _find_col(df, ["区分", "class", "ニース分類"])
    col_tm = _find_col(df, ["商標", "商標(検索用)", "商標テキスト", "trademark"])
    col_status = _find_col(df, ["ステータス", "status", "状態"])

    results: list[str] = []

    # 区分フィルタ
    df2 = df.copy()
    if target_classes and col_class:
        df2 = df2[df2[col_class].astype(str).str.contains(
            "|".join(str(c) for c in target_classes)
        )]
        results.append(f"対象区分 {target_classes}: {len(df2)}件")

    if keyword_filter and col_tm:
        df2 = df2[df2[col_tm].astype(str).str.contains(keyword_filter, case=False, na=False)]
        results.append(f"キーワード「{keyword_filter}」: {len(df2)}件")

    # 修飾語抽出（XXX Drip / Drip XXX パターン）
    modifiers: dict[str, str] = {}  # modifier -> position
    if col_tm:
        for tm in df2[col_tm].dropna().astype(str):
            tm_clean = tm.strip().upper()
            m_pre = re.match(r"^([A-Z]+)\s+DRIP$", tm_clean)
            m_suf = re.match(r"^DRIP\s+([A-Z]+)$", tm_clean)
            if m_pre:
                modifiers[m_pre.group(1)] = "前置型"
            elif m_suf:
                modifiers[m_suf.group(1)] = "後置型"

    # 白地図候補（一般的な形容詞で未使用のもの）
    candidates = [
        "FLOW", "WAVE", "PURE", "ARCH", "GRAIN", "EDGE", "FORM",
        "PRISM", "FACET", "RING", "ARC", "SLIM", "PEAK", "CREST",
        "FOLD", "CURL", "DRIFT", "MIST", "VEIL", "RIDGE",
    ]
    white_space = [c for c in candidates if c not in modifiers]

    # 区分別件数グラフ
    charts: list[str] = []
    if col_class:
        class_counts = df[col_class].value_counts().head(15)
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=BG)
        ax.set_facecolor(BG)
        ax.bar(range(len(class_counts)), class_counts.values,
               color="#6C3483", edgecolor="none")
        ax.set_xticks(range(len(class_counts)))
        ax.set_xticklabels(class_counts.index, rotation=30, ha="right",
                           color=TEXT, fontsize=8)
        ax.set_title("区分別商標件数", color=TEXT, fontsize=12, pad=8)
        ax.set_ylabel("件数", color=TEXT)
        ax.tick_params(colors=TEXT)
        for spine in ax.spines.values():
            spine.set_edgecolor("#334455")
        charts.append(_fig_to_b64(fig))

    # 白地図候補
    st.session_state.setdefault("analysis_results", {})
    st.session_state["analysis_results"]["tm_white_space"] = white_space
    st.session_state["analysis_results"]["tm_existing"] = modifiers

    results.append(f"\n既存修飾語（Drip系）: {list(modifiers.keys())[:10]}")
    results.append(f"白地図候補: {white_space[:10]}")

    result: dict[str, Any] = {"text": "\n".join(results)}
    if charts:
        result["chart_b64"] = charts[0]
        result["chart_title"] = "商標分析"
    return result


def tool_generate_shape_matrix(
    materials: list[str] | None = None, highlight_white_space: bool = True
) -> dict:
    """形状×素材マトリクスを生成する。"""
    BG = "#0F1923"
    TEXT = "#E8EDF2"

    if materials is None:
        materials = ["ステンレス", "樹脂/プラスチック", "セラミック", "チタン", "銅/真鍮", "ガラス", "木材/竹"]

    shapes = SHAPE_CATEGORIES

    # セッションから形状分類データを取得
    shape_df: pd.DataFrame | None = st.session_state.get("analysis_results", {}).get("shape_df")

    # デフォルト密度マトリクス（データがない場合のサンプル）
    density = np.array([
        [5, 4, 1, 0, 2, 0, 0],  # バスケット/円筒型
        [3, 2, 2, 1, 0, 1, 0],  # 円錐型
        [4, 3, 1, 0, 1, 0, 0],  # 台形/扇型
        [1, 2, 0, 1, 0, 0, 0],  # フレーム/ワイヤー型
        [2, 1, 1, 1, 0, 0, 0],  # フラットボトム/ウェーブ型
        [1, 1, 1, 0, 0, 1, 0],  # ドーム/半球型
        [0, 1, 0, 1, 0, 0, 0],  # ダイヤモンド/多面体型
        [2, 1, 0, 0, 0, 0, 0],  # 一体型
        [1, 0, 0, 0, 0, 0, 0],  # ドリップバッグ/袋型
    ])

    if shape_df is not None and not shape_df.empty:
        col_applicant = _find_col(shape_df, ["出願人", "権利者", "出願人/権利者"])
        col_item = _find_col(shape_df, ["意匠に係る物品", "物品"])
        if col_applicant and col_item:
            for i, shape in enumerate(shapes):
                mask = shape_df["shape"] == shape
                n = mask.sum()
                if n > 0:
                    density[i, 0] = max(density[i, 0], min(n, 9))

    # 白地図ハイライト（密度 == 0）
    white_space_cells: list[tuple[int, int]] = []
    for i in range(len(shapes)):
        for j in range(len(materials)):
            if density[i, j] == 0:
                white_space_cells.append((i, j))

    fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
    ax.set_facecolor(BG)

    cmap = plt.cm.Blues
    im = ax.imshow(density, cmap=cmap, vmin=0, vmax=10, aspect="auto")

    # セル値表示
    for i in range(len(shapes)):
        for j in range(len(materials)):
            val = density[i, j]
            color = TEXT if val < 5 else "#0F1923"
            ax.text(j, i, str(val) if val > 0 else "—",
                    ha="center", va="center", fontsize=10,
                    color=color, fontweight="bold" if val == 0 else "normal")

    # 白地図ハイライト
    if highlight_white_space:
        for (i, j) in white_space_cells:
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                  fill=False, edgecolor="#E74C3C",
                                  linewidth=2, linestyle="--")
            ax.add_patch(rect)

    ax.set_xticks(range(len(materials)))
    ax.set_xticklabels(materials, rotation=25, ha="right", color=TEXT, fontsize=9)
    ax.set_yticks(range(len(shapes)))
    ax.set_yticklabels(shapes, color=TEXT, fontsize=9)
    ax.set_title("形状カテゴリ × 素材 マトリクス（数値=登録意匠件数）",
                 color=TEXT, fontsize=13, pad=12)
    ax.tick_params(colors=TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334455")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors=TEXT, labelsize=8)
    cbar.set_label("登録件数", color=TEXT, fontsize=9)

    n_white = len(white_space_cells)
    white_desc = ", ".join(
        f"{shapes[i]}×{materials[j]}" for (i, j) in white_space_cells[:5]
    )
    if len(white_space_cells) > 5:
        white_desc += f"…他{n_white - 5}件"

    return {
        "text": (
            f"白地図セル数: {n_white}箇所\n"
            f"主な白地図: {white_desc}\n"
            "赤破線 = 登録ゼロの未充填領域（出願チャンス）"
        ),
        "chart_b64": _fig_to_b64(fig),
        "chart_title": "形状×素材マトリクス",
    }


def tool_score_concepts(concepts: list[dict]) -> dict:
    """5軸スコアリングのレーダーチャートを生成する。"""
    BG = "#0F1923"
    TEXT = "#E8EDF2"

    if not concepts:
        return {"text": "コンセプトが指定されていません。"}

    axes_labels = SCORE_AXES
    N = len(axes_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    colors_list = ["#2471A3", "#1E8449", "#B7770D", "#922B21", "#6C3483", "#117A65"]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True}, facecolor=BG)
    ax.set_facecolor("#1A2633")

    for idx, concept in enumerate(concepts):
        scores = concept.get("scores", [5] * N)
        if len(scores) < N:
            scores = scores + [5] * (N - len(scores))
        scores = scores[:N]
        values = scores + [scores[0]]
        color = colors_list[idx % len(colors_list)]
        ax.plot(angles, values, color=color, linewidth=2, label=concept.get("name", f"概念{idx+1}"))
        ax.fill(angles, values, color=color, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, color=TEXT, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], color="#778899", fontsize=8)
    ax.grid(color="#334455", linewidth=0.8)
    ax.spines["polar"].set_color("#334455")
    ax.set_title("5軸スコアリング", color=TEXT, fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15),
              facecolor="#1A2633", edgecolor="#334455",
              labelcolor=TEXT, fontsize=10)

    # テキストサマリー
    lines: list[str] = []
    for concept in concepts:
        scores = concept.get("scores", [])
        total = sum(scores) if scores else 0
        avg = total / len(scores) if scores else 0
        lines.append(f"**{concept.get('name')}** — 合計:{total:.1f} / 平均:{avg:.1f}")
        for ax_label, sc in zip(axes_labels, scores):
            lines.append(f"  {ax_label}: {sc}")

    return {
        "text": "\n".join(lines),
        "chart_b64": _fig_to_b64(fig),
        "chart_title": "5軸スコアリング レーダーチャート",
    }


def tool_propose_trademarks(
    concept_description: str,
    nice_classes: list[int] | None = None,
    style_preference: str = "both",
) -> dict:
    """商標を提案する。"""
    if nice_classes is None:
        nice_classes = [11, 21]

    white_space: list[str] = st.session_state.get("analysis_results", {}).get(
        "tm_white_space", [
            "FLOW", "WAVE", "PURE", "ARCH", "GRAIN", "PRISM", "FACET",
            "RIDGE", "CREST", "DRIFT",
        ]
    )
    existing: dict = st.session_state.get("analysis_results", {}).get("tm_existing", {})

    # コンセプトのキーワードから最適候補を選定
    concept_lower = concept_description.lower()
    scored: list[tuple[str, int]] = []
    keyword_weights = {
        "フラット": ["FLAT", "WAVE", "FLOW", "CREST"],
        "ウェーブ": ["WAVE", "CURL", "FLOW", "DRIFT"],
        "ダイヤ": ["PRISM", "FACET", "EDGE", "PEAK"],
        "多面体": ["PRISM", "FACET", "FORM", "EDGE"],
        "円錐": ["PEAK", "RIDGE", "GRAIN", "FLOW"],
        "金属": ["EDGE", "ARCH", "RIDGE", "FORM"],
        "ステンレス": ["EDGE", "FLOW", "ARCH", "PURE"],
        "チタン": ["PEAK", "EDGE", "PURE", "ARCH"],
        "セラミック": ["PURE", "FLOW", "WAVE", "MIST"],
        "木": ["GRAIN", "FOLD", "PURE", "FLOW"],
    }

    boost: dict[str, int] = {}
    for kw, candidates in keyword_weights.items():
        if kw in concept_lower:
            for c in candidates:
                boost[c] = boost.get(c, 0) + 2

    for candidate in white_space:
        score = boost.get(candidate, 0)
        scored.append((candidate, score))

    scored.sort(key=lambda x: -x[1])
    top_candidates = [s[0] for s in scored[:6]]

    proposals: list[dict] = []
    for i, word in enumerate(top_candidates[:3], 1):
        if style_preference in ("prefix", "both"):
            name_pre = f"{word} Drip"
        if style_preference in ("suffix", "both"):
            name_suf = f"Drip {word}"

        if style_preference == "prefix":
            tm_name = name_pre
        elif style_preference == "suffix":
            tm_name = name_suf
        else:
            tm_name = name_pre if i % 2 == 1 else name_suf

        proposals.append({
            "rank": i,
            "name": tm_name,
            "reasoning": f"「{word}」は既存登録なし（白地図候補）。コンセプトとの親和性スコア: {scored[i-1][1]}",
            "classes": nice_classes,
            "status": "要J-PlatPat確認",
        })

    lines: list[str] = [
        f"## 商標提案（対象区分: {nice_classes}）\n",
        f"コンセプト: {concept_description[:60]}...\n" if len(concept_description) > 60 else f"コンセプト: {concept_description}\n",
        f"既存Drip系修飾語（要注意）: {list(existing.keys())[:8]}\n",
        "",
    ]
    for p in proposals:
        lines.append(f"### 第{p['rank']}案: **{p['name']}**")
        lines.append(f"- 理由: {p['reasoning']}")
        lines.append(f"- 出願検討区分: {p['classes']}")
        lines.append(f"- ステータス: {p['status']}")
        lines.append("")

    lines.append("**注意**: 上記はすべて出願前にJ-PlatPat（https://www.j-platpat.inpit.go.jp/）で最終確認が必要です。")

    return {"text": "\n".join(lines)}


def tool_generate_sketch(
    concept_name: str,
    shape_type: str,
    material: str = "SUS304",
    dimensions: dict | None = None,
    features: list[str] | None = None,
) -> dict:
    """3Dデザインスケッチを生成する。"""
    BG = "#0F1923"
    TEXT = "#E8EDF2"
    if dimensions is None:
        dimensions = {}
    if features is None:
        features = []

    fig = plt.figure(figsize=(14, 6), facecolor=BG)
    fig.suptitle(f"{concept_name}  —  {shape_type} / {material}", color=TEXT, fontsize=14, y=0.98)

    views = [
        ("perspective", "パース"),
        ("top", "上面図"),
        ("front", "正面図"),
        ("side", "側面図"),
        ("section", "断面図"),
    ]

    L = np.array([-0.4, 0.3, 0.85])
    L = L / np.linalg.norm(L)

    for col_idx, (view, label) in enumerate(views):
        ax = fig.add_subplot(1, 5, col_idx + 1,
                             projection="3d" if view == "perspective" else None,
                             facecolor=BG)
        ax.set_title(label, color=TEXT, fontsize=9, pad=6)

        d_top = dimensions.get("diameter_top", 120)
        d_bot = dimensions.get("diameter_bottom", 40)
        h = dimensions.get("height", 80)
        r_top = d_top / 2
        r_bot = d_bot / 2

        if view == "perspective":
            _draw_3d_shape(ax, shape_type, r_top, r_bot, h, L, material)
            ax.set_box_aspect([1, 1, 1.2])
        elif view == "top":
            theta = np.linspace(0, 2 * np.pi, 64)
            ax.plot(r_top * np.cos(theta), r_top * np.sin(theta), color="#2471A3", linewidth=1.5)
            ax.plot(r_bot * np.cos(theta), r_bot * np.sin(theta), color="#2471A3",
                    linewidth=1, linestyle="--")
            ax.set_xlim(-r_top * 1.3, r_top * 1.3)
            ax.set_ylim(-r_top * 1.3, r_top * 1.3)
            ax.set_aspect("equal")
            ax.axis("off")
        elif view == "front":
            ax.fill([-r_top, -r_bot, r_bot, r_top],
                    [0, h, h, 0], color="#2471A3", alpha=0.6, edgecolor="#5DADE2", linewidth=1.5)
            ax.set_xlim(-r_top * 1.3, r_top * 1.3)
            ax.set_ylim(-h * 0.1, h * 1.3)
            ax.axis("off")
            ax.text(0, -h * 0.05, f"φ{d_top}→φ{d_bot}", ha="center", color=TEXT, fontsize=7)
        elif view == "side":
            ax.fill([-r_top, -r_bot, r_bot, r_top],
                    [0, h, h, 0], color="#1A5276", alpha=0.6, edgecolor="#5DADE2", linewidth=1.5)
            ax.set_xlim(-r_top * 1.3, r_top * 1.3)
            ax.set_ylim(-h * 0.1, h * 1.3)
            ax.axis("off")
            ax.text(0, h * 1.1, f"H={h}mm", ha="center", color=TEXT, fontsize=7)
        elif view == "section":
            xs = [-r_top, -r_bot, r_bot, r_top, -r_top]
            ys = [0, h, h, 0, 0]
            ax.fill(xs, ys, color="#0F1923", edgecolor="#E74C3C", linewidth=1.5, linestyle="-")
            ax.set_xlim(-r_top * 1.3, r_top * 1.3)
            ax.set_ylim(-h * 0.1, h * 1.3)
            ax.axis("off")

        if view != "perspective":
            ax.set_facecolor(BG)
            for spine in (ax.spines.values() if hasattr(ax, "spines") else []):
                spine.set_edgecolor("#334455")

    # 仕様テキスト
    spec_lines = [
        f"形状: {shape_type}",
        f"素材: {material}",
        f"φ上部: {dimensions.get('diameter_top', 120)}mm",
        f"φ下部: {dimensions.get('diameter_bottom', 40)}mm",
        f"高さ: {dimensions.get('height', 80)}mm",
    ] + (features[:4] if features else [])
    fig.text(0.01, 0.02, "\n".join(spec_lines), color="#778899",
             fontsize=7, va="bottom", family="monospace")

    return {
        "text": (
            f"{concept_name} スケッチ生成完了\n"
            f"形状: {shape_type} / 素材: {material}\n"
            f"寸法: φ{dimensions.get('diameter_top', 120)}→"
            f"φ{dimensions.get('diameter_bottom', 40)} / H={dimensions.get('height', 80)}mm\n"
            f"特徴: {', '.join(features) if features else 'なし'}"
        ),
        "chart_b64": _fig_to_b64(fig),
        "chart_title": f"{concept_name} デザインスケッチ",
    }


def _draw_3d_shape(
    ax: plt.Axes, shape_type: str, r_top: float, r_bot: float, h: float,
    L: np.ndarray, material: str
) -> None:
    """3Dパースビューを描画する。"""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    ax.set_facecolor("#0F1923")

    is_metal = any(k in material.upper() for k in ["SUS", "ステンレス", "チタン", "銅", "真鍮"])
    base_color = np.array([0.55, 0.65, 0.72]) if is_metal else np.array([0.65, 0.55, 0.45])

    if shape_type == "diamond":
        # 正多角錐 (ダイヤモンド型)
        n_faces = 8
        angles = np.linspace(0, 2 * np.pi, n_faces, endpoint=False)
        apex = np.array([0, 0, h])
        base_pts = np.array([[r_top * np.cos(a), r_top * np.sin(a), 0] for a in angles])
        polys = []
        colors_faces = []
        for i in range(n_faces):
            p0 = base_pts[i]
            p1 = base_pts[(i + 1) % n_faces]
            face = [p0, p1, apex]
            polys.append(face)
            v1 = p1 - p0
            v2 = apex - p0
            normal = np.cross(v1, v2)
            nlen = np.linalg.norm(normal)
            if nlen > 0:
                normal = normal / nlen
            diffuse = max(0.0, float(np.dot(normal, L)))
            specular = float(np.dot(normal, L / np.linalg.norm(L + np.array([0, 0, 1])) * 0.5 + 0.5)) ** 32 if is_metal else 0
            brightness = 0.15 + 0.65 * diffuse + 0.2 * specular
            c = np.clip(base_color * brightness, 0, 1)
            colors_faces.append(c)

        coll = Poly3DCollection(polys, alpha=0.92, linewidth=0.4, edgecolor="#AAB8C2")
        coll.set_facecolor([list(c) + [0.92] for c in colors_faces])
        ax.add_collection3d(coll)

    else:
        # 円錐/台形 (回転体)
        n_seg = 32
        theta = np.linspace(0, 2 * np.pi, n_seg, endpoint=False)
        polys = []
        colors_faces = []
        for i in range(n_seg):
            t0, t1 = theta[i], theta[(i + 1) % n_seg]
            if shape_type == "cone":
                p00 = np.array([r_top * np.cos(t0), r_top * np.sin(t0), 0])
                p10 = np.array([r_top * np.cos(t1), r_top * np.sin(t1), 0])
                p_apex = np.array([0, 0, h])
                face = [p00, p10, p_apex]
            else:
                p00 = np.array([r_top * np.cos(t0), r_top * np.sin(t0), 0])
                p10 = np.array([r_top * np.cos(t1), r_top * np.sin(t1), 0])
                p11 = np.array([r_bot * np.cos(t1), r_bot * np.sin(t1), h])
                p01 = np.array([r_bot * np.cos(t0), r_bot * np.sin(t0), h])
                face = [p00, p10, p11, p01]

            polys.append(face)
            v_pts = np.array(face)
            v1 = v_pts[1] - v_pts[0]
            v2 = v_pts[-1] - v_pts[0]
            normal = np.cross(v1, v2)
            nlen = np.linalg.norm(normal)
            if nlen > 0:
                normal = normal / nlen
            diffuse = max(0.0, float(np.dot(normal, L)))
            half = (L + np.array([0, 0, 1]))
            hlen = np.linalg.norm(half)
            specular = float(np.dot(normal, half / hlen)) ** 32 if (is_metal and hlen > 0) else 0
            brightness = 0.15 + 0.65 * diffuse + 0.2 * specular
            c = np.clip(base_color * brightness, 0, 1)
            colors_faces.append(c)

        coll = Poly3DCollection(polys, alpha=0.9, linewidth=0.2, edgecolor="#6699AA")
        coll.set_facecolor([list(c) + [0.9] for c in colors_faces])
        ax.add_collection3d(coll)

    lim = r_top * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, h * 1.2)
    ax.set_axis_off()


# ─────────────────────────────────────────────
# ユーティリティ
# ─────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """候補列名のいずれかにマッチする列を返す。"""
    for c in candidates:
        if c in df.columns:
            return c
        for col in df.columns:
            if c.lower() in col.lower():
                return col
    return None


def parse_next_actions(text: str) -> tuple[str, list[dict]]:
    """
    レスポンステキストから <next_actions> ブロックを抽出・除去する。
    Returns: (cleaned_text, actions_list)
    """
    pattern = r"<next_actions>\s*(.*?)\s*</next_actions>"
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return text, []
    raw_json = match.group(1).strip()
    cleaned = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    try:
        actions = json.loads(raw_json)
        if isinstance(actions, list):
            return cleaned, actions
    except (json.JSONDecodeError, ValueError):
        pass
    return cleaned, []


def detect_current_phase(messages: list[dict]) -> int:
    """会話履歴からおおよその現在フェーズを推定する（1-9、不明は0）。"""
    text = " ".join(
        m.get("content", "") for m in messages if isinstance(m.get("content"), str)
    )
    for phase in reversed(PHASES):
        if any(kw in text for kw in phase["keywords"]):
            return phase["num"]
    return 0


def _dispatch_tool(tool_name: str, tool_input: dict) -> str:
    """ツール名と入力をディスパッチして結果をJSON文字列で返す。"""
    try:
        if tool_name == "analyze_design_patents":
            result = tool_analyze_design_patents(**tool_input)
        elif tool_name == "analyze_trademarks":
            result = tool_analyze_trademarks(**tool_input)
        elif tool_name == "generate_shape_matrix":
            result = tool_generate_shape_matrix(**tool_input)
        elif tool_name == "score_concepts":
            result = tool_score_concepts(**tool_input)
        elif tool_name == "propose_trademarks":
            result = tool_propose_trademarks(**tool_input)
        elif tool_name == "generate_sketch":
            result = tool_generate_sketch(**tool_input)
        else:
            result = {"text": f"未知のツール: {tool_name}"}
    except Exception as e:
        result = {"text": f"ツール実行エラー ({tool_name}): {e}"}

    return json.dumps(result, ensure_ascii=False)


# ─────────────────────────────────────────────
# Claude APIアgentic loop
# ─────────────────────────────────────────────

def run_claude(user_message: str, api_key: str) -> tuple[str, list[dict], list[dict]]:
    """
    Claude APIを呼び出し、tool_useが完了するまでループする。
    Returns: (final_text, charts, next_actions)
    """
    client = anthropic.Anthropic(api_key=api_key)

    messages: list[dict] = []
    history: list[dict] = st.session_state.get("messages", [])

    # 過去の会話履歴（直近10ターン）
    for msg in history[-20:]:
        if msg["role"] in ("user", "assistant") and isinstance(msg.get("content"), str):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # PDFが添付されている場合はユーザーメッセージに追加
    pdf_b64: str | None = st.session_state.get("pdf_b64")
    if pdf_b64:
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                    "title": "意匠画像PDF",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": user_message},
            ],
        })
    else:
        messages.append({"role": "user", "content": user_message})

    charts: list[dict] = []
    final_text = ""
    max_iterations = 8

    for _iter in range(max_iterations):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # レスポンスをメッセージに追加
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                final_text += block.text
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })

        messages.append({"role": "assistant", "content": assistant_content})

        # stop_reason チェック
        if response.stop_reason != "tool_use":
            break

        # tool_use ブロックを処理
        tool_results: list[dict] = []
        for block in response.content:
            if block.type != "tool_use":
                continue

            result_json = _dispatch_tool(block.name, block.input)
            result_dict = json.loads(result_json)

            # チャートを抽出
            if "chart_b64" in result_dict:
                charts.append({
                    "b64": result_dict["chart_b64"],
                    "title": result_dict.get("chart_title", block.name),
                })
            if "charts" in result_dict:
                for ch in result_dict["charts"]:
                    charts.append({"b64": ch, "title": block.name})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": result_dict.get("text", "完了"),
            })

        messages.append({"role": "user", "content": tool_results})

    # next_actions を抽出
    final_text, next_actions = parse_next_actions(final_text)

    # フォールバック: Claudeが出力しなかった場合は静的候補を使用
    if not next_actions:
        current_phase = detect_current_phase(st.session_state.get("messages", []))
        next_actions = PHASE_NEXT_ACTIONS.get(current_phase, PHASE_NEXT_ACTIONS[0])

    return final_text, charts, next_actions


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

def init_session() -> None:
    """セッション状態を初期化する。"""
    defaults: dict[str, Any] = {
        "messages": [],
        "design_df": None,
        "trademark_df": None,
        "pdf_bytes": None,
        "pdf_b64": None,
        "api_key": "",
        "analysis_results": {},
        "next_actions": [],          # 最新のアクションボタン候補
        "pending_prompt": None,      # ボタン押下で注入するプロンプト
        "current_phase": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def sidebar() -> str | None:
    """サイドバーを描画する。API keyを返す。"""
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#2471A3;margin-bottom:4px'>⚙️ 設定</h2>",
            unsafe_allow_html=True,
        )
        api_key = st.text_input(
            "Anthropic API Key",
            value=st.session_state.get("api_key", ""),
            type="password",
            placeholder="sk-ant-...",
            help="Anthropic Console で発行したAPIキーを入力",
        )
        if api_key:
            st.session_state["api_key"] = api_key

        st.divider()

        # 意匠CSV
        st.markdown("**📂 意匠データ (CSV)**")
        design_file = st.file_uploader(
            "意匠CSV", type=["csv"], key="design_upload", label_visibility="collapsed"
        )
        if design_file is not None:
            try:
                df = pd.read_csv(design_file, encoding="utf-8-sig")
                st.session_state["design_df"] = df
                st.success(f"✓ {len(df)}件 読み込み済み")
            except Exception as e:
                st.error(f"読み込みエラー: {e}")

        if st.session_state.get("design_df") is not None:
            df = st.session_state["design_df"]
            st.caption(f"列: {', '.join(df.columns[:5])}{'…' if len(df.columns) > 5 else ''}")

        # 商標CSV
        st.markdown("**📂 商標データ (CSV)**")
        tm_file = st.file_uploader(
            "商標CSV", type=["csv"], key="tm_upload", label_visibility="collapsed"
        )
        if tm_file is not None:
            try:
                df_tm = pd.read_csv(tm_file, encoding="utf-8-sig")
                st.session_state["trademark_df"] = df_tm
                st.success(f"✓ {len(df_tm)}件 読み込み済み")
            except Exception as e:
                st.error(f"読み込みエラー: {e}")

        # 意匠画像PDF
        st.markdown("**📄 意匠画像 (PDF)**")
        pdf_file = st.file_uploader(
            "意匠PDF", type=["pdf"], key="pdf_upload", label_visibility="collapsed"
        )
        if pdf_file is not None:
            pdf_bytes = pdf_file.read()
            st.session_state["pdf_bytes"] = pdf_bytes
            st.session_state["pdf_b64"] = base64.b64encode(pdf_bytes).decode()
            st.success(f"✓ {len(pdf_bytes)//1024}KB 読み込み済み")

        st.divider()

        # フェーズ進捗
        current_ph = st.session_state.get("current_phase", 0)
        st.markdown("<small style='color:#778899'>フェーズ進捗</small>", unsafe_allow_html=True)
        st.progress(current_ph / 9)

        with st.expander("📋 フレームワーク 9フェーズ", expanded=True):
            for ph in PHASES:
                done = ph["num"] <= current_ph
                icon = "✅" if done else "○"
                color = ph["color"] if done else "#445566"
                st.markdown(
                    f"<div style='margin:2px 0'>"
                    f"<span style='color:{color}'>{icon} Ph{ph['num']}</span> "
                    f"<b style='color:{color}'>{ph['en']}</b> "
                    f"<small style='color:#778899'>{ph['ja']}</small>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        st.divider()

        # 会話リセット
        if st.button("🗑️ 会話をリセット", use_container_width=True):
            st.session_state["messages"] = []
            st.session_state["analysis_results"] = {}
            st.rerun()

        # データ状態
        st.markdown(
            "<small style='color:#556677'>データ状態:</small>",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        cols[0].metric("意匠", "✓" if st.session_state.get("design_df") is not None else "—")
        cols[1].metric("商標", "✓" if st.session_state.get("trademark_df") is not None else "—")
        cols[2].metric("PDF", "✓" if st.session_state.get("pdf_b64") else "—")

    return api_key or st.session_state.get("api_key", "")


def render_message(msg: dict) -> None:
    """1メッセージを描画する。"""
    role = msg["role"]
    content = msg["content"]
    charts = msg.get("charts", [])

    with st.chat_message(role):
        if isinstance(content, str):
            st.markdown(content)
        for chart in charts:
            st.image(
                base64.b64decode(chart["b64"]),
                caption=chart.get("title", ""),
                use_container_width=True,
            )


def _invoke_claude(prompt: str, api_key: str) -> None:
    """Claudeを呼び出してUIを更新する共通処理。"""
    user_msg: dict[str, Any] = {"role": "user", "content": prompt, "charts": []}
    st.session_state["messages"].append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        with st.spinner("分析中…"):
            try:
                response_text, charts, next_actions = run_claude(prompt, api_key)
            except anthropic.AuthenticationError:
                st.error("APIキーが無効です。サイドバーで確認してください。")
                st.session_state["messages"].pop()
                st.stop()
            except anthropic.RateLimitError:
                st.error("レート制限に達しました。しばらく待ってから再試行してください。")
                st.session_state["messages"].pop()
                st.stop()
            except Exception as e:
                st.error(f"エラー: {e}")
                st.session_state["messages"].pop()
                st.stop()

        st.markdown(response_text)
        for chart in charts:
            st.image(
                base64.b64decode(chart["b64"]),
                caption=chart.get("title", ""),
                use_container_width=True,
            )

    assistant_msg: dict[str, Any] = {
        "role": "assistant",
        "content": response_text,
        "charts": charts,
    }
    st.session_state["messages"].append(assistant_msg)
    st.session_state["next_actions"] = next_actions
    st.session_state["current_phase"] = detect_current_phase(st.session_state["messages"])


def _render_setup_checklist(api_key: str) -> None:
    """初期セットアップ状況チェックリストとスタートボタンを表示する。"""
    has_key = bool(api_key)
    has_design = st.session_state.get("design_df") is not None
    has_tm = st.session_state.get("trademark_df") is not None
    has_pdf = bool(st.session_state.get("pdf_b64"))

    st.markdown(
        """
        <div style="background:#111E2B;border:1px solid #2471A3;border-radius:10px;
                    padding:20px 24px;margin-bottom:20px">
        <h3 style="color:#2471A3;margin:0 0 12px 0;font-size:1.1rem">
            📋 セットアップチェック
        </h3>
        """,
        unsafe_allow_html=True,
    )

    items = [
        (has_key,    "APIキー入力済み",      "サイドバーで sk-ant-... を入力"),
        (has_design, "意匠CSV アップロード済み", "サイドバーから意匠CSVをアップロード"),
        (has_tm,     "商標CSV アップロード済み", "（任意）商標CSVをアップロード"),
        (has_pdf,    "意匠PDF アップロード済み", "（任意）意匠画像PDFをアップロード"),
    ]
    for ok, label, hint in items:
        icon = "✅" if ok else "⬜"
        color = "#E8EDF2" if ok else "#556677"
        sub = "" if ok else f"<br><small style='color:#445566;margin-left:22px'>{hint}</small>"
        st.markdown(
            f"<div style='margin:4px 0;color:{color}'>{icon} {label}{sub}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)

    ready = has_key and (has_design or has_tm or has_pdf)

    if ready:
        st.markdown(
            "<p style='color:#778899;font-size:0.85rem;margin-bottom:8px'>"
            "データが揃っています。分析を開始できます。</p>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶ 分析をスタート", type="primary", use_container_width=True):
                st.session_state["pending_prompt"] = (
                    "コーヒードリッパーのIPランドスケープ分析を開始してください。"
                    "まずPh1 SCOPEの設定を行い、続けてアップロードされたデータの概要分析（Ph3 READ）を実行してください。"
                )
                st.rerun()
        with col2:
            if st.button("Ph2 データ取得ガイド", use_container_width=True):
                st.session_state["pending_prompt"] = (
                    "Ph2 GATHERとして、J-PlatPatから意匠データと商標データを取得する手順を詳しく教えてください。"
                )
                st.rerun()
    elif not has_key:
        st.info("まずサイドバーにAnthropicのAPIキーを入力してください。")
    else:
        st.markdown(
            "<p style='color:#778899;font-size:0.85rem'>"
            "データなしでも分析ガイドを利用できます。</p>",
            unsafe_allow_html=True,
        )
        if st.button("▶ データなしで始める（ガイドモード）", use_container_width=True):
            st.session_state["pending_prompt"] = (
                "データはまだありませんが、コーヒードリッパーのIPランドスケープ分析の進め方を"
                "Ph1〜Ph9のフレームワークに沿ってガイドしてください。"
                "まずPh1 SCOPEの設定から始めてください。"
            )
            st.rerun()


def _render_action_buttons(api_key: str) -> None:
    """次のアクションボタン群を表示する。"""
    actions: list[dict] = st.session_state.get("next_actions", [])
    if not actions:
        return

    st.markdown(
        "<div style='margin:8px 0 4px 0'>"
        "<small style='color:#556677'>▼ 次のステップ</small>"
        "</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(min(len(actions), 3))
    for i, action in enumerate(actions[:3]):
        with cols[i]:
            label = action.get("label", f"アクション{i+1}")
            prompt = action.get("prompt", label)
            if st.button(label, key=f"action_{i}_{len(st.session_state['messages'])}",
                         use_container_width=True):
                st.session_state["pending_prompt"] = prompt
                st.rerun()


def main() -> None:
    """メインエントリーポイント。"""
    st.set_page_config(
        page_title="Design IP Landscape AI",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
        <style>
        .stChatMessage { border-radius: 10px; }
        .stChatMessage[data-testid="stChatMessageUser"] {
            background-color: #1A2633;
        }
        .stChatMessage[data-testid="stChatMessageAssistant"] {
            background-color: #111E2B;
        }
        code { color: #5DADE2 !important; }
        div[data-testid="stHorizontalBlock"] > div > div > button {
            border: 1px solid #2471A3 !important;
            border-radius: 6px !important;
            font-size: 0.82rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    init_session()
    api_key = sidebar()

    # ヘッダー
    st.markdown(
        """
        <div style="padding:8px 0 2px 0">
            <h1 style="color:#2471A3;margin:0;font-size:1.5rem">
                🔬 Design IP Landscape AI
            </h1>
            <p style="color:#778899;margin:2px 0 0 0;font-size:0.85rem">
                意匠・商標 IPランドスケープ — 9フェーズ全支援 | Powered by Claude
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # pending_prompt（ボタン押下）があれば即処理
    pending = st.session_state.pop("pending_prompt", None)
    if pending and api_key:
        _invoke_claude(pending, api_key)
        st.rerun()

    # 初回：セットアップ画面
    if not st.session_state["messages"]:
        _render_setup_checklist(api_key)
    else:
        # 過去メッセージ表示
        for msg in st.session_state["messages"]:
            render_message(msg)

        # 次のアクションボタン
        _render_action_buttons(api_key)

    # 自由入力チャット
    prompt = st.chat_input("自由入力…（または上のボタンで次のステップへ）")
    if prompt:
        if not api_key:
            st.error("APIキーをサイドバーに入力してください。")
            st.stop()
        _invoke_claude(prompt, api_key)
        st.rerun()


if __name__ == "__main__":
    main()
