"""
商標分析モジュール

統計ランキング・ランドスケープ・ホワイトスペース分析
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from trademark_parser import NICE_CLASS_DESC, explode_by_class

ALL_CLASSES = list(range(1, 46))  # Nice 1〜45


# ──────────────────────────────────────────────────────────
# 統計ランキング
# ──────────────────────────────────────────────────────────

def class_ranking(records: list[dict], top_n: int = 45) -> pd.DataFrame:
    """区分別登録件数ランキング。"""
    exploded = explode_by_class(records)
    cnt = Counter(r["single_class"] for r in exploded if r["single_class"])
    rows = []
    for cls in ALL_CLASSES:
        count = cnt.get(cls, 0)
        rows.append({
            "区分":   cls,
            "説明":   NICE_CLASS_DESC.get(cls, ""),
            "件数":   count,
            "割合(%)": 0.0,
        })
    total = sum(r["件数"] for r in rows)
    if total > 0:
        for r in rows:
            r["割合(%)"] = round(r["件数"] / total * 100, 1)
    df = pd.DataFrame(rows).sort_values("件数", ascending=False).reset_index(drop=True)
    df.index = df.index + 1
    return df.head(top_n)


def applicant_ranking(records: list[dict], top_n: int = 20) -> pd.DataFrame:
    """出願人別登録件数ランキング。"""
    cnt = Counter(r["applicant"] for r in records if r["applicant"])
    rows = [{"出願人": a, "件数": c} for a, c in cnt.most_common(top_n)]
    df = pd.DataFrame(rows)
    df.index = df.index + 1
    return df


def yearly_trend(records: list[dict]) -> pd.DataFrame:
    """年別出願件数トレンド。"""
    cnt = Counter(r["app_year"] for r in records if r["app_year"])
    years = sorted(cnt.keys())
    df = pd.DataFrame({"年": years, "件数": [cnt[y] for y in years]})
    return df


def status_distribution(records: list[dict]) -> pd.DataFrame:
    """ステータス別件数。"""
    cnt = Counter(r["status"] for r in records if r["status"])
    df = pd.DataFrame([{"ステータス": s, "件数": c} for s, c in cnt.most_common()])
    return df


def regtype_distribution(records: list[dict]) -> pd.DataFrame:
    """国内/国際/出願別件数。"""
    label_map = {
        "domestic":      "国内登録",
        "international": "国際登録",
        "application":   "出願中",
    }
    cnt = Counter(label_map.get(r["reg_type"], r["reg_type"]) for r in records)
    df = pd.DataFrame([{"種別": k, "件数": v} for k, v in cnt.most_common()])
    return df


def multiclass_stats(records: list[dict]) -> pd.DataFrame:
    """1件で指定している区分数の分布。"""
    cnt = Counter(len(r["classes"]) for r in records)
    df = pd.DataFrame(
        [{"区分数": k, "件数": v} for k, v in sorted(cnt.items())]
    )
    return df


# ──────────────────────────────────────────────────────────
# ランドスケープ (PCA 2D)
# ──────────────────────────────────────────────────────────

def compute_trademark_landscape(
    records: list[dict],
    n_clusters: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    商標をTF-IDF(称呼) + クラスone-hot で特徴量化し PCA 2D へ変換。

    Returns
    -------
    coords_2d : np.ndarray (N, 2)
    labels    : np.ndarray (N,) KMeans クラスタラベル
    """
    n = len(records)
    if n < 3:
        return np.zeros((n, 2)), np.zeros(n, dtype=int)

    # ── テキスト特徴量 (称呼 + 商標名) ───────────────────
    corpus = []
    for r in records:
        text = " ".join([r.get("trademark", "")] + r.get("readings", [])[:3])
        corpus.append(text if text.strip() else "不明")

    try:
        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 3),
            max_features=500, sublinear_tf=True, min_df=1,
        )
        X_text = vec.fit_transform(corpus).toarray().astype(np.float32)
        X_text = normalize(X_text)
    except Exception:
        X_text = np.zeros((n, 1), dtype=np.float32)

    # ── 区分 one-hot ─────────────────────────────────────
    X_cls = np.zeros((n, 45), dtype=np.float32)
    for i, r in enumerate(records):
        for cls in r.get("classes", []):
            if 1 <= cls <= 45:
                X_cls[i, cls - 1] = 1.0

    # ── 結合・PCA ────────────────────────────────────────
    X = np.hstack([X_text * 0.5, X_cls * 0.5])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    n_comp = min(2, X_scaled.shape[0] - 1, X_scaled.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X_scaled).astype(np.float32)

    if coords.shape[1] < 2:
        coords = np.hstack([coords, np.zeros((n, 1), dtype=np.float32)])

    # KMeans
    k = min(n_clusters, n)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled).astype(int)

    return coords, labels


# ──────────────────────────────────────────────────────────
# ホワイトスペース分析
# ──────────────────────────────────────────────────────────

def class_coverage(records: list[dict]) -> dict:
    """
    ニース区分 1〜45 のカバレッジを返す。

    Returns
    -------
    {
      "covered":  list[int]  件数 > 0 の区分
      "vacant":   list[dict] 件数0の区分 (whitespace)
      "counts":   dict[int, int]
    }
    """
    exploded = explode_by_class(records)
    counts = Counter(r["single_class"] for r in exploded if r["single_class"])

    covered = [c for c in ALL_CLASSES if counts.get(c, 0) > 0]
    vacant  = [
        {"区分": c, "説明": NICE_CLASS_DESC.get(c, ""), "件数": 0}
        for c in ALL_CLASSES if counts.get(c, 0) == 0
    ]

    return {"covered": covered, "vacant": vacant, "counts": dict(counts)}


def class_cooccurrence_matrix(records: list[dict]) -> pd.DataFrame:
    """
    区分ペア共起マトリックス (対称行列)。
    値 = その区分ペアが同一商標に含まれる件数。
    値が 0 = 両区分を同時カバーする商標がない → ホワイトスペース。
    """
    mat = np.zeros((45, 45), dtype=int)
    for r in records:
        classes = [c - 1 for c in r.get("classes", []) if 1 <= c <= 45]
        for i, a in enumerate(classes):
            for b in classes[i:]:
                mat[a, b] += 1
                if a != b:
                    mat[b, a] += 1

    labels = [f"{c}" for c in range(1, 46)]
    return pd.DataFrame(mat, index=labels, columns=labels)


def applicant_class_matrix(
    records: list[dict],
    top_n: int = 20,
) -> pd.DataFrame:
    """
    出願人(top_n) × 区分(1〜45) のヒートマップ用行列。
    値 = 該当出願人が当該区分を保有する件数。
    """
    # 上位出願人を特定
    cnt = Counter(r["applicant"] for r in records if r["applicant"])
    top_applicants = [a for a, _ in cnt.most_common(top_n)]

    mat: dict[str, dict[int, int]] = {a: defaultdict(int) for a in top_applicants}
    for r in records:
        appl = r.get("applicant", "")
        if appl in mat:
            for cls in r.get("classes", []):
                if 1 <= cls <= 45:
                    mat[appl][cls] += 1

    # 使われている区分のみ列に使う
    all_used = sorted(set(
        cls
        for r in records for cls in r.get("classes", [])
        if 1 <= cls <= 45
    ))

    rows = []
    for appl in top_applicants:
        row = {"出願人": appl[:20]}
        for cls in all_used:
            row[str(cls)] = mat[appl].get(cls, 0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("出願人")
    df.columns = [str(c) for c in all_used]
    return df


def whitespace_class_pairs(
    records: list[dict],
    min_class_count: int = 3,
) -> pd.DataFrame:
    """
    個別には登録実績のある区分同士で、組み合わせ件数が少ないペアを返す。
    = 「単独では混雑しているが組み合わせでは空白」のホワイトスペース候補。

    Returns
    -------
    DataFrame: class_a, class_b, desc_a, desc_b, count, opportunity_score
    """
    counts = Counter(
        cls
        for r in records
        for cls in r.get("classes", [])
        if 1 <= cls <= 45
    )
    # 一定件数以上ある区分のみ対象
    active_classes = [c for c, n in counts.items() if n >= min_class_count]

    cooc = class_cooccurrence_matrix(records)

    rows = []
    for i, a in enumerate(active_classes):
        for b in active_classes[i + 1:]:
            val = int(cooc.iloc[a - 1, b - 1])
            if val == 0:
                score = counts[a] + counts[b]  # 両区分の単独登録数の合計
                rows.append({
                    "区分A":    a,
                    "区分A説明": NICE_CLASS_DESC.get(a, ""),
                    "区分B":    b,
                    "区分B説明": NICE_CLASS_DESC.get(b, ""),
                    "共起件数": val,
                    "機会スコア": score,
                })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("機会スコア", ascending=False).reset_index(drop=True)


def yearly_class_trend(records: list[dict]) -> pd.DataFrame:
    """
    年×区分 の件数ピボットテーブル。
    成長区分・衰退区分の特定に使用。
    """
    exploded = explode_by_class(records)
    rows = [
        {"年": r["app_year"], "区分": r["single_class"]}
        for r in exploded
        if r["app_year"] and r["single_class"]
    ]
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    pivot = df.groupby(["年", "区分"]).size().unstack(fill_value=0)
    pivot.columns = [str(c) for c in pivot.columns]
    return pivot


# ──────────────────────────────────────────────────────────
# Gemini レポート用サマリー構築
# ──────────────────────────────────────────────────────────

def build_report_context(records: list[dict]) -> str:
    """
    Gemini 戦略レポート生成用のコンテキスト文字列を構築する。
    """
    n = len(records)
    cls_rank  = class_ranking(records, top_n=10)
    appl_rank = applicant_ranking(records, top_n=10)
    coverage  = class_coverage(records)
    vacant    = coverage["vacant"]

    lines = [
        f"## 分析データ概要",
        f"- 商標総数: {n} 件",
        f"- カバー区分数: {len(coverage['covered'])} / 45",
        f"- 空白区分数: {len(vacant)}",
        "",
        "## 区分別登録件数 Top10",
    ]
    for _, row in cls_rank.head(10).iterrows():
        lines.append(f"- 第{row['区分']}類 ({row['説明']}): {row['件数']}件")

    lines += ["", "## 出願人 Top10"]
    for _, row in appl_rank.head(10).iterrows():
        lines.append(f"- {row['出願人']}: {row['件数']}件")

    if vacant:
        lines += ["", "## 空白区分 (登録実績なし)"]
        for v in vacant[:10]:
            lines.append(f"- 第{v['区分']}類: {v['説明']}")

    return "\n".join(lines)
