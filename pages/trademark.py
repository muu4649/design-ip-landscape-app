"""
商標ランドスケープ分析

タブ:
  1. 📥 データ取込
  2. 📊 統計・ランキング
  3. 🗺️ 商標ランドスケープ
  4. 🔍 ホワイトスペース分析
  5. 📝 AI 戦略レポート
"""

from __future__ import annotations

import sys
import os

# pages/ から親ディレクトリのモジュールを import できるようにする
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from trademark_parser import parse_trademark_csv, NICE_CLASS_DESC
from trademark_analyzer import (
    class_ranking,
    applicant_ranking,
    yearly_trend,
    status_distribution,
    regtype_distribution,
    multiclass_stats,
    compute_trademark_landscape,
    class_coverage,
    class_cooccurrence_matrix,
    applicant_class_matrix,
    whitespace_class_pairs,
    yearly_class_trend,
    build_report_context,
)

# ──────────────────────────────────────────────────────────
# ページ設定
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="商標ランドスケープ",
    page_icon="™️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# Session State 初期化
# ──────────────────────────────────────────────────────────

_TM_DEFAULTS = {
    "tm_records":        [],
    "tm_coords":         None,
    "tm_labels":         None,
    "tm_landscape_done": False,
    "tm_report":         "",
    "tm_report_done":    False,
    "tm_n_clusters":     8,
    "tm_api_key":        "",
    "tm_api_validated":  False,
    "tm_llm_client":     None,
    "tm_objective":      "「ドリップ」関連商標の競合ポジションと未登録空白領域を把握する",
}
for k, v in _TM_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────
# サイドバー
# ──────────────────────────────────────────────────────────

with st.sidebar:
    st.title("™️ 商標ランドスケープ")
    st.caption("J-PlatPat 商標 CSV 分析")
    st.divider()

    st.subheader("⚙️ 分析設定")
    st.session_state.tm_n_clusters = st.slider(
        "ランドスケープ クラスター数",
        min_value=2, max_value=15,
        value=st.session_state.tm_n_clusters,
    )

    st.divider()
    st.subheader("🔑 Gemini API (レポート生成用)")
    tm_api_key_input = st.text_input(
        "APIキー",
        value=st.session_state.tm_api_key,
        type="password",
        placeholder="AIzaSy...",
    )
    if st.button("検証", key="tm_validate"):
        if tm_api_key_input:
            try:
                from gemini_client import LLMClient, validate_api_key
                ok, msg = validate_api_key(tm_api_key_input)
                if ok:
                    st.session_state.tm_api_key       = tm_api_key_input
                    st.session_state.tm_api_validated  = True
                    st.session_state.tm_llm_client     = LLMClient(tm_api_key_input)
                    st.success(msg)
                else:
                    st.error(msg)
            except ImportError:
                st.error("gemini_client が見つかりません")
        else:
            st.warning("APIキーを入力してください")

    if st.session_state.tm_api_validated:
        st.success("APIキー有効")

    if st.session_state.tm_records:
        st.divider()
        st.caption(f"読込済: {len(st.session_state.tm_records)} 件")

# ──────────────────────────────────────────────────────────
# タブ
# ──────────────────────────────────────────────────────────

tab_upload, tab_stats, tab_landscape, tab_whitespace, tab_report = st.tabs([
    "📥 データ取込",
    "📊 統計・ランキング",
    "🗺️ 商標ランドスケープ",
    "🔍 ホワイトスペース分析",
    "📝 AI 戦略レポート",
])


# ══════════════════════════════════════════════════════════
# Tab 1: データ取込
# ══════════════════════════════════════════════════════════

with tab_upload:
    st.header("📥 データ取込")

    col_up, col_info = st.columns([3, 2])

    with col_up:
        uploaded = st.file_uploader(
            "J-PlatPat 商標検索結果 CSV",
            type=["csv"],
            help="J-PlatPat > 商標検索 > 結果一覧 > CSV ダウンロード",
        )

        if uploaded is not None:
            with st.spinner("解析中..."):
                try:
                    records = parse_trademark_csv(uploaded.read())
                    st.session_state.tm_records        = records
                    st.session_state.tm_coords         = None
                    st.session_state.tm_labels         = None
                    st.session_state.tm_landscape_done = False
                    st.session_state.tm_report         = ""
                    st.session_state.tm_report_done    = False
                    st.success(f"{len(records)} 件の商標データを読み込みました")
                except Exception as e:
                    st.error(f"読み込みエラー: {e}")

    with col_info:
        if st.session_state.tm_records:
            records = st.session_state.tm_records
            st.metric("商標件数", len(records))
            registered = sum(1 for r in records if r["is_registered"])
            st.metric("登録済", registered)
            n_classes = len(set(
                c for r in records for c in r["classes"]
            ))
            st.metric("区分数 (使用)", f"{n_classes} / 45")

    # プレビュー
    if st.session_state.tm_records:
        st.subheader("データプレビュー")
        records = st.session_state.tm_records
        preview_rows = []
        for r in records[:100]:
            preview_rows.append({
                "登録番号":  r["reg_number"],
                "商標":     r["trademark"],
                "区分":     ", ".join(str(c) for c in r["classes"]),
                "出願人":   r["applicant"],
                "出願日":   r["app_date"],
                "ステータス": r["status"],
            })
        st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, height=350)


# ══════════════════════════════════════════════════════════
# Tab 2: 統計・ランキング
# ══════════════════════════════════════════════════════════

with tab_stats:
    st.header("📊 統計・ランキング")

    if not st.session_state.tm_records:
        st.info("先にデータを取り込んでください")
        st.stop()

    records = st.session_state.tm_records

    # ── 上段: 主要指標 ──────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("商標総数", len(records))
    m2.metric("登録済", sum(1 for r in records if r["is_registered"]))
    m3.metric("出願中", sum(1 for r in records if not r["is_registered"]))
    years_list = [r["app_year"] for r in records if r["app_year"]]
    if years_list:
        m4.metric("出願年範囲", f"{min(years_list)}〜{max(years_list)}")

    st.divider()

    # ── 区分ランキング ────────────────────────────────────
    st.subheader("区分別登録件数")
    df_cls = class_ranking(records)
    df_cls_nonzero = df_cls[df_cls["件数"] > 0]

    fig_cls = px.bar(
        df_cls_nonzero,
        x="件数", y=df_cls_nonzero["区分"].astype(str) + "類",
        orientation="h",
        hover_data=["説明", "割合(%)"],
        color="件数",
        color_continuous_scale="Blues",
        title=f"区分別件数 (Top {len(df_cls_nonzero)})",
        height=max(350, len(df_cls_nonzero) * 22),
    )
    fig_cls.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_cls, use_container_width=True)

    with st.expander("区分ランキング テーブル"):
        st.dataframe(df_cls, use_container_width=True)

    st.divider()

    # ── 出願人ランキング ──────────────────────────────────
    st.subheader("出願人別登録件数 Top20")
    df_appl = applicant_ranking(records, top_n=20)

    fig_appl = px.bar(
        df_appl,
        x="件数", y="出願人",
        orientation="h",
        color="件数",
        color_continuous_scale="Greens",
        title="出願人別件数",
        height=max(300, len(df_appl) * 26),
    )
    fig_appl.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig_appl, use_container_width=True)

    st.divider()

    # ── 年別トレンド ──────────────────────────────────────
    st.subheader("年別出願件数トレンド")
    df_year = yearly_trend(records)
    if not df_year.empty:
        fig_year = px.bar(
            df_year, x="年", y="件数",
            title="年別出願件数",
            color_discrete_sequence=["#2563EB"],
        )
        st.plotly_chart(fig_year, use_container_width=True)

    # ── ステータス・種別 ──────────────────────────────────
    col_s1, col_s2, col_s3 = st.columns(3)

    with col_s1:
        df_status = status_distribution(records)
        if not df_status.empty:
            fig_s = px.pie(df_status, names="ステータス", values="件数",
                           title="ステータス分布", hole=0.4)
            st.plotly_chart(fig_s, use_container_width=True)

    with col_s2:
        df_reg = regtype_distribution(records)
        if not df_reg.empty:
            fig_r = px.pie(df_reg, names="種別", values="件数",
                           title="国内/国際/出願", hole=0.4)
            st.plotly_chart(fig_r, use_container_width=True)

    with col_s3:
        df_mc = multiclass_stats(records)
        if not df_mc.empty:
            fig_mc = px.bar(df_mc, x="区分数", y="件数",
                            title="1商標あたりの区分数分布",
                            color_discrete_sequence=["#7C3AED"])
            st.plotly_chart(fig_mc, use_container_width=True)


# ══════════════════════════════════════════════════════════
# Tab 3: 商標ランドスケープ
# ══════════════════════════════════════════════════════════

with tab_landscape:
    st.header("🗺️ 商標ランドスケープ")

    if not st.session_state.tm_records:
        st.info("先にデータを取り込んでください")
        st.stop()

    records = st.session_state.tm_records

    if not st.session_state.tm_landscape_done or st.button("🔄 再計算"):
        with st.spinner("PCA でランドスケープを計算中..."):
            coords, labels = compute_trademark_landscape(
                records, n_clusters=st.session_state.tm_n_clusters
            )
        st.session_state.tm_coords         = coords
        st.session_state.tm_labels         = labels
        st.session_state.tm_landscape_done = True

        # cluster_id を records に書き込む
        for i, rec in enumerate(records):
            rec["cluster_id"] = int(labels[i])
        st.session_state.tm_records = records

    if st.session_state.tm_coords is not None:
        coords = st.session_state.tm_coords
        labels = st.session_state.tm_labels

        # プロット用 DataFrame
        df_plot = pd.DataFrame({
            "x":         coords[:, 0],
            "y":         coords[:, 1],
            "cluster":   [str(l) for l in labels],
            "商標名":    [r["trademark"]   for r in records],
            "出願人":    [r["applicant"]   for r in records],
            "区分":      [", ".join(str(c) for c in r["classes"]) for r in records],
            "出願年":    [str(r.get("app_year", "")) for r in records],
            "ステータス": [r["status"]     for r in records],
            "主区分":    [str(r["primary_class"] or "") for r in records],
        })

        color_by = st.selectbox(
            "カラーリング",
            ["cluster", "主区分", "出願年", "ステータス"],
            key="tm_color_by",
        )

        fig = px.scatter(
            df_plot, x="x", y="y",
            color=color_by,
            hover_data=["商標名", "出願人", "区分", "出願年", "ステータス"],
            title="商標ランドスケープ (PCA 2D)",
            width=900, height=650,
        )
        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
            yaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # クラスター集計
        st.subheader("クラスター別集計")
        from collections import Counter
        cluster_rows = []
        for cid in sorted(set(labels)):
            members = [r for r in records if r.get("cluster_id") == cid]
            top_cls = Counter(
                c for r in members for c in r.get("classes", [])
            ).most_common(3)
            top_appl = Counter(r["applicant"] for r in members if r["applicant"]).most_common(2)
            cluster_rows.append({
                "クラスター": cid,
                "件数":      len(members),
                "主要区分":  ", ".join(f"{c}類" for c, _ in top_cls),
                "主要出願人": ", ".join(a for a, _ in top_appl),
            })
        st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True)


# ══════════════════════════════════════════════════════════
# Tab 4: ホワイトスペース分析
# ══════════════════════════════════════════════════════════

with tab_whitespace:
    st.header("🔍 ホワイトスペース分析")

    if not st.session_state.tm_records:
        st.info("先にデータを取り込んでください")
        st.stop()

    records = st.session_state.tm_records

    # ── 区分カバレッジ ────────────────────────────────────
    st.subheader("区分カバレッジ (ニース分類 1〜45)")

    coverage = class_coverage(records)
    counts   = coverage["counts"]
    vacant   = coverage["vacant"]

    # 全45区分を棒グラフ表示
    cls_data = []
    for cls in range(1, 46):
        cls_data.append({
            "区分":   f"{cls}類",
            "件数":   counts.get(cls, 0),
            "説明":   NICE_CLASS_DESC.get(cls, ""),
            "状態":   "登録あり" if counts.get(cls, 0) > 0 else "空白",
        })
    df_cov = pd.DataFrame(cls_data)

    fig_cov = px.bar(
        df_cov, x="区分", y="件数",
        color="状態",
        color_discrete_map={"登録あり": "#2563EB", "空白": "#E5E7EB"},
        hover_data=["説明", "件数"],
        title=f"区分別登録件数 — 空白区分: {len(vacant)} / 45",
        height=380,
    )
    fig_cov.update_layout(xaxis={"tickangle": -45})
    st.plotly_chart(fig_cov, use_container_width=True)

    if vacant:
        st.write(f"**空白区分 ({len(vacant)}件)**")
        df_vacant = pd.DataFrame(vacant)
        st.dataframe(df_vacant, use_container_width=True, hide_index=True)

    st.divider()

    # ── 区分ペア ホワイトスペース ─────────────────────────
    st.subheader("区分ペア ホワイトスペース")
    st.caption("個別には登録実績のある区分同士で、組み合わせが存在しないペア（新規参入余地）")

    with st.spinner("区分ペアを分析中..."):
        df_ws = whitespace_class_pairs(records, min_class_count=2)

    if not df_ws.empty:
        st.metric("ホワイトスペース ペア数", len(df_ws))
        st.dataframe(
            df_ws.head(30)[["区分A", "区分A説明", "区分B", "区分B説明", "機会スコア"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("ホワイトスペースペアが見つかりませんでした（データ件数が少ない可能性があります）")

    st.divider()

    # ── 出願人×区分 ヒートマップ ──────────────────────────
    st.subheader("出願人 × 区分 マトリックス")

    top_n_appl = st.slider("表示出願人数", min_value=5, max_value=30, value=15)
    with st.spinner("マトリックスを計算中..."):
        df_mat = applicant_class_matrix(records, top_n=top_n_appl)

    if not df_mat.empty:
        fig_heat = px.imshow(
            df_mat,
            aspect="auto",
            color_continuous_scale="Blues",
            title="出願人 × 区分 ヒートマップ (値=件数、0=空白)",
            labels={"x": "区分", "y": "出願人", "color": "件数"},
        )
        fig_heat.update_layout(height=max(400, top_n_appl * 28))
        st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # ── 区分ペア 共起マトリックス ────────────────────────
    st.subheader("区分ペア 共起マトリックス (全体)")
    st.caption("値が 0 のセル = その区分ペアを同時カバーする商標が存在しない")

    with st.spinner("共起マトリックスを計算中..."):
        df_cooc = class_cooccurrence_matrix(records)

    # 使われている区分のみ絞り込む
    used_cols = [c for c in df_cooc.columns if df_cooc[c].sum() > 0]
    df_cooc_used = df_cooc.loc[used_cols, used_cols]

    if not df_cooc_used.empty:
        fig_cooc = px.imshow(
            df_cooc_used,
            color_continuous_scale="Viridis",
            title="区分ペア共起ヒートマップ",
            labels={"x": "区分B", "y": "区分A", "color": "共起件数"},
            aspect="equal",
        )
        fig_cooc.update_layout(height=600)
        st.plotly_chart(fig_cooc, use_container_width=True)

    st.divider()

    # ── 年別×区分 トレンド ───────────────────────────────
    st.subheader("年別 × 区分 出願トレンド")

    with st.spinner("年別トレンドを計算中..."):
        df_trend = yearly_class_trend(records)

    if not df_trend.empty:
        # 主要区分のみ折れ線グラフ
        used_cls = [c for c in df_trend.columns if df_trend[c].sum() > 0]
        top_trend_cls = sorted(
            used_cls, key=lambda c: df_trend[c].sum(), reverse=True
        )[:8]

        df_melt = df_trend[top_trend_cls].reset_index().melt(
            id_vars="年", var_name="区分", value_name="件数"
        )
        fig_trend = px.line(
            df_melt, x="年", y="件数", color="区分",
            markers=True,
            title="主要区分の年別出願件数推移",
        )
        st.plotly_chart(fig_trend, use_container_width=True)


# ══════════════════════════════════════════════════════════
# Tab 5: AI 戦略レポート
# ══════════════════════════════════════════════════════════

with tab_report:
    st.header("📝 AI 戦略レポート")

    if not st.session_state.tm_records:
        st.info("先にデータを取り込んでください")
        st.stop()

    if not st.session_state.tm_api_validated:
        st.warning("サイドバーで Gemini APIキーを設定してください")
        st.stop()

    records = st.session_state.tm_records

    st.session_state.tm_objective = st.text_area(
        "分析目的",
        value=st.session_state.tm_objective,
        height=80,
    )

    if st.session_state.tm_report_done:
        st.success("レポート生成済み")

    if st.button(
        "📊 レポート生成" if not st.session_state.tm_report_done else "🔄 再生成",
        type="primary",
    ):
        client = st.session_state.tm_llm_client
        context = build_report_context(records)

        system = (
            "あなたはIPストラテジストです。"
            "提供された商標ランドスケープデータをもとに、"
            "経営層向けの戦略レポートをMarkdownで作成してください。\n\n"
            "## エグゼクティブサマリー\n"
            "## 商標登録トレンド分析\n"
            "## 主要プレイヤーのポジション\n"
            "## ホワイトスペース（未登録空白領域）\n"
            "## 推奨アクション\n\n"
            "根拠となるデータには具体的な数値を引用してください。"
        )
        user = (
            f"分析目的: {st.session_state.tm_objective}\n\n"
            f"{context}"
        )

        with st.spinner("Gemini がレポートを生成中... (約20〜30秒)"):
            try:
                report = client.generate_text(system, user)
                st.session_state.tm_report      = report
                st.session_state.tm_report_done = True
                st.rerun()
            except Exception as e:
                st.error(f"レポート生成エラー: {e}")

    if st.session_state.tm_report_done and st.session_state.tm_report:
        st.markdown(st.session_state.tm_report)

        st.download_button(
            "📥 レポートをダウンロード (Markdown)",
            data=st.session_state.tm_report.encode("utf-8"),
            file_name="trademark_landscape_report.md",
            mime="text/markdown",
        )
