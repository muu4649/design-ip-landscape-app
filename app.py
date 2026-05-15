"""
デザインランドスケープ — Gemini 無料API版

使用モデル: gemini-2.5-flash (無料ティア: 15 RPM / 1500 RPD)

タブ構成:
  1. 📥 データ取込   — PDF / CSV アップロード
  2. 🤖 Gemini 分析  — 意匠を Gemini でバッチ分類
  3. 🗺️ ランドスケープ — PCA 2D 散布図 (Plotly)
  4. 📊 クラスター   — クラスター別集計・サマリー
  5. 📝 戦略レポート  — VOYAGER 3フェーズ Markdown レポート
"""

from __future__ import annotations

import io
import json
import time
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────
# ページ設定
# ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="デザインランドスケープ (Gemini)",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────
# インポート
# ──────────────────────────────────────────────────────────

try:
    from gemini_client import LLMClient, validate_api_key
    _GEMINI_OK = True
except ImportError:
    _GEMINI_OK = False

try:
    from pdf_parser import parse_jplatpat_pdf, build_meta_from_csv
    _PDF_OK = True
except ImportError:
    _PDF_OK = False

# ──────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────

MAX_DESIGNS   = 300   # Gemini 無料ティア RPD を考慮
N_CLUSTERS    = 8
PCA_DIM       = 2

# デザインスタイル → 色マッピング
STYLE_COLORS = {
    "minimalist":  "#2563EB",
    "industrial":  "#DC2626",
    "retro":       "#D97706",
    "modern":      "#059669",
    "traditional": "#7C3AED",
    "unknown":     "#9CA3AF",
}

# ──────────────────────────────────────────────────────────
# Session State 初期化
# ──────────────────────────────────────────────────────────

_DEFAULTS: dict = {
    "api_key":          "",
    "api_validated":    False,
    "llm_client":       None,
    "meta_list":        [],
    "images":           [],
    "classified":       [],   # analyze_design_images の結果
    "merged":           [],   # meta + classified をマージした dict list
    "coords_2d":        None, # PCA coords np.ndarray (N, 2)
    "cluster_labels":   None,
    "report":           None, # generate_landscape_report の結果
    "analysis_done":    False,
    "landscape_done":   False,
    "report_done":      False,
    "objective":        "コーヒー関連器具のデザイン動向と競合ポジションを把握する",
    "n_clusters":       N_CLUSTERS,
    "img_weight":       0.5,
}

for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────
# サイドバー: API キー設定
# ──────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🗺️ デザインランドスケープ")
    st.caption("Powered by Gemini 2.5 Flash (Free Tier)")
    st.divider()

    st.subheader("🔑 Gemini API 設定")
    api_key_input = st.text_input(
        "APIキー",
        value=st.session_state.api_key,
        type="password",
        placeholder="AIzaSy...",
        help="Google AI Studio (aistudio.google.com) で取得。無料ティア使用。",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("検証", use_container_width=True):
            if api_key_input:
                with st.spinner("検証中..."):
                    ok, msg = validate_api_key(api_key_input)
                if ok:
                    st.session_state.api_key       = api_key_input
                    st.session_state.api_validated  = True
                    st.session_state.llm_client     = LLMClient(api_key_input)
                    st.success(msg)
                else:
                    st.error(msg)
            else:
                st.warning("APIキーを入力してください")

    with col_b:
        if st.button("クリア", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    if st.session_state.api_validated:
        st.success("APIキー有効")
    else:
        st.info("APIキーを入力して「検証」してください")

    st.divider()
    st.subheader("⚙️ 分析設定")
    st.session_state.objective = st.text_area(
        "分析目的",
        value=st.session_state.objective,
        height=80,
    )
    st.session_state.n_clusters = st.slider(
        "クラスター数 (KMeans)",
        min_value=2, max_value=20,
        value=st.session_state.n_clusters,
    )

    st.divider()
    st.caption("無料ティア制限: 15 RPM / 1,500 RPD")
    if st.session_state.meta_list:
        n = len(st.session_state.meta_list)
        st.caption(f"現在のデータ: {n} 件")
        # 1バッチ5件 → API コール数
        n_calls = (n + 4) // 5
        st.caption(f"分析時 API コール数 (目安): ~{n_calls + 5} 回")


# ──────────────────────────────────────────────────────────
# メインタブ
# ──────────────────────────────────────────────────────────

tab_upload, tab_analyze, tab_landscape, tab_cluster, tab_report = st.tabs([
    "📥 データ取込",
    "🤖 Gemini 分析",
    "🗺️ ランドスケープ",
    "📊 クラスター",
    "📝 戦略レポート",
])


# ══════════════════════════════════════════════════════════
# Tab 1: データ取込
# ══════════════════════════════════════════════════════════

with tab_upload:
    st.header("📥 データ取込")

    upload_col, info_col = st.columns([3, 2])

    with upload_col:
        file_type = st.radio(
            "ファイル形式",
            ["PDF (J-PlatPat 意匠検索結果)", "CSV (J-PlatPat エクスポート)"],
            horizontal=True,
        )

        uploaded = st.file_uploader(
            "ファイルをアップロード",
            type=["pdf"] if "PDF" in file_type else ["csv"],
            help="J-PlatPat からダウンロードした意匠検索結果ファイル",
        )

        if uploaded is not None:
            with st.spinner("解析中..."):
                try:
                    if "PDF" in file_type:
                        if not _PDF_OK:
                            st.error("PyMuPDF が必要です: pip install pymupdf")
                        else:
                            meta_list, images = parse_jplatpat_pdf(uploaded.read())
                    else:
                        df = pd.read_csv(uploaded, encoding="utf-8-sig")
                        meta_list = build_meta_from_csv(df)
                        images    = [None] * len(meta_list)

                    # 上限チェック
                    if len(meta_list) > MAX_DESIGNS:
                        st.warning(
                            f"データが {len(meta_list)} 件あります。"
                            f"無料ティア制限のため先頭 {MAX_DESIGNS} 件のみ使用します。"
                        )
                        meta_list = meta_list[:MAX_DESIGNS]
                        images    = images[:MAX_DESIGNS]

                    st.session_state.meta_list      = meta_list
                    st.session_state.images         = images
                    st.session_state.classified     = []
                    st.session_state.merged         = []
                    st.session_state.coords_2d      = None
                    st.session_state.cluster_labels = None
                    st.session_state.report         = None
                    st.session_state.analysis_done  = False
                    st.session_state.landscape_done = False
                    st.session_state.report_done    = False

                    st.success(f"{len(meta_list)} 件の意匠データを読み込みました")

                except Exception as e:
                    st.error(f"読み込みエラー: {e}")

    with info_col:
        if st.session_state.meta_list:
            meta_list = st.session_state.meta_list
            images    = st.session_state.images

            st.metric("意匠数", len(meta_list))

            # 物品名 Top5
            articles = Counter(
                m["article_name"] for m in meta_list if m.get("article_name")
            )
            st.write("**物品名 Top5**")
            for art, cnt in articles.most_common(5):
                st.write(f"- {art}: {cnt}件")

            # 画像有無
            n_img = sum(1 for img in images if img is not None)
            st.metric("サムネイル取得数", f"{n_img} / {len(meta_list)}")

    # データプレビュー
    if st.session_state.meta_list:
        st.subheader("データプレビュー")
        df_preview = pd.DataFrame(st.session_state.meta_list)[
            ["reg_number", "article_name", "applicant", "class_code", "app_date"]
        ]
        st.dataframe(df_preview, use_container_width=True, height=300)

        # サムネイルプレビュー (最初の12件)
        imgs_to_show = [
            (i, img) for i, img in enumerate(st.session_state.images[:24])
            if img is not None
        ]
        if imgs_to_show:
            st.subheader("サムネイルプレビュー (最初の24件)")
            cols = st.columns(6)
            for j, (i, img) in enumerate(imgs_to_show[:12]):
                with cols[j % 6]:
                    meta = st.session_state.meta_list[i]
                    st.image(img, width=100,
                             caption=meta.get("article_name", f"#{i+1}")[:10])


# ══════════════════════════════════════════════════════════
# Tab 2: Gemini 分析
# ══════════════════════════════════════════════════════════

with tab_analyze:
    st.header("🤖 Gemini 分析")

    if not st.session_state.meta_list:
        st.info("先にデータを取り込んでください (Tab: データ取込)")
        st.stop()

    if not st.session_state.api_validated:
        st.warning("サイドバーで Gemini APIキーを設定・検証してください")
        st.stop()

    n_designs = len(st.session_state.meta_list)
    n_batches = (n_designs + 4) // 5
    est_sec   = n_batches * 5

    st.info(
        f"**{n_designs} 件**の意匠を Gemini で分析します。  \n"
        f"バッチ数: {n_batches} / 推定所要時間: {est_sec//60}分{est_sec%60}秒  \n"
        f"(無料ティア 15 RPM 制限により 4 秒/バッチ間隔)"
    )

    if st.session_state.analysis_done:
        st.success(f"分析完了 — {len(st.session_state.classified)} 件")

    run_btn = st.button(
        "🚀 分析開始" if not st.session_state.analysis_done else "🔄 再分析",
        type="primary",
        disabled=not st.session_state.api_validated,
    )

    if run_btn:
        client = st.session_state.llm_client
        items  = [
            {"image": st.session_state.images[i], "meta": st.session_state.meta_list[i]}
            for i in range(n_designs)
        ]

        progress_bar  = st.progress(0.0)
        status_text   = st.empty()
        results_place = st.empty()

        def _progress(done, total):
            progress_bar.progress(done / total)
            status_text.text(f"分析中... {done} / {total}")

        try:
            classified = client.analyze_design_images(
                items,
                batch_size=5,
                progress_callback=_progress,
            )
        except Exception as e:
            st.error(f"分析エラー: {e}")
            classified = []

        if classified:
            # meta + classified をマージ
            merged = []
            for i, (meta, cls) in enumerate(
                zip(st.session_state.meta_list, classified)
            ):
                merged.append({**meta, **cls, "idx": i})

            st.session_state.classified     = classified
            st.session_state.merged         = merged
            st.session_state.analysis_done  = True
            st.session_state.landscape_done = False
            st.session_state.report_done    = False

            progress_bar.progress(1.0)
            status_text.text("分析完了")
            st.success(f"✅ {len(classified)} 件の分析が完了しました")
            st.rerun()

    # 分析結果プレビュー
    if st.session_state.analysis_done and st.session_state.merged:
        st.subheader("分析結果サンプル (最初の20件)")
        df_cls = pd.DataFrame(st.session_state.merged)[
            ["reg_number", "article_name", "design_style",
             "shape_category", "material_feel",
             "target_segment", "innovation_score", "design_summary"]
        ].head(20)
        st.dataframe(df_cls, use_container_width=True)

        # 分布グラフ
        st.subheader("分析結果分布")
        dist_col1, dist_col2, dist_col3 = st.columns(3)

        all_merged = st.session_state.merged

        with dist_col1:
            style_cnt = Counter(m.get("design_style", "unknown") for m in all_merged)
            df_style  = pd.DataFrame(style_cnt.items(), columns=["スタイル", "件数"])
            st.bar_chart(df_style.set_index("スタイル"))

        with dist_col2:
            seg_cnt = Counter(m.get("target_segment", "unknown") for m in all_merged)
            df_seg  = pd.DataFrame(seg_cnt.items(), columns=["ターゲット", "件数"])
            st.bar_chart(df_seg.set_index("ターゲット"))

        with dist_col3:
            scores = [m.get("innovation_score", 0) for m in all_merged if m.get("innovation_score", 0) > 0]
            if scores:
                score_cnt = Counter(scores)
                df_score  = pd.DataFrame(
                    [(str(k), v) for k, v in sorted(score_cnt.items())],
                    columns=["革新性スコア", "件数"]
                )
                st.bar_chart(df_score.set_index("革新性スコア"))


# ══════════════════════════════════════════════════════════
# Tab 3: ランドスケープ (PCA 2D)
# ══════════════════════════════════════════════════════════

with tab_landscape:
    st.header("🗺️ ランドスケープ")

    if not st.session_state.analysis_done:
        st.info("先に Gemini 分析を実行してください (Tab: Gemini 分析)")
        st.stop()

    # PCA 特徴量を構築・計算
    if not st.session_state.landscape_done or st.button("🔄 再計算", key="recalc_landscape"):
        merged = st.session_state.merged

        with st.spinner("PCA で2D座標を計算中..."):
            # Gemini分析結果から数値特徴量を構築
            coords, labels = _build_landscape(merged, st.session_state.n_clusters)

        st.session_state.coords_2d      = coords
        st.session_state.cluster_labels = labels
        st.session_state.landscape_done = True

        # merged に cluster_id を追加
        for i, (item, label) in enumerate(zip(merged, labels)):
            item["cluster_id"] = int(label)
        st.session_state.merged = merged

    if st.session_state.coords_2d is not None:
        coords = st.session_state.coords_2d
        labels = st.session_state.cluster_labels
        merged = st.session_state.merged

        # 散布図データ構築
        df_plot = pd.DataFrame({
            "x":               coords[:, 0],
            "y":               coords[:, 1],
            "cluster":         [str(l) for l in labels],
            "article_name":    [m.get("article_name", "")  for m in merged],
            "applicant":       [m.get("applicant", "")     for m in merged],
            "design_style":    [m.get("design_style", "")  for m in merged],
            "target_segment":  [m.get("target_segment", "") for m in merged],
            "innovation_score":[m.get("innovation_score", 0) for m in merged],
            "design_summary":  [m.get("design_summary", "") for m in merged],
            "reg_number":      [m.get("reg_number", "")    for m in merged],
        })

        color_by = st.selectbox(
            "カラーリング",
            ["cluster", "design_style", "target_segment", "innovation_score"],
            index=0,
        )

        if color_by == "innovation_score":
            fig = px.scatter(
                df_plot, x="x", y="y",
                color="innovation_score",
                color_continuous_scale="Viridis",
                hover_data=["article_name", "applicant", "design_style",
                            "target_segment", "design_summary", "reg_number"],
                title="デザインランドスケープ (PCA 2D)",
                width=900, height=650,
            )
        else:
            fig = px.scatter(
                df_plot, x="x", y="y",
                color=color_by,
                hover_data=["article_name", "applicant", "design_style",
                            "target_segment", "design_summary", "reg_number"],
                title="デザインランドスケープ (PCA 2D)",
                width=900, height=650,
            )

        fig.update_traces(marker=dict(size=8, opacity=0.8))
        fig.update_layout(
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
            yaxis=dict(showgrid=True, gridcolor="#F0F0F0"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # クラスター集計テーブル
        st.subheader("クラスター別集計")
        cluster_summary = _compute_cluster_summary(merged, labels)
        df_summary = pd.DataFrame(cluster_summary)
        st.dataframe(df_summary, use_container_width=True)


# ══════════════════════════════════════════════════════════
# Tab 4: クラスター詳細
# ══════════════════════════════════════════════════════════

with tab_cluster:
    st.header("📊 クラスター詳細")

    if not st.session_state.landscape_done:
        st.info("先にランドスケープを計算してください (Tab: ランドスケープ)")
        st.stop()

    merged = st.session_state.merged
    labels = st.session_state.cluster_labels

    cluster_ids = sorted(set(labels))
    selected_cid = st.selectbox(
        "クラスターを選択",
        cluster_ids,
        format_func=lambda x: f"クラスター {x}",
    )

    members = [m for m in merged if m.get("cluster_id") == selected_cid]

    st.metric("意匠数", len(members))

    # 集計
    info_col1, info_col2, info_col3 = st.columns(3)

    with info_col1:
        articles = Counter(m.get("article_name", "") for m in members if m.get("article_name"))
        st.write("**物品名 Top5**")
        for art, cnt in articles.most_common(5):
            st.write(f"- {art}: {cnt}件")

    with info_col2:
        applicants = Counter(m.get("applicant", "") for m in members if m.get("applicant"))
        st.write("**出願人 Top5**")
        for app, cnt in applicants.most_common(5):
            st.write(f"- {app}: {cnt}件")

    with info_col3:
        styles = Counter(m.get("design_style", "") for m in members if m.get("design_style"))
        st.write("**デザインスタイル**")
        for sty, cnt in styles.most_common():
            st.write(f"- {sty}: {cnt}件")

    # サムネイルギャラリー
    st.subheader("サムネイル")
    member_indices = [m["idx"] for m in members if "idx" in m]
    gallery_imgs = [
        (i, st.session_state.images[i], st.session_state.merged[i])
        for i in member_indices[:30]
        if i < len(st.session_state.images) and st.session_state.images[i] is not None
    ]

    if gallery_imgs:
        cols = st.columns(6)
        for j, (i, img, meta) in enumerate(gallery_imgs):
            with cols[j % 6]:
                caption = (
                    f"{meta.get('article_name', '')[:8]}\n"
                    f"★{meta.get('innovation_score', 0)}"
                )
                st.image(img, width=100, caption=caption)
    else:
        # サムネイルなし → テキストリスト
        df_members = pd.DataFrame(members)[
            ["reg_number", "article_name", "applicant",
             "design_style", "innovation_score", "design_summary"]
        ]
        st.dataframe(df_members, use_container_width=True)

    # Gemini サマリー (クラスター)
    st.subheader("このクラスターのAIサマリー")
    if st.button("📝 このクラスターをAI要約", key=f"summarize_cluster_{selected_cid}"):
        if st.session_state.llm_client:
            with st.spinner("Gemini が要約中..."):
                summaries = [m.get("design_summary", "") for m in members if m.get("design_summary")]
                prompt = (
                    f"クラスター{selected_cid}の意匠 {len(members)} 件をまとめてください。\n\n"
                    f"物品: {', '.join(a for a, _ in articles.most_common(3))}\n"
                    f"スタイル: {', '.join(s for s, _ in styles.most_common(3))}\n"
                    f"代表的サマリー:\n" + "\n".join(f"- {s}" for s in summaries[:5])
                )
                try:
                    result = st.session_state.llm_client.generate_text(
                        "デザインIPアナリストとして、このクラスターの特徴を200字程度で要約してください。",
                        prompt
                    )
                    st.write(result)
                except Exception as e:
                    st.error(f"エラー: {e}")
        else:
            st.warning("APIキーを設定してください")


# ══════════════════════════════════════════════════════════
# Tab 5: 戦略レポート (VOYAGER 3フェーズ)
# ══════════════════════════════════════════════════════════

with tab_report:
    st.header("📝 戦略レポート")

    if not st.session_state.landscape_done:
        st.info("先にランドスケープを計算してください (Tab: ランドスケープ)")
        st.stop()

    if not st.session_state.api_validated:
        st.warning("Gemini APIキーが必要です")
        st.stop()

    n_clusters_actual = len(set(st.session_state.cluster_labels))
    n_phase1_calls    = n_clusters_actual
    est_report_sec    = (n_phase1_calls + 2) * 5  # Phase1 + Phase2 + Phase3

    st.info(
        f"**VOYAGER 3フェーズ分析レポート**を生成します。  \n"
        f"Phase 1 (クラスター別): {n_phase1_calls} コール  \n"
        f"Phase 2 (統合分析): 1 コール  \n"
        f"Phase 3 (戦略レポート): 1 コール  \n"
        f"推定所要時間: 約 {est_report_sec//60}分{est_report_sec%60}秒"
    )

    if st.session_state.report_done:
        st.success("レポート生成済み")

    if st.button(
        "📊 レポート生成" if not st.session_state.report_done else "🔄 再生成",
        type="primary",
    ):
        client = st.session_state.llm_client
        merged = st.session_state.merged

        # 進捗表示
        status  = st.empty()
        prog    = st.progress(0.0)
        phase_n = [0]
        total_calls = n_phase1_calls + 2

        def _report_progress(msg, done, total):
            status.text(msg)
            prog.progress(min(phase_n[0] / total_calls, 1.0))
            phase_n[0] += 1

        try:
            report = client.generate_landscape_report(
                merged,
                objective=st.session_state.objective,
                progress_callback=_report_progress,
            )
            st.session_state.report      = report
            st.session_state.report_done = True
            prog.progress(1.0)
            status.text("レポート生成完了")
            st.success("✅ 戦略レポートが完成しました")
            st.rerun()

        except Exception as e:
            st.error(f"レポート生成エラー: {e}")

    # レポート表示
    if st.session_state.report_done and st.session_state.report:
        report = st.session_state.report

        # Phase 3 レポート (主要表示)
        st.subheader("戦略レポート")
        st.markdown(report["phase3_report"])

        # Evidence リスト
        with st.expander("📎 エビデンスリスト"):
            for i, ev in enumerate(report["evidence_list"]):
                st.write(f"**[Evidence {i+1}]** {ev}")

        with st.expander("🔍 Phase 1: クラスター別分析"):
            for i, analysis in enumerate(report["phase1_analyses"]):
                st.write(f"**クラスター {i}**")
                st.write(analysis)
                st.divider()

        with st.expander("🔍 Phase 2: 統合分析"):
            st.write(report["phase2_synthesis"])

        # Markdown ダウンロード
        full_report_md = _build_report_markdown(
            report,
            st.session_state.objective,
            len(st.session_state.merged),
        )
        st.download_button(
            "📥 レポートをダウンロード (Markdown)",
            data=full_report_md.encode("utf-8"),
            file_name="design_landscape_report.md",
            mime="text/markdown",
        )


# ══════════════════════════════════════════════════════════
# ヘルパー関数
# ══════════════════════════════════════════════════════════

def _build_landscape(
    merged: list[dict],
    n_clusters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Gemini 分析結果からワンホット + 数値特徴量を構築し、
    PCA 2D に変換してクラスタリングする。

    Returns
    -------
    coords_2d : np.ndarray  (N, 2)
    labels    : np.ndarray  (N,)
    """
    n = len(merged)

    # カテゴリ変数のユニーク値収集
    all_shapes   = sorted(set(m.get("shape_category", "unknown") for m in merged))
    all_materials= sorted(set(m.get("material_feel", "unknown")  for m in merged))
    all_styles   = sorted(set(m.get("design_style", "unknown")   for m in merged))
    all_segments = sorted(set(m.get("target_segment", "unknown") for m in merged))

    def _onehot(val, categories):
        vec = np.zeros(len(categories), dtype=np.float32)
        if val in categories:
            vec[categories.index(val)] = 1.0
        return vec

    feature_rows = []
    for m in merged:
        row = np.concatenate([
            _onehot(m.get("shape_category", "unknown"),   all_shapes),
            _onehot(m.get("material_feel", "unknown"),    all_materials),
            _onehot(m.get("design_style", "unknown"),     all_styles),
            _onehot(m.get("target_segment", "unknown"),   all_segments),
            [float(m.get("innovation_score", 0)) / 5.0],  # 正規化
        ])
        feature_rows.append(row)

    X = np.array(feature_rows, dtype=np.float32)

    # PCA 2D
    n_comp = min(PCA_DIM, X.shape[0] - 1, X.shape[1])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_comp, random_state=42)
    coords = pca.fit_transform(X_scaled)

    # ゼロパディングで 2D 保証
    if coords.shape[1] < 2:
        pad = np.zeros((n, 2 - coords.shape[1]), dtype=np.float32)
        coords = np.hstack([coords, pad])

    # KMeans クラスタリング
    k = min(n_clusters, n)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)

    return coords, labels.astype(int)


def _compute_cluster_summary(
    merged: list[dict],
    labels: np.ndarray,
) -> list[dict]:
    summaries = []
    for cid in sorted(set(labels)):
        members = [m for m in merged if m.get("cluster_id") == int(cid)]
        if not members:
            continue
        articles   = Counter(m.get("article_name", "") for m in members if m.get("article_name"))
        applicants = Counter(m.get("applicant", "")    for m in members if m.get("applicant"))
        styles     = Counter(m.get("design_style", "") for m in members if m.get("design_style"))
        segments   = Counter(m.get("target_segment", "") for m in members if m.get("target_segment"))
        scores     = [m.get("innovation_score", 0) for m in members if m.get("innovation_score", 0) > 0]

        summaries.append({
            "クラスター":        f"クラスター {cid}",
            "件数":              len(members),
            "主要物品":          ", ".join(a for a, _ in articles.most_common(2)),
            "主要出願人":        ", ".join(a for a, _ in applicants.most_common(2)),
            "デザインスタイル":  ", ".join(s for s, _ in styles.most_common(2)),
            "ターゲット":        ", ".join(s for s, _ in segments.most_common(2)),
            "平均革新性":        round(float(np.mean(scores)), 2) if scores else 0.0,
        })
    return summaries


def _build_report_markdown(
    report: dict,
    objective: str,
    n_designs: int,
) -> str:
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")

    lines = [
        f"# デザインランドスケープ 戦略レポート",
        f"",
        f"**分析目的**: {objective}  ",
        f"**意匠数**: {n_designs} 件  ",
        f"**生成日**: {today}  ",
        f"**生成モデル**: Gemini 2.5 Flash (Free Tier)  ",
        f"",
        f"---",
        f"",
        report["phase3_report"],
        f"",
        f"---",
        f"",
        f"## Phase 2: クロスクラスター統合分析",
        f"",
        report["phase2_synthesis"],
        f"",
        f"---",
        f"",
        f"## Phase 1: クラスター別分析",
        f"",
    ]

    for i, analysis in enumerate(report["phase1_analyses"]):
        lines.append(f"### クラスター {i}")
        lines.append(analysis)
        lines.append("")

    lines += [
        "---",
        "",
        "## エビデンスリスト",
        "",
    ]
    for i, ev in enumerate(report["evidence_list"]):
        lines.append(f"**[Evidence {i+1}]** {ev}  ")

    return "\n".join(lines)
