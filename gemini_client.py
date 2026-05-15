"""
Gemini API クライアント — 無料ティア対応

無料枠制限 (2025年時点):
  gemini-2.5-flash : 15 RPM / 1,500 RPD / 1,000,000 TPM

  → バッチ送信間隔 4秒 (60s / 15RPM)
  → 429エラー時は 60秒待機後リトライ

分析スキーマ (analyze_design_images の戻り値 per 意匠):
  shape_category  : str   主形状カテゴリ (例: "cylindrical", "flat", "organic")
  material_feel   : str   素材感 (例: "metallic", "plastic", "ceramic", "wood")
  design_style    : str   デザインスタイル (例: "minimalist", "industrial", "retro")
  key_features    : list  特徴キーワード 最大5個
  innovation_score: int   革新性スコア 1〜5
  target_segment  : str   推定ターゲット層 (例: "home_user", "professional", "gift")
  design_summary  : str   日本語1文サマリー
"""

from __future__ import annotations

import base64
import io
import json
import re
import time
from typing import Any, Optional

from PIL import Image


# ──────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────

RPM_LIMIT      = 15
INTER_CALL_SEC = 4.0    # 60 / 15 = 4秒
RETRY_WAIT_SEC = 65.0   # 429 時の待機時間
MAX_RETRIES    = 3

# 1バッチあたりの意匠数 (画像 + テキストのトークン数を考慮)
DEFAULT_BATCH_SIZE = 5

DESIGN_CLASSIFY_SCHEMA = {
    "shape_category":   "主形状を英単語1語で (例: cylindrical / flat / organic / geometric / angular)",
    "material_feel":    "素材感を英単語1語で (例: metallic / plastic / ceramic / wood / glass / fabric)",
    "design_style":     "デザインスタイルを英単語1語で (例: minimalist / industrial / retro / modern / traditional)",
    "key_features":     "視覚的特徴キーワードをリストで最大5個 (日本語可)",
    "innovation_score": "革新性スコア 整数1〜5 (5=非常に革新的)",
    "target_segment":   "推定ターゲット層 英単語1語で (例: home_user / professional / cafe / gift / industrial)",
    "design_summary":   "意匠の特徴を日本語1文で要約",
}


# ──────────────────────────────────────────────────────────
# LLMClient (VOYAGER パターン準拠)
# ──────────────────────────────────────────────────────────

class LLMClient:
    """
    Gemini API ラッパー。
    VOYAGER参照実装の generate_text インターフェースに加え、
    マルチモーダル画像解析メソッドを提供する。
    """

    def __init__(
        self,
        api_key: str,
        model_name: str = "gemini-2.5-flash",
    ) -> None:
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self._genai   = genai
            self._model   = genai.GenerativeModel(model_name)
            self._model_name = model_name
        except ImportError as e:
            raise ImportError(
                "google-generativeai が必要です: pip install google-generativeai"
            ) from e

        self._last_call_time: float = 0.0

    # ── 基本テキスト生成 ──────────────────────────────────

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        max_retries: int = MAX_RETRIES,
    ) -> str:
        """
        テキスト生成。429エラー時は RETRY_WAIT_SEC 待機してリトライ。
        """
        self._rate_limit_wait()
        prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt

        for attempt in range(max_retries):
            try:
                response = self._model.generate_content(prompt)
                self._last_call_time = time.time()
                return response.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = RETRY_WAIT_SEC * (attempt + 1)
                    time.sleep(wait)
                    continue
                raise

        raise RuntimeError(f"Gemini API: {max_retries}回リトライ後も失敗")

    # ── 意匠画像バッチ分類 ────────────────────────────────

    def analyze_design_images(
        self,
        items: list[dict],  # {"image": PIL.Image | None, "meta": dict}
        batch_size: int = DEFAULT_BATCH_SIZE,
        progress_callback=None,
    ) -> list[dict]:
        """
        意匠リストを Gemini でバッチ分類する。

        Parameters
        ----------
        items : list of {"image": PIL.Image|None, "meta": dict}
            meta には reg_number, article_name, class_code を含む
        batch_size : int
            1回の API コール当たりの意匠数
        progress_callback : callable(done, total) | None

        Returns
        -------
        list[dict]  各 item に対応する分類結果。
                    API 失敗時は _empty_result() を返す。
        """
        results: list[dict] = []
        total = len(items)

        for start in range(0, total, batch_size):
            batch = items[start : start + batch_size]
            batch_results = self._classify_batch(batch)
            results.extend(batch_results)

            if progress_callback:
                progress_callback(min(start + batch_size, total), total)

        return results

    def _classify_batch(self, batch: list[dict]) -> list[dict]:
        """1バッチを Gemini multimodal に送信して分類結果を返す。"""
        self._rate_limit_wait()

        content_parts = []
        content_parts.append(self._build_classify_prompt(len(batch)))

        for i, item in enumerate(batch):
            meta = item.get("meta", {})
            img  = item.get("image")

            # メタデータテキスト
            content_parts.append(
                f"\n--- 意匠 {i+1} ---\n"
                f"登録番号: {meta.get('reg_number', '不明')}\n"
                f"物品名: {meta.get('article_name', '不明')}\n"
                f"分類: {meta.get('class_code', '不明')}\n"
                f"出願人: {meta.get('applicant', '不明')}\n"
            )

            # 画像 (PIL → inline_data)
            if img is not None:
                try:
                    img_part = self._pil_to_part(img)
                    content_parts.append(img_part)
                except Exception:
                    content_parts.append("[画像なし]")
            else:
                content_parts.append("[画像なし]")

        content_parts.append(
            "\n\n上記の意匠を番号順にJSON配列で出力してください。"
            "```json\n[...]\n```"
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = self._model.generate_content(content_parts)
                self._last_call_time = time.time()
                return self._parse_batch_response(response.text, len(batch))
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait = RETRY_WAIT_SEC * (attempt + 1)
                    time.sleep(wait)
                    continue
                # その他エラーは空結果を返す
                return [self._empty_result(item.get("meta", {})) for item in batch]

        return [self._empty_result(item.get("meta", {})) for item in batch]

    # ── VOYAGER スタイル 3フェーズ レポート生成 ───────────

    def generate_landscape_report(
        self,
        classified_data: list[dict],
        objective: str = "デザインランドスケープの戦略的インサイト",
        progress_callback=None,
    ) -> dict:
        """
        VOYAGER パターンの 3フェーズ分析レポートを生成する。

        Phase 1: クラスター別アナリスト (クラスタ数分の API コール)
        Phase 2: クロスクラスター統合アナリスト
        Phase 3: 戦略レポート (Evidence 引用付き)

        Returns
        -------
        {
          "phase1_analyses": list[str],
          "phase2_synthesis": str,
          "phase3_report": str,
          "evidence_list": list[str],
        }
        """
        from collections import defaultdict

        # クラスタ別にグループ化
        clusters: dict[int, list[dict]] = defaultdict(list)
        for item in classified_data:
            cid = item.get("cluster_id", 0)
            clusters[cid].append(item)

        cluster_ids = sorted(clusters.keys())
        n_clusters  = len(cluster_ids)

        # ── Phase 1: クラスター別アナリスト ──────────────
        if progress_callback:
            progress_callback("Phase 1: クラスター分析中...", 0, n_clusters)

        phase1_analyses: list[str] = []
        for idx, cid in enumerate(cluster_ids):
            members = clusters[cid]
            analysis = self._phase1_cluster_analysis(cid, members, objective)
            phase1_analyses.append(analysis)

            if progress_callback:
                progress_callback(f"Phase 1: クラスター {cid} 完了", idx + 1, n_clusters)

        # ── Phase 2: クロスクラスター統合 ────────────────
        if progress_callback:
            progress_callback("Phase 2: クロスクラスター統合中...", 0, 1)

        phase2_synthesis = self._phase2_cross_cluster(
            phase1_analyses, cluster_ids, objective
        )
        if progress_callback:
            progress_callback("Phase 2: 完了", 1, 1)

        # ── Phase 3: 戦略レポート ─────────────────────────
        if progress_callback:
            progress_callback("Phase 3: 戦略レポート生成中...", 0, 1)

        evidence_list = self._build_evidence_list(phase1_analyses)
        phase3_report = self._phase3_strategist(
            phase2_synthesis, evidence_list, objective
        )
        if progress_callback:
            progress_callback("Phase 3: 完了", 1, 1)

        return {
            "phase1_analyses": phase1_analyses,
            "phase2_synthesis": phase2_synthesis,
            "phase3_report": phase3_report,
            "evidence_list": evidence_list,
        }

    # ── Phase 実装 ────────────────────────────────────────

    def _phase1_cluster_analysis(
        self,
        cluster_id: int,
        members: list[dict],
        objective: str,
    ) -> str:
        system = (
            "あなたはデザインIPのアナリストです。"
            "指定されたクラスターの意匠データを分析し、"
            "このクラスターの特徴・傾向・競合状況を簡潔にまとめてください。"
            "200〜300字程度。"
        )

        # クラスタ内の集計情報を構築
        articles   = [m.get("article_name", "") for m in members if m.get("article_name")]
        styles     = [m.get("design_style", "")   for m in members if m.get("design_style")]
        segments   = [m.get("target_segment", "") for m in members if m.get("target_segment")]
        applicants = [m.get("applicant", "")      for m in members if m.get("applicant")]
        summaries  = [m.get("design_summary", "") for m in members if m.get("design_summary")]

        from collections import Counter
        top_articles  = [a for a, _ in Counter(articles).most_common(3)]
        top_styles    = [s for s, _ in Counter(styles).most_common(3)]
        top_segments  = [s for s, _ in Counter(segments).most_common(2)]
        top_applicants = [a for a, _ in Counter(applicants).most_common(3)]

        user = (
            f"分析目的: {objective}\n\n"
            f"クラスター ID: {cluster_id}\n"
            f"意匠数: {len(members)}\n"
            f"主要物品: {', '.join(top_articles)}\n"
            f"デザインスタイル傾向: {', '.join(top_styles)}\n"
            f"ターゲット層傾向: {', '.join(top_segments)}\n"
            f"主要出願人: {', '.join(top_applicants[:3])}\n"
            f"代表的サマリー:\n"
            + "\n".join(f"  - {s}" for s in summaries[:5])
        )

        return self.generate_text(system, user)

    def _phase2_cross_cluster(
        self,
        phase1_analyses: list[str],
        cluster_ids: list[int],
        objective: str,
    ) -> str:
        system = (
            "あなたはデザインIPのシニアアナリストです。"
            "各クラスターの分析結果を統合し、"
            "市場全体のデザイントレンド・空白領域・競合構造を分析してください。"
            "400〜600字程度。"
        )

        analyses_text = "\n\n".join(
            f"【クラスター {cid} の分析】\n{analysis}"
            for cid, analysis in zip(cluster_ids, phase1_analyses)
        )

        user = (
            f"分析目的: {objective}\n\n"
            f"{analyses_text}\n\n"
            "上記を踏まえ、全クラスターを横断した統合分析を行ってください。"
            "特に: (1)デザイントレンドの方向性、(2)競合の棲み分け、(3)白地領域の所在"
        )

        return self.generate_text(system, user)

    def _build_evidence_list(self, phase1_analyses: list[str]) -> list[str]:
        """Phase1 分析からエビデンス文を抽出する。"""
        evidence = []
        for analysis in phase1_analyses:
            # 「。」で区切り、短い文をエビデンスとして収集
            sentences = re.split(r"[。！？\n]", analysis)
            for sent in sentences:
                sent = sent.strip()
                if 15 <= len(sent) <= 80:
                    evidence.append(sent)
        return evidence[:20]  # 最大20件

    def _phase3_strategist(
        self,
        phase2_synthesis: str,
        evidence_list: list[str],
        objective: str,
    ) -> str:
        system = (
            "あなたはIPストラテジストです。"
            "デザインランドスケープ分析を基に、経営層向けの戦略レポートを作成してください。\n\n"
            "フォーマット:\n"
            "## エグゼクティブサマリー\n"
            "## 市場デザイントレンド\n"
            "## 競合ポジション分析\n"
            "## 白地領域・機会\n"
            "## 推奨アクション\n\n"
            "根拠となる分析には [[Evidence X]] 形式で引用番号を付記してください。"
        )

        evidence_text = "\n".join(
            f"[Evidence {i+1}] {e}" for i, e in enumerate(evidence_list)
        )

        user = (
            f"分析目的: {objective}\n\n"
            f"統合分析:\n{phase2_synthesis}\n\n"
            f"エビデンスリスト:\n{evidence_text}"
        )

        return self.generate_text(system, user)

    # ── ユーティリティ ─────────────────────────────────────

    def _rate_limit_wait(self) -> None:
        """RPM 制限に対応するため、最低 INTER_CALL_SEC 秒待機する。"""
        elapsed = time.time() - self._last_call_time
        if elapsed < INTER_CALL_SEC:
            time.sleep(INTER_CALL_SEC - elapsed)

    @staticmethod
    def _pil_to_part(img: Image.Image) -> dict:
        """PIL画像を Gemini の inline_data パーツに変換する。"""
        buf = io.BytesIO()
        # サイズを抑えてトークン節約
        img_resized = img.copy()
        img_resized.thumbnail((256, 256), Image.LANCZOS)
        img_resized.save(buf, format="JPEG", quality=80)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": b64,
            }
        }

    @staticmethod
    def _build_classify_prompt(n: int) -> str:
        schema_lines = "\n".join(
            f'  "{k}": "{v}"' for k, v in DESIGN_CLASSIFY_SCHEMA.items()
        )
        return (
            f"以下の {n} 件の意匠を分析し、各意匠について次のスキーマのJSONオブジェクトを生成してください。\n\n"
            f"スキーマ:\n{{\n{schema_lines}\n}}\n\n"
            "必ずJSON配列 [...] で出力し、インデックス順は入力の意匠番号順に従ってください。"
        )

    @staticmethod
    def _parse_batch_response(text: str, expected_n: int) -> list[dict]:
        """Gemini レスポンスから JSON 配列を抽出してパースする。"""
        # ```json ... ``` ブロックを抽出
        match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
        if not match:
            # コードブロックなしで直接 JSON がある場合
            match = re.search(r"\[.*?\]", text, re.DOTALL)
        if not match:
            return [_empty_result_static() for _ in range(expected_n)]

        try:
            parsed = json.loads(match.group(1) if match.lastindex else match.group(0))
            # 件数が足りなければ空で補完
            while len(parsed) < expected_n:
                parsed.append(_empty_result_static())
            return parsed[:expected_n]
        except json.JSONDecodeError:
            return [_empty_result_static() for _ in range(expected_n)]

    @staticmethod
    def _empty_result(meta: dict) -> dict:
        return {
            **_empty_result_static(),
            "article_name": meta.get("article_name", ""),
            "reg_number":   meta.get("reg_number", ""),
        }


def _empty_result_static() -> dict:
    return {
        "shape_category":   "unknown",
        "material_feel":    "unknown",
        "design_style":     "unknown",
        "key_features":     [],
        "innovation_score": 0,
        "target_segment":   "unknown",
        "design_summary":   "分析失敗",
    }


# ──────────────────────────────────────────────────────────
# API キー検証ユーティリティ
# ──────────────────────────────────────────────────────────

def validate_api_key(api_key: str) -> tuple[bool, str]:
    """
    Gemini API キーの有効性を確認する。

    Returns
    -------
    (is_valid, message)
    """
    if not api_key or len(api_key) < 10:
        return False, "APIキーが短すぎます"
    try:
        client = LLMClient(api_key)
        client.generate_text("", "テスト: 「OK」とだけ返してください")
        return True, "APIキーが有効です"
    except Exception as e:
        return False, f"APIエラー: {e}"
