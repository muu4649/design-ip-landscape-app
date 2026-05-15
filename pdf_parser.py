"""
J-PlatPat 意匠検索結果 PDF パーサー

design_landscape_ml/pdf_parser.py を Gemini 版用に適合。
"""

from __future__ import annotations

import io
import re
import unicodedata
from typing import Optional

import fitz  # PyMuPDF
from PIL import Image

# ──────────────────────────────────────────────────────────
# 定数
# ──────────────────────────────────────────────────────────
RE_REG    = re.compile(r"意匠登録(\d{7,8})")
RE_CLASS  = re.compile(r"([A-Z]\d?-\d{3}(?:\.\d+)?|C[A-Z]-\d{2,4}(?:\.\d+)?)")
RE_TYPE   = re.compile(r"(関連意匠|基礎意匠|部分意匠)")
RE_YEAR   = re.compile(r"(20\d{2}|19\d{2})")
MIN_THUMB_PX = 60


# ──────────────────────────────────────────────────────────
# パブリック API
# ──────────────────────────────────────────────────────────

def parse_jplatpat_pdf(
    pdf_bytes: bytes,
) -> tuple[list[dict], list[Optional[Image.Image]]]:
    """
    J-PlatPat 意匠検索結果PDFを解析する。

    Returns
    -------
    metadata : list[dict]
        {reg_number, article_name, applicant, class_code, design_type, app_date, page_idx}
    images   : list[PIL.Image | None]
        各意匠のサムネイル
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_meta:   list[dict]              = []
    all_images: list[Optional[Image.Image]] = []

    for page_idx, page in enumerate(doc):
        m_list, img_list = _parse_page(doc, page, page_idx)
        all_meta.extend(m_list)
        all_images.extend(img_list)

    doc.close()
    return all_meta, all_images


def build_meta_from_csv(df) -> list[dict]:
    """
    J-PlatPat CSVから parse_jplatpat_pdf と同形式のメタデータを生成する。
    """
    def _find(candidates: list[str]) -> Optional[str]:
        for c in candidates:
            for col in df.columns:
                if c in col:
                    return col
        return None

    col_reg  = _find(["意匠登録番号", "登録番号"])
    col_item = _find(["意匠に係る物品", "物品名", "物品"])
    col_appl = _find(["出願人", "権利者"])
    col_date = _find(["出願日", "登録日"])
    col_cls  = _find(["ロカルノ", "国際意匠分類", "日本意匠分類", "分類"])

    return [
        {
            "reg_number":   str(row[col_reg]).strip()  if col_reg  else "",
            "article_name": _normalize(str(row[col_item]).strip()) if col_item else "",
            "applicant":    str(row[col_appl]).strip() if col_appl else "",
            "class_code":   str(row[col_cls]).strip()  if col_cls  else "",
            "design_type":  "",
            "app_date":     str(row[col_date]).strip() if col_date else "",
            "page_idx":     0,
        }
        for _, row in df.iterrows()
    ]


# ──────────────────────────────────────────────────────────
# 内部実装
# ──────────────────────────────────────────────────────────

def _parse_page(
    doc: fitz.Document,
    page: fitz.Page,
    page_idx: int,
) -> tuple[list[dict], list[Optional[Image.Image]]]:
    raw_text  = page.get_text("text")
    meta_list = _parse_text(raw_text, page_idx)

    if not meta_list:
        return [], []

    thumb_images = _extract_thumbnails(doc, page)

    img_list: list[Optional[Image.Image]] = []
    for i in range(len(meta_list)):
        img_list.append(thumb_images[i] if i < len(thumb_images) else None)

    return meta_list, img_list


def _parse_text(raw_text: str, page_idx: int) -> list[dict]:
    lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
    entries: list[dict] = []
    current: Optional[dict] = None

    for line in lines:
        reg_m = RE_REG.match(line)
        if reg_m:
            if current:
                entries.append(current)
            current = {
                "reg_number":   reg_m.group(1),
                "article_name": "",
                "applicant":    "",
                "class_code":   "",
                "design_type":  "",
                "app_date":     "",
                "page_idx":     page_idx,
            }
            continue

        if current is None:
            continue

        if RE_CLASS.match(line):
            current["class_code"] = line
            continue

        type_m = RE_TYPE.match(line)
        if type_m:
            current["design_type"] = type_m.group(1)
            continue

        year_m = RE_YEAR.fullmatch(line)
        if year_m:
            current["app_date"] = line
            continue

        if line in ("-", "ハーグ", "") or re.fullmatch(r"\d{1,3}\s*", line):
            continue

        if not current["article_name"] and len(line) >= 2:
            current["article_name"] = _normalize(line)
        elif not current["applicant"] and len(line) >= 2:
            current["applicant"] = line[:30]

    if current:
        entries.append(current)

    return entries


def _extract_thumbnails(
    doc: fitz.Document,
    page: fitz.Page,
) -> list[Image.Image]:
    thumbnails: list[Image.Image] = []
    for img_item in page.get_images(full=True):
        xref = img_item[0]
        try:
            base = doc.extract_image(xref)
            pil  = Image.open(io.BytesIO(base["image"])).convert("RGB")
            if pil.width >= MIN_THUMB_PX and pil.height >= MIN_THUMB_PX:
                thumbnails.append(pil)
        except Exception:
            continue
    return thumbnails


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).strip()
