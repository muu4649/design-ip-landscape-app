"""
J-PlatPat 商標検索結果 CSV パーサー

カラム構成:
  出願番号/登録番号/国際登録番号, 商標(検索用), 称呼基準, 称呼(参考情報),
  区分, 出願人/権利者/名義人, 出願日/国際登録日(事後指定日), 登録日, ステータス, 文献URL
"""

from __future__ import annotations

import io
import re
from typing import Optional

import pandas as pd

# ──────────────────────────────────────────────────────────
# ニース分類 説明マッピング
# ──────────────────────────────────────────────────────────

NICE_CLASS_DESC: dict[int, str] = {
    1:  "化学品・工業用材料",
    2:  "塗料・染料",
    3:  "化粧品・洗浄剤",
    4:  "工業用油脂・燃料",
    5:  "医薬品・衛生材料",
    6:  "金属材料・建材",
    7:  "機械・エンジン",
    8:  "手動工具・器具",
    9:  "電気・電子機器",
    10: "医療用機械器具",
    11: "照明・加熱・冷却設備",
    12: "乗り物",
    13: "火器・火薬",
    14: "貴金属・宝飾品",
    15: "楽器",
    16: "紙・文房具・印刷物",
    17: "ゴム・プラスチック製品",
    18: "革製品・バッグ",
    19: "非金属建材",
    20: "家具・木工品",
    21: "台所用品・ガラス製品",
    22: "ロープ・テント・繊維資材",
    23: "糸",
    24: "布地・テキスタイル",
    25: "衣類・履物",
    26: "リボン・レース・ボタン",
    27: "床・壁敷物",
    28: "玩具・スポーツ用品",
    29: "食肉・魚・乳製品",
    30: "コーヒー・茶・菓子・パン",
    31: "農産物・生きた動植物",
    32: "ビール・清涼飲料",
    33: "アルコール飲料",
    34: "タバコ",
    35: "広告・事業管理",
    36: "金融・保険・不動産",
    37: "建設・修理・清掃",
    38: "通信",
    39: "輸送・旅行",
    40: "材料処理",
    41: "教育・娯楽・スポーツ",
    42: "科学技術・IT",
    43: "飲食提供・宿泊",
    44: "医療・美容",
    45: "法律・警備・個人",
}


# ──────────────────────────────────────────────────────────
# パブリック API
# ──────────────────────────────────────────────────────────

def parse_trademark_csv(
    file_content: bytes,
    encoding: str = "utf-8-sig",
) -> list[dict]:
    """
    J-PlatPat 商標検索結果CSVをパースし、正規化済みレコードリストを返す。

    各レコード:
      reg_number, trademark, readings, classes (list[int]),
      applicant, app_date, reg_date, status, url,
      app_year (int|None), reg_type ("domestic"|"international"|"application"),
      primary_class (int|None)
    """
    try:
        df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(file_content), encoding="cp932")

    # カラム名マッピング
    col_map = _detect_columns(df)
    records: list[dict] = []

    for _, row in df.iterrows():
        rec = _parse_row(row, col_map)
        if rec:
            records.append(rec)

    return records


def explode_by_class(records: list[dict]) -> list[dict]:
    """
    1レコードが複数区分を持つ場合に区分ごとに展開する。
    区分別集計に使用する。
    """
    exploded = []
    for rec in records:
        for cls in (rec["classes"] or [0]):
            exploded.append({**rec, "single_class": cls})
    return exploded


# ──────────────────────────────────────────────────────────
# 内部実装
# ──────────────────────────────────────────────────────────

def _detect_columns(df: pd.DataFrame) -> dict:
    """カラム名を自動検出してマッピングを返す。"""
    cols = list(df.columns)

    def _find(*candidates) -> Optional[str]:
        for c in candidates:
            for col in cols:
                if c in col:
                    return col
        return None

    return {
        "reg_number": _find("番号"),
        "trademark":  _find("商標(検索用)", "商標"),
        "readings":   _find("称呼(参考情報)", "称呼"),
        "classes":    _find("区分"),
        "applicant":  _find("出願人", "権利者", "名義人"),
        "app_date":   _find("出願日", "国際登録日"),
        "reg_date":   _find("登録日"),
        "status":     _find("ステータス"),
        "url":        _find("文献URL", "URL"),
    }


def _parse_row(row, col_map: dict) -> Optional[dict]:
    def _get(key: str, default: str = "") -> str:
        col = col_map.get(key)
        if col is None:
            return default
        val = str(row.get(col, "")).strip()
        return "" if val in ("nan", "NaN", "None") else val

    reg_number = _get("reg_number")
    if not reg_number:
        return None

    trademark  = _get("trademark")
    readings   = _get("readings")
    classes_raw = _get("classes")
    applicant  = _get("applicant")
    app_date   = _get("app_date")
    reg_date   = _get("reg_date")
    status     = _get("status")
    url        = _get("url")

    # 区分を整数リストに変換
    classes = _parse_classes(classes_raw)

    # 出願年
    app_year = _extract_year(app_date)

    # 登録種別判定
    reg_type = _detect_reg_type(reg_number)

    return {
        "reg_number":    reg_number,
        "trademark":     trademark,
        "readings":      [r.strip() for r in readings.split(",") if r.strip()],
        "classes":       classes,
        "primary_class": classes[0] if classes else None,
        "applicant":     applicant,
        "app_date":      app_date,
        "reg_date":      reg_date,
        "app_year":      app_year,
        "status":        status,
        "url":           url,
        "reg_type":      reg_type,
        "is_registered": "登録" in status,
    }


def _parse_classes(raw: str) -> list[int]:
    """区分文字列 "17,21" → [17, 21]"""
    if not raw:
        return []
    result = []
    for part in re.split(r"[,，\s]+", raw):
        part = part.strip()
        if re.fullmatch(r"\d{1,2}", part):
            n = int(part)
            if 1 <= n <= 45:
                result.append(n)
    return sorted(set(result))


def _extract_year(date_str: str) -> Optional[int]:
    """日付文字列から年を抽出。"""
    m = re.search(r"(20\d{2}|19\d{2})", date_str)
    return int(m.group(1)) if m else None


def _detect_reg_type(reg_number: str) -> str:
    if reg_number.startswith("国際登録"):
        return "international"
    if reg_number.startswith("商願"):
        return "application"
    return "domestic"
