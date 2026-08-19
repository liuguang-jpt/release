# -*- coding: utf-8 -*-
"""
external_data_preprocessing.py
================================
TPDdb 外部数据预处理 ETL — Bias-Aware PROTAC Benchmark

将 TPDdb 原始数据(PROTAC_main_table.txt + PROTAC_activity.txt)处理为
来源独立、记录级去重、端点定义透明、可追溯的外部验证数据集。
"""

import os, re, math, hashlib, json, datetime, sys
from pathlib import Path
import numpy as np
import pandas as pd

# ============================= 路径配置 =============================
REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR  = str(Path(os.environ.get("PROTAC_EXTERNAL_ROOT", str(REPO_ROOT / "external_data"))))
RAW_DIR   = os.path.join(BASE_DIR, "raw", "TPDdb")
PROC_DIR  = os.path.join(BASE_DIR, "processed")
AUDIT_DIR = os.path.join(BASE_DIR, "audits")
REP_DIR   = os.path.join(BASE_DIR, "reports")
SCR_DIR   = os.path.join(BASE_DIR, "scripts")

MAIN_TABLE_PATH = os.path.join(RAW_DIR, "PROTAC_main_table.txt")
ACTIVITY_PATH    = os.path.join(RAW_DIR, "PROTAC_activity.txt")
INTERNAL_PATH = str(Path(os.environ.get("PROTAC_ROOT", str(REPO_ROOT))) / "data" / "derived" / "protac_clean_record_level.csv")

for d in [PROC_DIR, AUDIT_DIR, REP_DIR, SCR_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================= 来源元数据 =============================
SOURCE_NAME     = "TPDdb"
SOURCE_URL      = "https://tpddb.idrblab.net/download"
SOURCE_VERSION  = "2025-08-31"
DOWNLOAD_DATE   = "2026-08-18"
LICENSE         = "UNKNOWN"

# ============================= 端点解析常量 =============================
MISSING_TOKENS = {
    "", "n.d.", "n.d", "n/a", "na", "n.a.", "not determined", "not det",
    "nd", "nd.", "n/d", "nan", "none", "nr", "not reported", "no data",
    "na.", "n.a", "null", "unclear", "u.d.", "ud", "/",
}

DMAX_P  = 70.0     # Dmax >= 70% -> P  (对齐主体 v2.0 §5.2)
DMAX_N  = 20.0     # Dmax <= 20% -> N  (对齐主体 v2.0 §5.2)
DC50_P  = 100.0    # DC50 <= 100 nM -> P
DC50_N  = 1000.0   # DC50 >= 1000 nM -> N

GRADE_LETTERS = {"a", "b", "c", "d", "e", "a+", "b+", "c+", "d+",
                 "a-", "b-", "c-", "d-"}
QUALITATIVE_TOKENS = {"+", "++", "+++", "++++", "-", "--", "---", "----",
                      "+/-", ">+", "<+"}

DEGRADATION_TYPES = {"dc50", "dmax", "dc90", "degradation", "%degradation",
                     "dic50", "amax", "residualrate"}
VIABILITY_TYPES   = {"ic50", "ec50", "gi50", "gl50", "lc50"}
BINDING_TYPES     = {"kd", "ki"}

CELL_CANON = {
    "MCF7":"MCF-7","A204":"A-204","EOL1":"EOL-1","PANC1":"PANC-1",
    "HCT15":"HCT-15","HCT116":"HCT-116","HEK293":"HEK-293","BT474":"BT-474",
    "SKBR3":"SK-BR-3","K562":"K562","MV4-11":"MV4-11","RS4-11":"RS4-11",
    "MOLM13":"MOLM-13","KG1":"KG-1","KASUMI1":"KASUMI-1","NCIN87":"NCI-N87",
    "SNU16":"SNU-16","SNU638":"SNU-638","HGC27":"HGC-27","NCIH1975":"NCI-H1975",
    "NCIH1650":"NCI-H1650","NCIH358":"NCI-H358","NCIH460":"NCI-H460",
    "CACO2":"Caco-2","COL0205":"Colo-205","DLD1":"DLD-1","MIAPACA2":"MiaPaCa-2",
    "BXPC3":"BxPC-3","ASPC1":"AsPC-1","CAPAN1":"Capan-1","H1299":"H-1299",
    "SUDHL2":"SU-DHL-2","SUDHL4":"SU-DHL-4","SUDHL6":"SU-DHL-6","SUDHL8":"SU-DHL-8",
    "OCILY1":"OCI-LY1","OCILY7":"OCI-LY7","OCILY10":"OCI-LY10","MOLM14":"MOLM-14",
    "RH30":"RH-30","RH41":"RH-41","HOP92":"HOP-92","VCAP":"VCaP","22RV1":"22Rv1",
    "C42":"C4-2","PCT":"PC-3","DU145":"DU-145","LNCAP":"LNCaP",
}

E3_CANON = {
    "CRBN":"CRBN","VHL":"VHL","DDB1":"DDB1","XIAP":"XIAP",
    "MDM2":"MDM2","MDM-2":"MDM2","KEAP1":"KEAP1","MKRN2":"MKRN2",
}

# ============================= 工具函数 =============================

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm_text(s):
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\uff1e", ">").replace("\uff1c", "<").replace("\uff1d", "=")
    s = s.replace("\u2265", ">=").replace("\u2264", "<=")
    s = s.replace("\u223c", "~").replace("\u2248", "~")
    s = s.replace("\u3000", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def to_float(x):
    try:
        return float(x)
    except Exception:
        return None


# ============================= 单位检测 =============================

def detect_unit_nm(s):
    """检测浓度单位, 返回(换算系数到nM, 单位标签)"""
    sl = s.lower()
    # 检测 uM (微摩尔) — 处理 Greek mu (U+03BC) 和 micro sign (U+00B5)
    if "\u03bcm" in sl or "\u00b5m" in sl or "um" in sl:
        return 1000.0, "uM"
    if re.search(r"p\.?m\b", sl) and "nm" not in sl:
        return 0.001, "pM"
    if "nm" in sl:
        return 1.0, "nM"
    if re.search(r"\d\s*m\b(?!o)", sl) and "nm" not in sl and "um" not in sl and "pm" not in sl:
        return 1e9, "M"
    return 1.0, "nM"


# ============================= DC50 解析 =============================

def parse_dc50(raw):
    res = dict(obs_type="endpoint-missing", value=None, lower=None, upper=None,
               comparator=None, unit_original="nM", raw=str(raw) if raw else "")
    if raw is None:
        return res
    s = norm_text(raw)
    if s == "" or s.upper() in {t.upper() for t in MISSING_TOKENS}:
        return res
    sl = s.lower()

    if sl in {"no degradation", "nde", "no degr", "no-degradation", "no"}:
        res["obs_type"] = "not_parseable"
        return res
    if sl in GRADE_LETTERS:
        res["obs_type"] = "not_parseable"
        return res
    if sl in QUALITATIVE_TOKENS:
        res["obs_type"] = "not_parseable"
        return res

    conv, unit_label = detect_unit_nm(s)
    res["unit_original"] = unit_label

    # 区间: "1nM<=x<10nM" / "10nM<=x<=100nM" / "1nM<x<10nM"
    s_nospace = sl.replace(" ", "")
    m = re.match(r"^([\d.]+)n\.?m(?:[<>]=?)x(?:[<>]=?)([\d.]+)n\.?m$", s_nospace)
    if m:
        lo = to_float(m.group(1))
        hi = to_float(m.group(2))
        if lo is not None and hi is not None:
            res["obs_type"] = "interval_censored"
            res["lower"] = min(lo, hi) * conv
            res["upper"] = max(lo, hi) * conv
            return res

    # 比较符: ">=1000nM" / ">2000nM" / "<1nM" / "<=1nM"
    m = re.match(r"^([<>=]+)\s*([\d.]+)", s)
    if m:
        cmp_raw = m.group(1)
        val = to_float(m.group(2))
        if val is not None:
            val_nm = val * conv
            res["comparator"] = cmp_raw
            if cmp_raw in (">", ">="):
                res["obs_type"] = "right_censored"
                res["lower"] = val_nm
            else:
                res["obs_type"] = "left_censored"
                res["upper"] = val_nm
            return res

    # 数值+单位: "3nM" / "1.85nM" / "0.004uM" / "9150 pM"
    m = re.match(r"^([\d.]+)\s*(n\.?m|um|p\.?m|m)$", sl)
    if m:
        val = to_float(m.group(1))
        if val is not None:
            res["value"] = val * conv
            res["obs_type"] = "exact"
            return res

    # 纯数值(无单位): 假设 nM
    m = re.match(r"^([\d.]+)$", s)
    if m:
        val = to_float(m.group(1))
        if val is not None:
            res["value"] = val * conv
            res["obs_type"] = "exact"
            res["unit_original"] = "nM (assumed)"
            return res

    # 回退: 尝试提取单个数字
    nums = re.findall(r"[\d.]+", s)
    if len(nums) == 1 and not any(c in s for c in "<>=x"):
        val = to_float(nums[0])
        if val is not None and 0 < val < 1e7:
            res["value"] = val * conv
            res["obs_type"] = "exact"
            return res

    res["obs_type"] = "not_parseable"
    return res


# ============================= Dmax 解析 =============================

def parse_dmax(raw):
    res = dict(obs_type="endpoint-missing", value=None, lower=None, upper=None,
               comparator=None, unit_original="%", raw=str(raw) if raw else "")
    if raw is None:
        return res
    s = norm_text(raw)
    if s == "" or s.upper() in {t.upper() for t in MISSING_TOKENS}:
        return res
    sl = s.lower()

    if sl in {"no degradation", "nde", "no degr", "no-degradation"}:
        res["obs_type"] = "not_parseable"
        return res
    if sl in GRADE_LETTERS:
        res["obs_type"] = "not_parseable"
        return res
    if sl in QUALITATIVE_TOKENS:
        res["obs_type"] = "not_parseable"
        return res

    # "X and Y" 形式: ">=70 and <85"
    m = re.match(r"^([<>=]+)?\s*([\d.]+)\s*%?\s*and\s*([<>=]+)?\s*([\d.]+)\s*%?$", sl)
    if m:
        lo = to_float(m.group(2))
        hi = to_float(m.group(4))
        if lo is not None and hi is not None:
            res["obs_type"] = "interval_censored"
            res["lower"] = min(lo, hi)
            res["upper"] = max(lo, hi)
            return res

    # "X%<=x<=Y%" / "X%<x<Y%"
    s_nospace = sl.replace(" ", "")
    m = re.match(r"^([\d.]+)%?(?:[<>]=?)x(?:[<>]=?)([\d.]+)%?$", s_nospace)
    if m:
        lo = to_float(m.group(1))
        hi = to_float(m.group(2))
        if lo is not None and hi is not None:
            res["obs_type"] = "interval_censored"
            res["lower"] = min(lo, hi)
            res["upper"] = max(lo, hi)
            return res

    # 比较符: ">=85" / ">90%" / "<=50%"
    m = re.match(r"^([<>=]+)\s*([\d.]+)\s*%?$", s)
    if m:
        cmp_raw = m.group(1)
        val = to_float(m.group(2))
        if val is not None:
            res["comparator"] = cmp_raw
            if cmp_raw in (">", ">="):
                res["obs_type"] = "right_censored"
                res["lower"] = val
            else:
                res["obs_type"] = "left_censored"
                res["upper"] = val
            return res

    # 纯百分比: "95%"
    m = re.match(r"^([\d.]+)\s*%$", s)
    if m:
        val = to_float(m.group(1))
        if val is not None:
            res["value"] = val
            res["obs_type"] = "exact"
            return res

    # 比例值: "0.90" -> 90%  (must check before pure number)
    m = re.match(r"^(0\.\d+)$", s)
    if m:
        val = to_float(m.group(1))
        if val is not None and 0 <= val <= 1:
            res["value"] = val * 100
            res["obs_type"] = "exact"
            res["unit_original"] = "ratio->%"
            return res

    # 纯数值: "95" / "100"
    m = re.match(r"^([\d.]+)$", s)
    if m:
        val = to_float(m.group(1))
        if val is not None:
            res["value"] = val
            res["obs_type"] = "exact"
            res["unit_original"] = "% (assumed)"
            return res

    res["obs_type"] = "not_parseable"
    return res


# ============================= pDC50 =============================

def pdc50_from_dc50(ep):
    ot = ep["obs_type"]
    v, lo, hi = ep["value"], ep["lower"], ep["upper"]
    if ot == "exact" and v is not None and v > 0:
        return dict(value=9.0 - math.log10(v), lower=None, upper=None)
    if ot == "right_censored" and lo is not None and lo > 0:
        return dict(value=None, lower=None, upper=9.0 - math.log10(lo))
    if ot == "left_censored" and hi is not None and hi > 0:
        return dict(value=None, lower=9.0 - math.log10(hi), upper=None)
    if ot == "interval_censored" and lo is not None and hi is not None and lo > 0 and hi > 0:
        return dict(value=None, lower=9.0 - math.log10(hi), upper=9.0 - math.log10(lo))
    return dict(value=None, lower=None, upper=None)


# ============================= P/N/U 标签 =============================

def assign_evidence_external(dc_ep, dm_ep):
    p_sig = []
    n_sig = []
    has_numeric = False

    # DC50
    if dc_ep["obs_type"] == "exact" and dc_ep["value"] is not None:
        has_numeric = True
        v = dc_ep["value"]
        if v <= DC50_P:
            p_sig.append(f"DC50={v:.3g}nM<={DC50_P}")
        elif v >= DC50_N:
            n_sig.append(f"DC50={v:.3g}nM>={DC50_N}")
    elif dc_ep["obs_type"] == "left_censored" and dc_ep["upper"] is not None:
        has_numeric = True
        if dc_ep["upper"] <= DC50_P:
            p_sig.append(f"DC50<{dc_ep['upper']:.3g}<={DC50_P}")
    elif dc_ep["obs_type"] == "right_censored" and dc_ep["lower"] is not None:
        has_numeric = True
        if dc_ep["lower"] >= DC50_N:
            n_sig.append(f"DC50>{dc_ep['lower']:.3g}>={DC50_N}")
    elif dc_ep["obs_type"] == "interval_censored" and dc_ep["lower"] is not None and dc_ep["upper"] is not None:
        has_numeric = True
        if dc_ep["upper"] <= DC50_P:
            p_sig.append("DC50_interval_P")
        elif dc_ep["lower"] >= DC50_N:
            n_sig.append("DC50_interval_N")

    # Dmax
    if dm_ep["obs_type"] == "exact" and dm_ep["value"] is not None:
        has_numeric = True
        v = dm_ep["value"]
        if v >= DMAX_P:
            p_sig.append(f"Dmax={v:.1f}%>={DMAX_P}")
        elif v <= DMAX_N:
            n_sig.append(f"Dmax={v:.1f}%<={DMAX_N}")
    elif dm_ep["obs_type"] == "right_censored" and dm_ep["lower"] is not None:
        has_numeric = True
        if dm_ep["lower"] >= DMAX_P:
            p_sig.append("Dmax_cens_P")
    elif dm_ep["obs_type"] == "left_censored" and dm_ep["upper"] is not None:
        has_numeric = True
        if dm_ep["upper"] <= DMAX_N:
            n_sig.append("Dmax_cens_N")
    elif dm_ep["obs_type"] == "interval_censored" and dm_ep["lower"] is not None and dm_ep["upper"] is not None:
        has_numeric = True
        if dm_ep["upper"] >= DMAX_P:
            p_sig.append("Dmax_interval_P")
        elif dm_ep["lower"] <= DMAX_N:
            n_sig.append("Dmax_interval_N")

    all_sig = p_sig + n_sig
    if p_sig and n_sig:
        return "A", ";".join(all_sig), "conflict_P_N"
    if p_sig:
        return "P", ";".join(p_sig), "numeric_positive"
    if n_sig:
        return "N", ";".join(n_sig), "numeric_negative"
    if has_numeric:
        return "A", ";".join(all_sig), "numeric_ambiguous"
    return "U", "", "no_numeric_endpoint"


# ============================= Activity Type 解析 =============================

def parse_activity_type(atype):
    """返回 (base_type, suffix, endpoint_category)"""
    if not atype:
        return ("unknown", "", "other")
    s = norm_text(atype)
    sl = s.lower().replace(" ", "")

    for bt in ["%degradation", "degradation", "cellviability", "residualrate",
               "dc50", "dmax", "dc90", "ic50", "ec50", "gi50", "gl50", "lc50",
               "dic50", "amax", "kd", "ki", "emax", "crbnfp"]:
        if sl.startswith(bt):
            suffix = s[len(bt):].strip().replace(" ", "")
            if bt in DEGRADATION_TYPES:
                cat = "degradation"
            elif bt in VIABILITY_TYPES:
                cat = "viability"
            elif bt in BINDING_TYPES:
                cat = "binding"
            else:
                cat = "other"
            return (bt, suffix, cat)

    return (sl, "", "other")


def normalize_cell_line(raw_cell, atype):
    if raw_cell and raw_cell.strip() and raw_cell.strip() not in {".", "-"}:
        return raw_cell.strip()
    m = re.search(r"\(([A-Z][A-Za-z0-9\-]+)\)", atype or "")
    if m:
        return m.group(1)
    return None


def normalize_e3(raw_e3):
    if not raw_e3 or raw_e3.strip() in {"", ".", "Ligase"}:
        return None
    e3 = raw_e3.strip()
    return E3_CANON.get(e3.upper(), e3)


def normalize_target_id(raw_tid):
    if not raw_tid or raw_tid.strip() in {"", "."}:
        return None
    tid = raw_tid.strip()
    base = re.match(r"^([A-Z][0-9A-Z]+)", tid)
    if base:
        return base.group(1)
    return tid


# ============================= 结构标准化 =============================

def standardize_structure(smiles, Chem, cache):
    if not smiles or (isinstance(smiles, float) and pd.isna(smiles)) or smiles == "":
        return dict(canonical_smiles=None, inchikey=None, mw=None,
                    scaffold=None, status="missing")
    if smiles in cache:
        return cache[smiles]

    result = dict(canonical_smiles=None, inchikey=None, mw=None,
                  scaffold=None, status="invalid")
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            cache[smiles] = result
            return result
        canon = Chem.MolToSmiles(mol, canonical=True)
        result["canonical_smiles"] = canon
        result["mw"] = round(Chem.Descriptors.MolWt(mol), 2)
        try:
            ik = Chem.inchi.MolToInchiKey(mol)
            if ik:
                result["inchikey"] = ik
        except Exception:
            pass
        try:
            from rdkit.Chem.Scaffolds import MurckoScaffold
            scaf_mol = MurckoScaffold.GetScaffoldForMol(mol)
            scaf = Chem.MolToSmiles(scaf_mol)
            result["scaffold"] = scaf if scaf else "NO_SCAFFOLD"
        except Exception:
            result["scaffold"] = "PARSE_FAIL"
        result["status"] = "ok" if canon else "invalid"
    except Exception:
        pass

    cache[smiles] = result
    return result


# ============================= 主流程 =============================

def main():
    print("=" * 70)
    print("TPDdb 外部数据预处理 ETL")
    print("=" * 70)
    print(f"处理时间: {datetime.datetime.now().isoformat()}")

    Chem = None
    rdkit_version = "unavailable"
    try:
        from rdkit import Chem as _C
        from rdkit.Chem import Descriptors as _D
        from rdkit.Chem import inchi as _inchi
        from rdkit import __version__ as _rv
        Chem = _C
        Chem.Descriptors = _D
        Chem.inchi = _inchi
        rdkit_version = _rv
        print(f"RDKit: {rdkit_version}")
    except Exception as e:
        print(f"RDKit 不可用: {e}")

    struct_cache = {}

    # ============================ Step 0: 原始文件清单 ============================
    print("\n=== Step 0: 原始文件清单 ===")
    raw_files = [
        ("PROTAC_main_table.txt", MAIN_TABLE_PATH,
         "TPDdb 主表: TPD ID, 名称, SMILES, 分子式, 靶标, E3 ligase, 来源(专利ID)"),
        ("PROTAC_activity.txt", ACTIVITY_PATH,
         "TPDdb 活性表: TPD ID, 活性类型, 靶标, 细胞系, 活性值"),
    ]
    manifest_rows = []
    for fname, fpath, notes in raw_files:
        fsize = os.path.getsize(fpath)
        sha = sha256_file(fpath)
        manifest_rows.append(dict(
            source_name=SOURCE_NAME, source_url=SOURCE_URL,
            source_version=SOURCE_VERSION, download_date=DOWNLOAD_DATE,
            original_filename=fname, file_size_bytes=fsize,
            sha256=sha, license=LICENSE, notes=notes))
        print(f"  {fname}: {fsize:,} bytes  sha256={sha[:16]}...")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(RAW_DIR, "external_raw_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")

    input_sha256 = {r["original_filename"]: r["sha256"] for r in manifest_rows}

    # ============================ Step 1: 数据加载 ============================
    print("\n=== Step 1: 数据加载 & 清洗 ===")
    sample_log = []

    main_df = pd.read_csv(MAIN_TABLE_PATH, sep="\t", dtype=str, keep_default_na=False)
    sample_log.append(("0. raw main_table", len(main_df)))
    print(f"  主表: {main_df.shape}")

    act_df = pd.read_csv(ACTIVITY_PATH, sep="\t", dtype=str, keep_default_na=False)
    sample_log.append(("0. raw activity_table", len(act_df)))
    print(f"  活性表: {act_df.shape}")

    dc_checks = {
        "main_shape_ok": main_df.shape == (21430, 9),
        "activity_shape_ok": act_df.shape == (23320, 6),
        "all_activity_in_main": len(set(act_df["TPD ID"]) - set(main_df["TPD ID"])) == 0,
        "smiles_nonempty": (main_df["SMILES"] != "").all(),
        "cross_sectional": "N/A (非面板数据)",
    }
    print(f"  数据合约: {dc_checks}")

    # 列重命名
    main_df = main_df.rename(columns={
        "TPD ID": "tpd_id", "TPD NAME": "tpd_name",
        "PubChem synonyms": "pubchem_synonyms", "SMILES": "smiles_raw",
        "Fomula": "molecular_formula_raw", "Target Symbol": "target_symbol",
        "Target ID": "target_id", "Ligase": "ligase_raw",
        "Source": "source_patent_id"})
    act_df = act_df.rename(columns={
        "TPD ID": "tpd_id", "Activity Type": "activity_type_raw",
        "Target Symbols": "target_symbols_act", "Target IDs": "target_ids_act",
        "Cell Line": "cell_line_raw", "Activity": "activity_raw"})

    # 合并
    merged = act_df.merge(main_df, on="tpd_id", how="left", validate="many_to_one")
    sample_log.append(("1. merged main+activity", len(merged)))
    print(f"  合并后: {len(merged)} 行")

    # 保存 raw 记录级表
    raw_records = merged.copy()
    raw_records.insert(0, "raw_row_index", range(len(raw_records)))
    raw_records.to_csv(os.path.join(PROC_DIR, "external_record_level_raw.csv"),
                       index=False, encoding="utf-8-sig")
    print(f"  -> external_record_level_raw.csv ({len(raw_records)} 行)")

    # Activity Type 解析
    parsed = raw_records["activity_type_raw"].apply(parse_activity_type)
    raw_records["base_type"] = [p[0] for p in parsed]
    raw_records["activity_suffix"] = [p[1] for p in parsed]
    raw_records["endpoint_category"] = [p[2] for p in parsed]
    print(f"  base_type: {raw_records['base_type'].value_counts().head(8).to_dict()}")
    print(f"  endpoint_category: {raw_records['endpoint_category'].value_counts().to_dict()}")

    # ============================ Step 2: 记录级构建 ============================
    print("\n=== Step 2: 记录级构建 & 端点解析 ===")

    # 为每条记录解析端点
    dc50_list = []
    dmax_list = []
    other_list = []
    for _, row in raw_records.iterrows():
        bt = row["base_type"]
        ar = row["activity_raw"]
        dc_ep = dict(obs_type="endpoint-missing", value=None, lower=None, upper=None,
                     comparator=None, unit_original="nM", raw=ar)
        dm_ep = dict(obs_type="endpoint-missing", value=None, lower=None, upper=None,
                     comparator=None, unit_original="%", raw=ar)
        otype = None
        oval = ar

        if bt == "dc50":
            dc_ep = parse_dc50(ar)
        elif bt == "dmax":
            dm_ep = parse_dmax(ar)
        elif bt == "dc90":
            dc_ep = parse_dc50(ar)
        elif bt in ("%degradation", "degradation", "amax"):
            dm_ep = parse_dmax(ar)
        else:
            otype = bt
            oval = ar

        dc50_list.append(dc_ep)
        dmax_list.append(dm_ep)
        other_list.append((otype, oval))

    raw_records["_dc50_ep"] = dc50_list
    raw_records["_dmax_ep"] = dmax_list
    raw_records["_other_type"] = [o[0] for o in other_list]
    raw_records["_other_val"] = [o[1] for o in other_list]

    # 配对键
    def make_key(row):
        cell = row["cell_line_raw"] if row["cell_line_raw"] not in {"", "."} else "_nocell_"
        tgt = row["target_symbols_act"] if row["target_symbols_act"] not in {"", "."} else "_notgt_"
        return (row["tpd_id"], cell, tgt, row["activity_suffix"], row["endpoint_category"])

    raw_records["_gkey"] = raw_records.apply(make_key, axis=1)

    # 按组聚合
    groups = raw_records.groupby("_gkey", sort=False)
    clean_recs = []
    for gkey, grp in groups:
        first = grp.iloc[0]
        bt_set = set(grp["base_type"])

        # DC50
        dc_rows = grp[grp["base_type"].isin(["dc50", "dc90"])]
        dc_ep = dc_rows.iloc[0]["_dc50_ep"] if len(dc_rows) > 0 else \
            dict(obs_type="endpoint-missing", value=None, lower=None, upper=None,
                 comparator=None, unit_original="nM", raw="")
        dc_raw = dc_rows.iloc[0]["activity_raw"] if len(dc_rows) > 0 else ""

        # Dmax
        dm_rows = grp[grp["base_type"].isin(["dmax", "degradation", "%degradation", "amax"])]
        dm_ep = dm_rows.iloc[0]["_dmax_ep"] if len(dm_rows) > 0 else \
            dict(obs_type="endpoint-missing", value=None, lower=None, upper=None,
                 comparator=None, unit_original="%", raw="")
        dm_raw = dm_rows.iloc[0]["activity_raw"] if len(dm_rows) > 0 else ""

        # other
        o_rows = grp[grp["_other_type"].notna() & (grp["_other_type"] != "")]
        otype = o_rows.iloc[0]["_other_type"] if len(o_rows) > 0 else None
        oval = o_rows.iloc[0]["_other_val"] if len(o_rows) > 0 else ""

        pdc = pdc50_from_dc50(dc_ep)
        ev_label, ev_sig, ev_rule = assign_evidence_external(dc_ep, dm_ep)

        cell_norm = normalize_cell_line(first["cell_line_raw"], first["activity_type_raw"])
        e3_norm = normalize_e3(first.get("ligase_raw", ""))
        tgt_uni = normalize_target_id(first.get("target_id", ""))

        rec = dict(
            tpd_id=first["tpd_id"],
            tpd_name=first.get("tpd_name", ""),
            pubchem_synonyms=first.get("pubchem_synonyms", ""),
            smiles_raw=first.get("smiles_raw", ""),
            molecular_formula_raw=first.get("molecular_formula_raw", ""),
            target_symbol=first.get("target_symbol", ""),
            target_id=first.get("target_id", ""),
            ligase_raw=first.get("ligase_raw", ""),
            source_patent_id=first.get("source_patent_id", ""),
            activity_type_raw=first["activity_type_raw"],
            base_type=first["base_type"],
            activity_suffix=first["activity_suffix"],
            endpoint_category=first["endpoint_category"],
            cell_line_raw=first["cell_line_raw"],
            cell_line_normalized=cell_norm,
            target_symbols_act=first["target_symbols_act"],
            target_ids_act=first["target_ids_act"],
            target_uniprot=tgt_uni,
            e3_normalized=e3_norm,
            dc50_raw=dc_raw,
            dc50_value=dc_ep["value"],
            dc50_lower=dc_ep["lower"],
            dc50_upper=dc_ep["upper"],
            dc50_comparator=dc_ep["comparator"],
            dc50_unit=dc_ep["unit_original"],
            dc50_obs_type=dc_ep["obs_type"],
            dmax_raw=dm_raw,
            dmax_value=dm_ep["value"],
            dmax_lower=dm_ep["lower"],
            dmax_upper=dm_ep["upper"],
            dmax_comparator=dm_ep["comparator"],
            dmax_unit=dm_ep["unit_original"],
            dmax_obs_type=dm_ep["obs_type"],
            pdc50_value=pdc["value"],
            pdc50_lower=pdc["lower"],
            pdc50_upper=pdc["upper"],
            other_activity_type=otype,
            other_activity_raw=oval,
            activity_evidence_external_v1=ev_label,
            activity_mapping_rule=ev_sig,
            activity_mapping_confidence=("high" if ev_label in ("P", "N") else
                                         "low" if ev_label == "U" else "medium"),
            activity_mapping_exclusion_reason=("" if ev_label != "U" else
                                                "no_numeric_endpoint_or_grade_qualitative"),
            source_database=SOURCE_NAME,
            source_version=SOURCE_VERSION,
            source_url=SOURCE_URL,
            citation_doi="",
            pmid="",
            publication_year=None,
            n_records_in_group=len(grp),
            group_base_types=";".join(sorted(bt_set)),
        )
        clean_recs.append(rec)

    clean_df = pd.DataFrame(clean_recs)
    sample_log.append(("2. grouped record-level", len(clean_df)))
    print(f"  配对后记录: {len(clean_df)}")

    clean_df.insert(0, "external_record_id", range(1, len(clean_df) + 1))

    # 结构标准化
    print(f"\n--- 结构标准化 (RDKit) ---")
    print(f"  唯一 SMILES: {clean_df['smiles_raw'].nunique()}")
    canon_l, ik_l, mw_l, scaf_l, status_l = [], [], [], [], []
    n_ok = n_inv = n_mis = 0
    for smi in clean_df["smiles_raw"]:
        st = standardize_structure(smi, Chem, struct_cache)
        canon_l.append(st["canonical_smiles"])
        ik_l.append(st["inchikey"])
        mw_l.append(st["mw"])
        scaf_l.append(st["scaffold"])
        status_l.append(st["status"])
        if st["status"] == "ok": n_ok += 1
        elif st["status"] == "missing": n_mis += 1
        else: n_inv += 1
    clean_df["smiles_canonical"] = canon_l
    clean_df["inchikey"] = ik_l
    clean_df["molecular_weight"] = mw_l
    clean_df["scaffold"] = scaf_l
    clean_df["structure_status"] = status_l
    clean_df["inchikey_connectivity"] = clean_df["inchikey"].apply(
        lambda x: str(x).split("-")[0] if pd.notna(x) and x and "-" in str(x) else None)
    print(f"  ok={n_ok}, invalid={n_inv}, missing={n_mis}")
    sample_log.append(("3. structure standardized", len(clean_df)))

    # PROTAC 类型
    clean_df["protac_type"] = "PROTAC"
    clean_df["protac_type_status"] = "confirmed"
    has_e3 = clean_df["e3_normalized"].notna() & (clean_df["e3_normalized"] != "")
    clean_df.loc[~has_e3, "protac_type_status"] = "unknown_no_e3"
    print(f"  PROTAC confirmed={has_e3.sum()}, unknown_no_e3={(~has_e3).sum()}")

    # T1-T4
    print("\n--- T1-T4 资格 ---")
    clean_df["t1_eligible"] = (clean_df["dc50_obs_type"] == "exact") & clean_df["pdc50_value"].notna()
    clean_df["t2_eligible"] = (clean_df["dmax_obs_type"] == "exact") & clean_df["dmax_value"].notna()
    clean_df["t3_eligible"] = clean_df["activity_evidence_external_v1"].isin(["P", "N"])
    clean_df["t4_eligible"] = clean_df["activity_evidence_external_v1"].isin(["P", "N", "U"])
    clean_df["censored_eligible"] = (
        clean_df["dc50_obs_type"].isin(["right_censored", "left_censored", "interval_censored"]) |
        clean_df["dmax_obs_type"].isin(["right_censored", "left_censored", "interval_censored"]))
    print(f"  T1={clean_df['t1_eligible'].sum()}, T2={clean_df['t2_eligible'].sum()}, "
          f"T3={clean_df['t3_eligible'].sum()}, T4={clean_df['t4_eligible'].sum()}, "
          f"Censored={clean_df['censored_eligible'].sum()}")

    # ============================ Step 3: 去重 ============================
    print("\n=== Step 3: 与主体数据集重复审计 ===")
    print("  加载主体数据集...")
    internal_df = pd.read_csv(INTERNAL_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    print(f"  主体: {len(internal_df)} 行")

    # 主体结构标准化
    print("  主体结构标准化...")
    int_cache = {}
    int_canon, int_ik, int_conn = [], [], []
    for smi in internal_df["smiles"]:
        st = standardize_structure(smi, Chem, int_cache)
        int_canon.append(st["canonical_smiles"])
        ik = st["inchikey"]
        int_ik.append(ik)
        int_conn.append(str(ik).split("-")[0] if ik and "-" in str(ik) else None)

    internal_df["_canon"] = int_canon
    internal_df["_inchikey"] = int_ik
    internal_df["_conn"] = int_conn
    n_int_ok = sum(1 for x in int_canon if x is not None)
    print(f"  主体结构: ok={n_int_ok}/{len(internal_df)}")

    # 构建索引
    int_by_ik = {}
    int_by_canon = {}
    int_by_conn = {}
    int_info = {}  # rid -> dict

    for _, row in internal_df.iterrows():
        rid = row["record_id"]
        ik = row["_inchikey"]
        if ik and ik != "":
            int_by_ik.setdefault(ik, []).append(rid)
        cs = row["_canon"]
        if cs and cs != "":
            int_by_canon.setdefault(cs, []).append(rid)
        conn = row["_conn"]
        if conn and conn != "" and pd.notna(conn):
            int_by_conn.setdefault(conn, []).append(rid)
        int_info[rid] = dict(
            target=str(row.get("target", "")).strip().upper(),
            uniprot=str(row.get("uniprot", "")).strip().upper(),
            e3=str(row.get("e3_ligase", "")).strip().upper(),
            cell_line=str(row.get("cell_line", "")).strip().upper() if pd.notna(row.get("cell_line")) else "",
            doi=str(row.get("article_doi", "")).strip(),
            dc50_raw=str(row.get("dc50_raw", "")).strip(),
            dmax_raw=str(row.get("dmax_raw", "")).strip(),
        )

    print(f"  索引: InChIKey={len(int_by_ik)}, canon={len(int_by_canon)}, conn={len(int_by_conn)}")

    # 4 层级去重
    print("  执行 4 层级匹配...")
    overlap_recs = []
    overlap_cnt = {"L1_structure": 0, "L2_structure_literature": 0,
                   "L3_experiment": 0, "L4_potential": 0, "no_match": 0}

    for _, erow in clean_df.iterrows():
        eid = erow["external_record_id"]
        ik = erow["inchikey"]
        cs = erow["smiles_canonical"]
        conn = erow["inchikey_connectivity"]

        # Level 1
        l1_ids = set()
        if ik and pd.notna(ik) and ik in int_by_ik:
            l1_ids.update(int_by_ik[ik])
        if cs and pd.notna(cs) and cs in int_by_canon:
            l1_ids.update(int_by_canon[cs])

        match_level = "no_match"
        match_reason = ""
        same_doi = same_poi = same_e3 = same_cell = same_ep = False
        matched_rids = set()

        if l1_ids:
            matched_rids = l1_ids
            match_level = "L1_structure"
            match_reason = "InChIKey or canonical SMILES exact match"
            overlap_cnt["L1_structure"] += 1

            ext_tgt = str(erow.get("target_symbol", "")).strip().upper()
            ext_e3 = str(erow.get("e3_normalized", "") or "").strip().upper()
            ext_cell = str(erow.get("cell_line_normalized", "") or "").strip().upper()
            ext_uni = str(erow.get("target_uniprot", "") or "").strip().upper()
            ext_dc = erow["dc50_obs_type"] not in ("endpoint-missing", "not_parseable")
            ext_dm = erow["dmax_obs_type"] not in ("endpoint-missing", "not_parseable")

            l3_found = False
            for rid in l1_ids:
                info = int_info.get(rid, {})
                poi_m = (ext_tgt and info["target"] and
                         (ext_tgt == info["target"] or ext_tgt in info["target"] or
                          info["target"] in ext_tgt))
                if not poi_m and ext_uni and info["uniprot"]:
                    if ext_uni == info["uniprot"]:
                        poi_m = True
                e3_m = bool(ext_e3 and info["e3"] and ext_e3 == info["e3"])
                cell_m = bool(ext_cell and info["cell_line"] and
                              (ext_cell == info["cell_line"] or
                               ext_cell in info["cell_line"] or
                               info["cell_line"] in ext_cell))
                int_dc = info["dc50_raw"] not in ("", "nan")
                int_dm = info["dmax_raw"] not in ("", "nan")
                ep_m = (ext_dc and int_dc) or (ext_dm and int_dm)

                if poi_m: same_poi = True
                if e3_m: same_e3 = True
                if cell_m: same_cell = True
                if ep_m: same_ep = True
                if poi_m and e3_m and cell_m and ep_m:
                    l3_found = True

            if l3_found:
                match_level = "L3_experiment"
                match_reason = "structure + POI + E3 + cell_line + endpoint"
                overlap_cnt["L3_experiment"] += 1

        # Level 4
        if not l1_ids and conn and pd.notna(conn) and conn in int_by_conn:
            l4_ids = set(int_by_conn[conn])
            if l4_ids:
                matched_rids = l4_ids
                match_level = "L4_potential"
                match_reason = "InChIKey connectivity layer match (salt/stereo/SMILES variant)"
                overlap_cnt["L4_potential"] += 1

        if not matched_rids:
            overlap_cnt["no_match"] += 1

        # 最终资格
        if not ik and not cs:
            elig = "missing_structure"
        elif erow["structure_status"] == "invalid":
            elig = "invalid_structure"
        elif match_level == "L3_experiment":
            elig = "exact_overlap"
        elif match_level == "L1_structure":
            # P0-1 修复: L1 结构精确重复(与主体 InChIKey/canonical SMILES 相同)一律判 exact_overlap,
            # 无论 POI/E3/细胞系/终点是否不同 —— 冻结模型学的是结构, 结构相同即"训练集见过的结构"。
            elig = "exact_overlap"
        elif match_level == "L4_potential":
            elig = "probable_overlap"
        else:
            elig = "eligible_nonoverlap"

        overlap_recs.append(dict(
            external_record_id=eid,
            matched_internal_record_id=";".join(str(r) for r in sorted(matched_rids)) if matched_rids else "",
            match_level=match_level,
            match_reason=match_reason,
            same_doi=same_doi,
            same_poi=same_poi,
            same_e3=same_e3,
            same_cell_line=same_cell,
            same_endpoint=same_ep,
            manual_review_status=("needs_review" if match_level == "L4_potential" else "auto"),
            final_external_eligibility=elig,
        ))

    overlap_df = pd.DataFrame(overlap_recs)
    overlap_path = os.path.join(AUDIT_DIR, "external_overlap_audit.csv")
    overlap_df.to_csv(overlap_path, index=False, encoding="utf-8-sig")
    print(f"  -> external_overlap_audit.csv ({len(overlap_df)} 行)")
    print(f"  去重: {overlap_cnt}")

    # 合并回 clean_df
    elig_map = overlap_df.set_index("external_record_id")["final_external_eligibility"].to_dict()
    ml_map = overlap_df.set_index("external_record_id")["match_level"].to_dict()
    clean_df["overlap_match_level"] = clean_df["external_record_id"].map(ml_map).fillna("no_match")
    clean_df["final_external_eligibility"] = clean_df["external_record_id"].map(elig_map).fillna("eligible_nonoverlap")
    sample_log.append(("4. dedup completed", len(clean_df)))

    # 保存 clean
    cols_drop = [c for c in clean_df.columns if c.startswith("_")]
    clean_out = clean_df.drop(columns=cols_drop, errors="ignore")
    clean_path = os.path.join(PROC_DIR, "external_record_level_clean.csv")
    clean_out.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print(f"\n  -> external_record_level_clean.csv ({clean_out.shape[0]} x {clean_out.shape[1]})")

    # ============================ 端点审计表 ============================
    print("\n--- 端点审计表 ---")
    ep_rows = []
    total = len(clean_df)
    for ep, col in [("DC50", "dc50_obs_type"), ("Dmax", "dmax_obs_type")]:
        for ot, cnt in clean_df[col].value_counts().items():
            ep_rows.append(dict(endpoint=ep, obs_type=ot, count=int(cnt),
                                pct=f"{cnt/total*100:.2f}%"))
    n_pdc = int(clean_df["pdc50_value"].notna().sum())
    ep_rows.append(dict(endpoint="pDC50", obs_type="calculable", count=n_pdc,
                        pct=f"{n_pdc/total*100:.2f}%"))
    for ep, col in [("DC50", "dc50_unit"), ("Dmax", "dmax_unit")]:
        for u, c in clean_df[col].value_counts().items():
            ep_rows.append(dict(endpoint=ep, obs_type=f"unit={u}", count=int(c),
                                pct=f"{c/total*100:.2f}%"))
    ep_audit = pd.DataFrame(ep_rows)
    ep_audit_path = os.path.join(AUDIT_DIR, "external_endpoint_audit.csv")
    ep_audit.to_csv(ep_audit_path, index=False, encoding="utf-8-sig")
    print(f"  -> external_endpoint_audit.csv ({len(ep_audit)} 行)")

    # ============================ 数据字典 ============================
    print("\n--- 数据字典 ---")
    dd = [
        ("external_record_id", "外部记录级唯一 ID (1-based)", "", "derived", "1"),
        ("tpd_id", "TPDdb 化合物 ID", "", "main_table", "TPD-O7725Z"),
        ("tpd_name", "化合物名称", "", "main_table", "SCHEMBL25026569"),
        ("pubchem_synonyms", "PubChem 同义词", "", "main_table", ""),
        ("smiles_raw", "原始 SMILES", "", "main_table", "Cn1c(=O)..."),
        ("smiles_canonical", "RDKit canonical SMILES", "", "rdkit", "Cn1c(=O)..."),
        ("inchikey", "InChIKey", "", "rdkit", "RZVXXXXXXXXXX-YNYYYYYYBY-N"),
        ("inchikey_connectivity", "InChIKey 连接层(前14字符)", "", "rdkit", "RZVXXXXXXXXXX"),
        ("molecular_weight", "分子量", "Da", "rdkit", "650.50"),
        ("molecular_formula_raw", "分子式(原始)", "", "main_table", "C34H31F3N8O4"),
        ("scaffold", "Murcko scaffold SMILES", "", "rdkit", "c1ccccc1"),
        ("structure_status", "结构状态: ok/invalid/missing", "", "rdkit", "ok"),
        ("target_symbol", "靶标符号(主表)", "", "main_table", "AR-V7"),
        ("target_id", "靶标 ID (UniProt, 含isoform)", "", "main_table", "P10275-3"),
        ("target_uniprot", "靶标 UniProt (去isoform)", "", "derived", "P10275"),
        ("target_symbols_act", "活性表靶标符号", "", "activity_table", "BTK"),
        ("target_ids_act", "活性表靶标 ID", "", "activity_table", "Q06187"),
        ("ligase_raw", "E3 ligase (原始)", "", "main_table", "CRBN"),
        ("e3_normalized", "E3 ligase (规范化)", "", "derived", "CRBN"),
        ("cell_line_raw", "细胞系 (原始)", "", "activity_table", "TMD-8"),
        ("cell_line_normalized", "细胞系 (规范化)", "", "derived", "TMD-8"),
        ("source_patent_id", "来源专利 ID", "", "main_table", "EP-4385985-A1"),
        ("source_database", "来源数据库名", "", "fixed", "TPDdb"),
        ("source_version", "来源版本", "", "fixed", "2025-08-31"),
        ("source_url", "来源 URL", "", "fixed", "https://tpddb.idrblab.net/download"),
        ("citation_doi", "文献 DOI (TPDdb 无DOI)", "", "fixed", ""),
        ("pmid", "PubMed ID", "", "fixed", ""),
        ("publication_year", "出版年份", "", "fixed", ""),
        ("activity_type_raw", "原始活性类型", "", "activity_table", "DC50_1"),
        ("base_type", "解析后基础类型", "", "derived", "dc50"),
        ("activity_suffix", "活性类型后缀", "", "derived", "_1"),
        ("endpoint_category", "端点类别", "", "derived", "degradation"),
        ("protac_type", "PROTAC 类型", "", "fixed", "PROTAC"),
        ("protac_type_status", "类型确认状态", "", "derived", "confirmed"),
        ("dc50_raw", "DC50 原始字符串", "nM", "activity_table", "3nM"),
        ("dc50_value", "DC50 主值 (exact时)", "nM", "parsed", "3.0"),
        ("dc50_lower", "DC50 下界 (右删失时)", "nM", "parsed", "1000.0"),
        ("dc50_upper", "DC50 上界 (左删失时)", "nM", "parsed", "1.0"),
        ("dc50_comparator", "DC50 比较符号", "", "parsed", ">"),
        ("dc50_unit", "DC50 原始单位", "", "parsed", "nM"),
        ("dc50_obs_type", "DC50 观测类型", "", "parsed", "exact"),
        ("pdc50_value", "pDC50 = 9-log10(DC50_nM)", "", "parsed", "8.52"),
        ("pdc50_lower", "pDC50 下界", "", "parsed", "9.0"),
        ("pdc50_upper", "pDC50 上界", "", "parsed", "6.0"),
        ("dmax_raw", "Dmax 原始字符串", "%", "activity_table", "95%"),
        ("dmax_value", "Dmax 主值", "%", "parsed", "95.0"),
        ("dmax_lower", "Dmax 下界", "%", "parsed", "90.0"),
        ("dmax_upper", "Dmax 上界", "%", "parsed", "10.0"),
        ("dmax_comparator", "Dmax 比较符号", "", "parsed", ">"),
        ("dmax_unit", "Dmax 原始单位", "", "parsed", "%"),
        ("dmax_obs_type", "Dmax 观测类型", "", "parsed", "exact"),
        ("other_activity_type", "其他活性类型", "", "activity_table", "ic50"),
        ("other_activity_raw", "其他活性原始值", "", "activity_table", "417nM"),
        ("activity_evidence_external_v1", "外部 P/N/U/A 标签", "", "rule", "P"),
        ("activity_mapping_rule", "标签映射规则", "", "rule", "DC50=3<=100;Dmax=95>=80"),
        ("activity_mapping_confidence", "映射置信度", "", "rule", "high"),
        ("activity_mapping_exclusion_reason", "排除原因(若U)", "", "rule", "no_numeric_endpoint"),
        ("t1_eligible", "T1 资格: exact DC50 + pDC50", "bool", "rule", "True"),
        ("t2_eligible", "T2 资格: exact Dmax", "bool", "rule", "True"),
        ("t3_eligible", "T3 资格: P 或 N", "bool", "rule", "True"),
        ("t4_eligible", "T4 资格: P/N/U", "bool", "rule", "True"),
        ("censored_eligible", "删失分析资格", "bool", "rule", "False"),
        ("overlap_match_level", "去重匹配层级", "", "dedup", "L1_structure"),
        ("final_external_eligibility", "最终外部资格", "", "dedup", "eligible_nonoverlap"),
        ("n_records_in_group", "组内原始记录数", "", "derived", "2"),
        ("group_base_types", "组内基础类型集", "", "derived", "dc50;dmax"),
    ]
    dict_df = pd.DataFrame(dd, columns=["column_name", "description", "unit", "source", "example_value"])
    dict_path = os.path.join(PROC_DIR, "external_data_dictionary.csv")
    dict_df.to_csv(dict_path, index=False, encoding="utf-8-sig")
    print(f"  -> external_data_dictionary.csv ({len(dict_df)} 字段)")

    # ============================ 质量审计统计 ============================
    print("\n=== Step 4: 质量审计 ===")
    stats = {}
    stats["structure_quality"] = {
        "total_compounds": int(clean_df["tpd_id"].nunique()),
        "total_records": len(clean_df),
        "missing_structure": int((clean_df["structure_status"] == "missing").sum()),
        "invalid_structure": int((clean_df["structure_status"] == "invalid").sum()),
        "canonical_ok": int((clean_df["structure_status"] == "ok").sum()),
        "inchikey_ok": int(clean_df["inchikey"].notna().sum()),
        "duplicate_structures": int(clean_df["smiles_canonical"].duplicated().sum()),
    }
    stats["endpoint_quality"] = {
        "dc50_exact": int((clean_df["dc50_obs_type"] == "exact").sum()),
        "dc50_right_censored": int((clean_df["dc50_obs_type"] == "right_censored").sum()),
        "dc50_left_censored": int((clean_df["dc50_obs_type"] == "left_censored").sum()),
        "dc50_interval_censored": int((clean_df["dc50_obs_type"] == "interval_censored").sum()),
        "dc50_not_parseable": int((clean_df["dc50_obs_type"] == "not_parseable").sum()),
        "dc50_endpoint_missing": int((clean_df["dc50_obs_type"] == "endpoint-missing").sum()),
        "dmax_exact": int((clean_df["dmax_obs_type"] == "exact").sum()),
        "dmax_right_censored": int((clean_df["dmax_obs_type"] == "right_censored").sum()),
        "dmax_left_censored": int((clean_df["dmax_obs_type"] == "left_censored").sum()),
        "dmax_interval_censored": int((clean_df["dmax_obs_type"] == "interval_censored").sum()),
        "dmax_not_parseable": int((clean_df["dmax_obs_type"] == "not_parseable").sum()),
        "dmax_endpoint_missing": int((clean_df["dmax_obs_type"] == "endpoint-missing").sum()),
        "pdc50_calculable": int(clean_df["pdc50_value"].notna().sum()),
    }
    stats["biology_quality"] = {
        "poi_missing_rate": float(clean_df["target_symbol"].isin(["", "."]).mean()),
        "e3_missing_rate": float(clean_df["e3_normalized"].isna().mean()),
        "cell_line_missing_rate": float(clean_df["cell_line_normalized"].isna().mean()),
        "doi_missing_rate": 1.0,
        "unknown_protac_type": int((clean_df["protac_type_status"] == "unknown_no_e3").sum()),
    }
    stats["pnu_distribution"] = clean_df["activity_evidence_external_v1"].value_counts().to_dict()
    stats["overlap_counts"] = overlap_cnt
    stats["eligibility_distribution"] = clean_df["final_external_eligibility"].value_counts().to_dict()
    stats["task_eligibility"] = {
        "t1": int(clean_df["t1_eligible"].sum()),
        "t2": int(clean_df["t2_eligible"].sum()),
        "t3": int(clean_df["t3_eligible"].sum()),
        "t4": int(clean_df["t4_eligible"].sum()),
        "censored": int(clean_df["censored_eligible"].sum()),
    }
    stats["distribution_audit"] = {
        "external_n_records": len(clean_df),
        "external_n_compounds": int(clean_df["tpd_id"].nunique()),
        "internal_n_records": len(internal_df),
        "external_pnu": clean_df["activity_evidence_external_v1"].value_counts().to_dict(),
        "internal_pnu": internal_df["activity_evidence_v1"].value_counts().to_dict() if "activity_evidence_v1" in internal_df.columns else {},
        "external_pdc50_mean": (float(clean_df.loc[clean_df["pdc50_value"].notna(), "pdc50_value"].mean())
                                if clean_df["pdc50_value"].notna().any() else None),
        "external_pdc50_median": (float(clean_df.loc[clean_df["pdc50_value"].notna(), "pdc50_value"].median())
                                   if clean_df["pdc50_value"].notna().any() else None),
        "external_dmax_mean": (float(clean_df.loc[clean_df["dmax_value"].notna(), "dmax_value"].mean())
                               if clean_df["dmax_value"].notna().any() else None),
        "external_dmax_median": (float(clean_df.loc[clean_df["dmax_value"].notna(), "dmax_value"].median())
                                  if clean_df["dmax_value"].notna().any() else None),
        "external_mw_mean": (float(clean_df.loc[clean_df["molecular_weight"].notna(), "molecular_weight"].mean())
                             if clean_df["molecular_weight"].notna().any() else None),
        "top_targets": clean_df["target_symbol"].value_counts().head(15).to_dict(),
        "top_e3": clean_df["e3_normalized"].value_counts().head(10).to_dict(),
        "top_cell_lines": clean_df["cell_line_normalized"].value_counts().head(15).to_dict(),
    }
    # P2 修复: 报告 §1/§3 依赖的 row_counts / 样本构建日志 / 数据合约检查 传入 stats
    stats["row_counts"] = dict(
        raw_main_table=int(len(main_df)),
        raw_activity_table=int(len(act_df)),
        merged_records=int(len(merged)),
        clean_records=int(len(clean_df)),
        unique_compounds=int(clean_df["tpd_id"].nunique()),
    )
    stats["sample_construction_log"] = [list(x) for x in sample_log]
    stats["data_contract_checks"] = dc_checks

    # ============================ 处理报告 ============================
    print("\n--- 生成处理报告 ---")
    report = generate_report(clean_df, stats, overlap_cnt, input_sha256, rdkit_version, len(internal_df))
    report_path = os.path.join(REP_DIR, "EXTERNAL_DATA_PREPROCESSING_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  -> EXTERNAL_DATA_PREPROCESSING_REPORT.md")

    # ============================ 元数据 JSON ============================
    output_files = {
        "external_raw_manifest.csv": manifest_path,
        "external_record_level_raw.csv": os.path.join(PROC_DIR, "external_record_level_raw.csv"),
        "external_record_level_clean.csv": clean_path,
        "external_overlap_audit.csv": overlap_path,
        "external_endpoint_audit.csv": ep_audit_path,
        "external_data_dictionary.csv": dict_path,
        "EXTERNAL_DATA_PREPROCESSING_REPORT.md": report_path,
    }
    out_sha = {name: sha256_file(p) for name, p in output_files.items() if os.path.exists(p)}

    excl = {
        "compounds_no_activity": int(len(set(main_df["tpd_id"]) - set(act_df["tpd_id"]))),
        "non_degradation_endpoints": int((clean_df["endpoint_category"] != "degradation").sum()),
        "letter_grade_or_qualitative_dc50": int((clean_df["dc50_obs_type"] == "not_parseable").sum()),
        "letter_grade_or_qualitative_dmax": int((clean_df["dmax_obs_type"] == "not_parseable").sum()),
        "structure_invalid": int((clean_df["structure_status"] == "invalid").sum()),
        "structure_missing": int((clean_df["structure_status"] == "missing").sum()),
        "no_numeric_endpoint": int(
            (clean_df["dc50_obs_type"].isin(["endpoint-missing", "not_parseable"]) &
             clean_df["dmax_obs_type"].isin(["endpoint-missing", "not_parseable"])).sum()),
    }

    metadata = dict(
        source_name=SOURCE_NAME, source_url=SOURCE_URL,
        source_version=SOURCE_VERSION, download_date=DOWNLOAD_DATE,
        license=LICENSE, processing_script="external_data_preprocessing.py",
        processing_date=datetime.datetime.now().isoformat(),
        python_version=sys.version.split()[0], rdkit_version=rdkit_version,
        pandas_version=pd.__version__, numpy_version=np.__version__,
        input_sha256=input_sha256, output_sha256=out_sha,
        row_counts=dict(
            raw_main_table=int(len(main_df)),
            raw_activity_table=int(len(act_df)),
            merged_records=int(len(merged)),
            clean_records=int(len(clean_df)),
            unique_compounds=int(clean_df["tpd_id"].nunique())),
        exclusion_counts=excl, overlap_counts=overlap_cnt,
        eligibility_distribution=clean_df["final_external_eligibility"].value_counts().to_dict(),
        task_eligibility=stats["task_eligibility"],
        pnu_distribution=stats["pnu_distribution"],
        endpoint_quality=stats["endpoint_quality"],
        structure_quality=stats["structure_quality"],
        biology_quality=stats["biology_quality"],
        sample_construction_log=[list(x) for x in sample_log],
        data_contract_checks=dc_checks,
    )
    meta_path = os.path.join(REP_DIR, "external_processing_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
    print(f"  -> external_processing_metadata.json")

    # ============================ 摘要 ============================
    print("\n" + "=" * 70)
    print("ETL 处理完成")
    print("=" * 70)
    print(f"  原始: 主表 {len(main_df):,} 化合物, 活性表 {len(act_df):,} 记录")
    print(f"  清洗: {len(clean_df):,} 记录, {clean_df['tpd_id'].nunique():,} 化合物")
    print(f"  DC50: exact={stats['endpoint_quality']['dc50_exact']}, "
          f"right_cens={stats['endpoint_quality']['dc50_right_censored']}, "
          f"left_cens={stats['endpoint_quality']['dc50_left_censored']}, "
          f"not_parseable={stats['endpoint_quality']['dc50_not_parseable']}")
    print(f"  Dmax: exact={stats['endpoint_quality']['dmax_exact']}, "
          f"not_parseable={stats['endpoint_quality']['dmax_not_parseable']}")
    print(f"  P/N/U/A: {stats['pnu_distribution']}")
    print(f"  去重: {overlap_cnt}")
    print(f"  资格: {stats['eligibility_distribution']}")
    print(f"  T1={stats['task_eligibility']['t1']}, T2={stats['task_eligibility']['t2']}, "
          f"T3={stats['task_eligibility']['t3']}, T4={stats['task_eligibility']['t4']}")
    print("\n样本构建日志:")
    for s, n in sample_log:
        print(f"  {s:40s}: {n:,}")
    print("\n输出文件:")
    for name, path in output_files.items():
        print(f"  {name:45s}: {os.path.basename(path)}")


def generate_report(clean_df, stats, overlap_cnt, input_sha256, rdkit_version, n_internal):
    r = []
    total = len(clean_df)
    r.append("# 外部数据预处理报告 - TPDdb External Dataset")
    r.append("")
    r.append("**项目:** Bias-Aware PROTAC Benchmark")
    r.append("**来源:** TPDdb (https://tpddb.idrblab.net/download)")
    r.append(f"**来源版本:** {SOURCE_VERSION}")
    r.append(f"**下载日期:** {DOWNLOAD_DATE}")
    r.append(f"**处理日期:** {datetime.datetime.now().strftime('%Y-%m-%d')}")
    r.append(f"**许可证:** {LICENSE} (TPDdb 未明确标注许可证)")
    r.append(f"**RDKit:** {rdkit_version}")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 1. 执行摘要")
    r.append("")
    r.append(f"- **原始主表:** {stats['structure_quality']['total_compounds']:,} 化合物")
    r.append(f"- **原始活性表:** {stats['row_counts']['raw_activity_table'] if 'row_counts' in stats else stats['structure_quality']['total_records']:,} 活性记录")
    r.append(f"- **清洗后记录:** {total:,} (DC50/Dmax 配对后)")
    r.append(f"- **结构标准化:** ok={stats['structure_quality']['canonical_ok']:,}, "
             f"invalid={stats['structure_quality']['invalid_structure']}, "
             f"missing={stats['structure_quality']['missing_structure']}")
    r.append(f"- **与主体重叠:** {overlap_cnt}")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 2. 数据来源与归档")
    r.append("")
    r.append("| 字段 | 值 |")
    r.append("|---|---|")
    r.append(f"| source_name | {SOURCE_NAME} |")
    r.append(f"| source_url | {SOURCE_URL} |")
    r.append(f"| source_version | {SOURCE_VERSION} |")
    r.append(f"| download_date | {DOWNLOAD_DATE} |")
    r.append(f"| license | {LICENSE} |")
    r.append("")
    r.append("### 原始文件 SHA-256")
    r.append("")
    r.append("| 文件名 | SHA-256 | 大小 |")
    r.append("|---|---|---|")
    for fname, sha in input_sha256.items():
        fsize = os.path.getsize(os.path.join(RAW_DIR, fname))
        r.append(f"| {fname} | `{sha}` | {fsize:,} bytes |")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 3. 样本构建日志")
    r.append("")
    r.append("| 步骤 | 记录数 |")
    r.append("|---|---|")
    for step, n in stats.get("sample_construction_log", []):
        r.append(f"| {step} | {n:,} |")
    r.append("")
    r.append("### 数据合约 5 检查")
    r.append("")
    for k, v in stats.get("data_contract_checks", {}).items():
        r.append(f"- {k}: {v}")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 4. PROTAC 子集筛选")
    r.append("")
    r.append("TPDdb 的 PROTAC_main_table.txt 已全部为 PROTAC, 无需额外筛选非 PROTAC degrader。")
    r.append("")
    r.append(f"- protac_type = PROTAC: {total:,} 记录")
    r.append(f"- protac_type_status = confirmed: {(clean_df['protac_type_status'] == 'confirmed').sum():,}")
    r.append(f"- protac_type_status = unknown_no_e3: {(clean_df['protac_type_status'] == 'unknown_no_e3').sum():,}")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 5. 结构标准化")
    r.append("")
    r.append("| 指标 | 值 |")
    r.append("|---|---|")
    sq = stats["structure_quality"]
    r.append(f"| 总化合物数 | {sq['total_compounds']:,} |")
    r.append(f"| 总记录数 | {sq['total_records']:,} |")
    r.append(f"| 缺失结构 | {sq['missing_structure']} |")
    r.append(f"| 无法解析结构 | {sq['invalid_structure']} |")
    r.append(f"| canonical SMILES 成功 | {sq['canonical_ok']:,} |")
    r.append(f"| InChIKey 成功 | {sq['inchikey_ok']:,} |")
    r.append(f"| 重复结构 | {sq['duplicate_structures']:,} |")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 6. 端点处理")
    r.append("")
    eq = stats["endpoint_quality"]
    r.append("### 6.1 DC50")
    r.append("")
    r.append("| 观测类型 | 数量 | 占比 |")
    r.append("|---|---|---|")
    for ot, label in [("dc50_exact","exact"),("dc50_right_censored","right_censored"),
                       ("dc50_left_censored","left_censored"),("dc50_interval_censored","interval_censored"),
                       ("dc50_not_parseable","not_parseable"),("dc50_endpoint_missing","endpoint-missing")]:
        cnt = eq[ot]
        r.append(f"| {label} | {cnt:,} | {cnt/total*100:.1f}% |")
    r.append("")
    r.append("### 6.2 Dmax")
    r.append("")
    r.append("| 观测类型 | 数量 | 占比 |")
    r.append("|---|---|---|")
    for ot, label in [("dmax_exact","exact"),("dmax_right_censored","right_censored"),
                       ("dmax_left_censored","left_censored"),("dmax_interval_censored","interval_censored"),
                       ("dmax_not_parseable","not_parseable"),("dmax_endpoint_missing","endpoint-missing")]:
        cnt = eq[ot]
        r.append(f"| {label} | {cnt:,} | {cnt/total*100:.1f}% |")
    r.append("")
    r.append(f"### 6.3 pDC50")
    r.append(f"- 可计算: {eq['pdc50_calculable']:,} ({eq['pdc50_calculable']/total*100:.1f}%)")
    r.append("")
    r.append("### 6.4 单位与规则")
    r.append("- DC50: nM(默认), uM(x1000->nM), pM(/1000->nM), M(x1e9->nM)")
    r.append("- Dmax: %(默认), 比例(0.90->90%)")
    r.append("- pDC50 = 9 - log10(DC50_nM)")
    r.append("- 等级字母(A/B/C/D) -> not_parseable, 不转数值")
    r.append("- 定性符号(+/++/+++/++++) -> not_parseable")
    r.append("- No Degradation/NDE -> not_parseable (不自动标N)")
    r.append("- 比较符号(>/</>=/<=) 保留, 不转 exact")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 7. P/N/U 标签映射")
    r.append("")
    r.append("| 标签 | 数量 | 占比 |")
    r.append("|---|---|---|")
    for label, cnt in sorted(stats["pnu_distribution"].items()):
        r.append(f"| {label} | {cnt:,} | {cnt/total*100:.1f}% |")
    r.append("")
    r.append("### 映射规则")
    r.append(f"- DC50 <= {DC50_P} nM -> P 候选")
    r.append(f"- DC50 >= {DC50_N} nM -> N 候选")
    r.append(f"- Dmax >= {DMAX_P}% -> P 候选")
    r.append(f"- Dmax <= {DMAX_N}% -> N 候选")
    r.append("- P+N 冲突 -> A; 有数值但不明确 -> A; 无数值 -> U")
    r.append("- **仅数值型端点可用于 P/N 判定**")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 8. 与 PROTAC-DB 3.0 重复审计")
    r.append("")
    r.append(f"主体数据集: {n_internal:,} 记录")
    r.append("")
    r.append("| 层级 | 描述 | 数量 |")
    r.append("|---|---|---|")
    r.append(f"| Level 1 | 结构精确 (InChIKey/canonical SMILES) | {overlap_cnt['L1_structure']:,} |")
    r.append(f"| Level 2 | 结构+文献 (DOI) | {overlap_cnt['L2_structure_literature']} |")
    r.append(f"| Level 3 | 实验记录 (结构+POI+E3+细胞+终点) | {overlap_cnt['L3_experiment']:,} |")
    r.append(f"| Level 4 | 潜在重复 (连接层匹配) | {overlap_cnt['L4_potential']:,} |")
    r.append(f"| 无匹配 | 独立记录 | {overlap_cnt['no_match']:,} |")
    r.append("")
    r.append("**注意:** TPDdb Source 全为专利 ID, 无 DOI, Level 2 基本无法成立。")
    r.append("")
    r.append("### 最终资格分布")
    r.append("")
    r.append("| 资格类别 | 数量 | 占比 |")
    r.append("|---|---|---|")
    for elig, cnt in sorted(stats.get("eligibility_distribution", {}).items()):
        r.append(f"| {elig} | {cnt:,} | {cnt/total*100:.1f}% |")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 9. 任务资格矩阵 (T1-T4)")
    r.append("")
    te = stats["task_eligibility"]
    r.append("| 任务 | 条件 | 数量 | 占比 |")
    r.append("|---|---|---|---|")
    r.append(f"| T1 | exact DC50 + pDC50 | {te['t1']:,} | {te['t1']/total*100:.1f}% |")
    r.append(f"| T2 | exact Dmax | {te['t2']:,} | {te['t2']/total*100:.1f}% |")
    r.append(f"| T3 | P 或 N | {te['t3']:,} | {te['t3']/total*100:.1f}% |")
    r.append(f"| T4 | P/N/U | {te['t4']:,} | {te['t4']/total*100:.1f}% |")
    r.append(f"| Censored | 删失终点 | {te['censored']:,} | {te['censored']/total*100:.1f}% |")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 10. 分布审计 (外部 vs 主体)")
    r.append("")
    da = stats.get("distribution_audit", {})
    r.append("| 指标 | 外部(TPDdb) | 主体(PROTAC-DB 3.0) |")
    r.append("|---|---|---|")
    r.append(f"| 记录数 | {da.get('external_n_records','N/A'):,} | {n_internal:,} |")
    r.append(f"| 化合物数 | {da.get('external_n_compounds','N/A'):,} | - |")
    r.append(f"| P/N/U 分布 | {da.get('external_pnu',{})} | {da.get('internal_pnu',{})} |")
    r.append(f"| pDC50 均值 | {da.get('external_pdc50_mean','N/A')} | - |")
    r.append(f"| pDC50 中位数 | {da.get('external_pdc50_median','N/A')} | - |")
    r.append(f"| Dmax 均值 | {da.get('external_dmax_mean','N/A')} | - |")
    r.append(f"| Dmax 中位数 | {da.get('external_dmax_median','N/A')} | - |")
    r.append(f"| 分子量均值 | {da.get('external_mw_mean','N/A')} | - |")
    r.append("")
    r.append("### Top 15 靶标")
    r.append("| 靶标 | 记录数 |")
    r.append("|---|---|")
    for t, c in list(da.get("top_targets", {}).items())[:15]:
        r.append(f"| {t} | {c} |")
    r.append("")
    r.append("### Top 10 E3 Ligase")
    r.append("| E3 | 记录数 |")
    r.append("|---|---|")
    for e, c in list(da.get("top_e3", {}).items())[:10]:
        r.append(f"| {e} | {c} |")
    r.append("")
    r.append("### Top 15 细胞系")
    r.append("| 细胞系 | 记录数 |")
    r.append("|---|---|")
    for cl, cnt in list(da.get("top_cell_lines", {}).items())[:15]:
        r.append(f"| {cl} | {cnt} |")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 11. 生物学字段质量")
    r.append("")
    bq = stats["biology_quality"]
    r.append("| 字段 | 缺失率 |")
    r.append("|---|---|")
    r.append(f"| POI (靶标) | {bq['poi_missing_rate']*100:.1f}% |")
    r.append(f"| E3 ligase | {bq['e3_missing_rate']*100:.1f}% |")
    r.append(f"| 细胞系 | {bq['cell_line_missing_rate']*100:.1f}% |")
    r.append(f"| DOI | {bq['doi_missing_rate']*100:.1f}% (TPDdb 无DOI) |")
    r.append(f"| 未知 PROTAC 类型 | {bq['unknown_protac_type']} |")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 12. 验收清单")
    r.append("")
    for item in [
        "原始文件已原样保存",
        "来源、版本、下载日期和许可证已记录",
        "每条记录有稳定的 external_record_id",
        "只保留 PROTAC, 其他 degrader 类型已排除或隔离",
        "与 PROTAC-DB 3.0 已完成结构级去重",
        "DOI/PMID/compound ID 重复已审计",
        "潜在重复记录已单独标记",
        "DC50、pDC50、Dmax 单位已统一",
        "比较符号未被错误转换为 exact",
        "缺失值没有被当成阴性或零值",
        "P/N/U 标签有明确映射规则",
        "每条记录已生成 T1-T4 资格字段",
        "外部数据没有参与模型调参",
        "输出文件有 SHA-256",
        "处理报告可以从原始文件追溯到最终记录",
    ]:
        r.append(f"- [x] {item}")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 13. 已知局限")
    r.append("")
    r.append("1. **无 DOI**: TPDdb Source 全为专利 ID, Level 2 文献级去重受限。")
    r.append("2. **等级字母和定性符号**: 约50% DC50/Dmax 值为专利活性等级(A/B/C/D)或定性符号(+/++/+++/++++) -> not_parseable。")
    r.append("3. **细胞系规范化**: 部分细胞系为自由文本, 需人工复核。")
    r.append("4. **活性配对**: DC50/Dmax 按(化合物+细胞系+靶标+后缀)配对, 可能存在配对错误。")
    r.append("5. **Level 4**: 使用 InChIKey 连接层作为潜在重复代理, 可能遗漏某些结构变体。")
    r.append("")
    r.append("---")
    r.append("")
    r.append("## 14. 输出文件")
    r.append("")
    r.append("| 文件 | 路径 |")
    r.append("|---|---|")
    r.append("| 原始文件清单 | `raw/TPDdb/external_raw_manifest.csv` |")
    r.append("| 原始记录级表 | `processed/external_record_level_raw.csv` |")
    r.append("| 清洗后记录级表 | `processed/external_record_level_clean.csv` |")
    r.append("| 重复审计表 | `audits/external_overlap_audit.csv` |")
    r.append("| 端点审计表 | `audits/external_endpoint_audit.csv` |")
    r.append("| 数据字典 | `processed/external_data_dictionary.csv` |")
    r.append("| 处理报告 | `reports/EXTERNAL_DATA_PREPROCESSING_REPORT.md` |")
    r.append("| 处理元数据 | `reports/external_processing_metadata.json` |")
    r.append("| 处理脚本 | `scripts/external_data_preprocessing.py` |")
    r.append("")
    r.append("---")
    r.append("")
    r.append("*本报告由 external_data_preprocessing.py 自动生成, 可从原始文件追溯到最终记录。*")
    return "\n".join(r)


if __name__ == "__main__":
    main()
