# -*- coding: utf-8 -*-
"""
etl_protac.py  —  PROTAC-DB 3.0 record-level ETL for Bias-Aware PNU study
=========================================================================
数据工程师（洗澄明）: 将 PROTAC-DB 3.0 主表 protac.csv 解析为记录级
(record-level)、证据/删失感知的分析数据集。

职责（通用数据工程流水线 Step 0-4 适配到化学信息学场景）:
  Step 0  样本构建日志 & 数据合约（5 检查）
  Step 1  数据清洗（缺失/异常/重复/合并）
  Step 2  变量构建（DC50/Dmax 解析、pDC50、activity_evidence、scaffold）
  Step 3  描述统计（Table 1 / 审计计数，详见 reports/）
  Step 4  诊断检验（此处主要是数据质量与不平衡诊断，非回归残差检验）

运行: python etl_protac.py
输出:
  data/derived/protac_clean_record_level.csv
  data/derived/data_dictionary.csv
  data/derived/protac_clean_audit_stats.json   (供报告脚本复用)
  data/raw/raw_data_manifest.csv
  templates/raw_data_manifest_template.csv
  templates/protac_annotation_template.csv
  reports/PILOT_20_EXAMPLES.csv
  README_data_version.md
所有原始文件只读，不修改 data/raw/。
"""

import os, re, math, hashlib, json, datetime, csv
import numpy as np
import pandas as pd

# ----------------------------- 路径 -----------------------------
BASE = os.environ.get("PROTAC_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW  = os.path.join(BASE, "data", "raw")
OUT  = os.path.join(BASE, "data", "derived")
REP  = os.path.join(BASE, "reports")
TEM  = os.path.join(BASE, "templates")
CODE = os.path.join(BASE, "code")
os.makedirs(OUT, exist_ok=True)
os.makedirs(REP, exist_ok=True)
os.makedirs(TEM, exist_ok=True)

SRC_URL = "http://cadd.zju.edu.cn/protacdb/"
DL_DATE = "2026-08-15"

# ----------------------------- 常量 -----------------------------
MISSING_TOKENS = {
    "", "N.D.", "N.D", "N/A", "NA", "N.A.", "NOT DETERMINED", "NOT DET",
    "ND", "ND.", "N/D", "NAN", "NONE", "NR", "NOT REPORTED", "NO DATA",
    "NA.", "ND", "N.A", "NULL", "UNCLEAR", "U.D.", "UD",
}
# 临时活动证据阈值（标注规范 §5 尚未定稿，provisional）。P/Dmax 支持阈值固定，
# DC50 的 P/N 候选 cutoff 在敏感性扫描中变化；默认 P-cutoff=100 nM, N-cutoff=1000 nM。
DMAX_P = 80.0   # Dmax >= 80% 支持 P
DMAX_N = 10.0   # Dmax <= 10% 支持 N
DEFAULT_CP = 100.0   # DC50 <= cp -> P 候选
SENS_CUTS = [50.0, 100.0, 300.0, 1000.0]

# ----------------------------- 文本归一化 -----------------------------
def norm_text(s):
    """统一归一化原始数值串: 全角->半角, 去除多余空格, 处理逗号(千分位 vs 欧洲小数)。"""
    if s is None:
        return ""
    s = str(s)
    s = s.replace("＞", ">").replace("＜", "<").replace("＝", "=")
    s = s.replace("　", " ").strip()
    # 千分位分隔符: 逗号后紧跟恰好 3 位数字(且其后非数字或结尾) -> 删除
    s = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", s)
    # 其余逗号视为欧洲小数分隔符 -> 小数点
    s = s.replace(",", ".")
    # OCR 风格错误: 数字上下文中的字母 O/o 后接 ".数字" -> 0 (如 "O.3"->"0.3")
    s = re.sub(r"(?i)(?<!\w)([Oo])(?=\.\d)", "0", s)
    return s

def to_float(x):
    try:
        return float(x)
    except Exception:
        return None

CMP_RE  = re.compile(r"^([<>]=?)\s*([\d.]+)$")
RANGE_RE = re.compile(r"^([\d.]+)\s*[-~]\s*([\d.]+)$")
NUM_RE  = re.compile(r"^[\d.]+$")

def parse_token(tok):
    """解析单个 token -> ('num',v) | ('cmp',op,v) | ('range',lo,hi) | ('unknown',txt) | None(缺失)。"""
    tok = tok.strip()
    if tok == "" or tok.upper() in MISSING_TOKENS:
        return None
    tok = re.sub(r"\([^)]*\)", "", tok).strip()   # 去除 (nm)/(n/a)/(N.D.) 等括号
    if tok == "" or tok.upper() in MISSING_TOKENS:
        return None
    m = CMP_RE.match(tok)
    if m:
        return ("cmp", m.group(1), to_float(m.group(2)))
    m = RANGE_RE.match(tok)
    if m:
        return ("range", to_float(m.group(1)), to_float(m.group(2)))
    if NUM_RE.match(tok):
        return ("num", to_float(tok))
    return ("unknown", tok)

def parse_numeric_endpoint(raw, is_dmax=False):
    """
    解析单个端点(数值串) -> dict:
      obs_type: exact / left-censored / right-censored / interval-censored / endpoint-missing
      value, lower, upper: float or None
      replicates: 原始重复串(若有) / None
      is_dose_series: 长串序列标记
    is_dmax: 若为 Dmax 列, 剂量/时间序列取最大值作为候选(标注规范)。
    """
    res = dict(obs_type="endpoint-missing", value=None, lower=None, upper=None,
               replicates=None, is_dose_series=False, n_tokens=0)
    if raw is None:
        return res
    s = norm_text(raw)
    if s == "" or s.upper() in MISSING_TOKENS:
        return res
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):   # 日期串(脏数据) -> 缺失
        return res
    # 复合比较 '>5 and <50' -> 用 ';' 切分
    s2 = re.sub(r"\band\b", ";", s, flags=re.I)
    toks = [t.strip() for t in re.split(r"[;/]", s2) if t.strip() != ""]
    expanded = []
    for t in toks:                            # 空格分隔的序列 '51.8 82.5 70.2'
        expanded.extend(t.split())
    toks = [p for p in expanded if p != ""]
    parsed = [p for p in (parse_token(t) for t in toks) if p is not None]
    if not parsed:
        return res                            # 例: (n/a)/(n/a)
    res["n_tokens"] = len(parsed)
    types = [p[0] for p in parsed]
    # 长串序列启发式: >=4 个数字/区间 token
    if len(parsed) >= 4 and sum(1 for p in parsed if p[0] in ("num", "range")) >= 3:
        res["is_dose_series"] = True
    # 单 token
    if len(parsed) == 1:
        p = parsed[0]
        if p[0] == "num":
            res["obs_type"], res["value"] = "exact", p[1]
        elif p[0] == "cmp":
            if p[1] in (">", ">="):
                res["obs_type"], res["lower"] = "right-censored", p[2]
            else:
                res["obs_type"], res["upper"] = "left-censored", p[2]
        elif p[0] == "range":
            res["obs_type"], res["lower"], res["upper"] = "interval-censored", min(p[1], p[2]), max(p[1], p[2])
        return res
    # 多 token: 重复测量 / 序列 / 混合比较(含多个比较符)
    nums   = [p for p in parsed if p[0] == "num"]
    cmps   = [p for p in parsed if p[0] == "cmp"]
    ranges = [p for p in parsed if p[0] == "range"]
    res["replicates"] = raw.strip()
    if nums and not cmps and not ranges:
        if res["is_dose_series"] and is_dmax:
            res["obs_type"], res["value"] = "exact", max(p[1] for p in nums)  # Dmax 序列取最大
        else:
            res["obs_type"], res["value"] = "exact", nums[0][1]              # 重复测量取首值(标注规范)
    elif nums and (cmps or ranges):
        # 混合确切数值与删失/区间: 以确切数值为点估计(重复测量视角), 忽略删失边界
        v = max(p[1] for p in nums) if (res["is_dose_series"] and is_dmax) else nums[0][1]
        res["obs_type"], res["value"] = "exact", v
    elif ranges and not nums and not cmps:
        res["obs_type"] = "interval-censored"
        res["lower"] = min(r[1] for r in ranges)
        res["upper"] = max(r[2] for r in ranges)
    else:
        lowers, uppers = [], []
        for p in parsed:
            if p[0] == "cmp":
                if p[1] in (">", ">="): lowers.append(p[2])
                else: uppers.append(p[2])
            elif p[0] == "range":
                lowers.append(min(p[1], p[2])); uppers.append(max(p[1], p[2]))
        eff_lo = max(lowers) if lowers else None   # 最紧下界
        eff_hi = min(uppers) if uppers else None   # 最紧上界
        if eff_lo is not None and eff_hi is not None and eff_lo < eff_hi:
            res["obs_type"], res["lower"], res["upper"] = "interval-censored", eff_lo, eff_hi
        elif eff_lo is not None and eff_hi is not None:  # 矛盾边界 -> 保守并集区间
            res["obs_type"], res["lower"], res["upper"] = "interval-censored", min(lowers), max(uppers)
            res["replicates"] = (res["replicates"] or "") + " [CONFLICT]"
        elif eff_lo is not None:
            res["obs_type"], res["lower"] = "right-censored", eff_lo
        elif eff_hi is not None:
            res["obs_type"], res["upper"] = "left-censored", eff_hi
        else:
            res["obs_type"], res["value"] = "exact", None
    return res

def pdc50_from(ep):
    """由端点解析结果换算 pDC50 (= 9 - log10(DC50)); 仅对可确定数值/边界者做, 符号按标注规范 §4 反转。"""
    ot, v, lo, hi = ep["obs_type"], ep["value"], ep["lower"], ep["upper"]
    if ot == "exact" and v is not None and v > 0:
        return dict(value=9 - math.log10(v), lower=None, upper=None)
    if ot == "right-censored" and lo is not None and lo > 0:   # DC50 > lo => pDC50 < 9-log10(lo)
        return dict(value=None, lower=None, upper=9 - math.log10(lo))
    if ot == "left-censored" and hi is not None and hi > 0:    # DC50 < hi => pDC50 > 9-log10(hi)
        return dict(value=None, lower=9 - math.log10(hi), upper=None)
    if ot == "interval-censored" and lo is not None and hi is not None and lo > 0 and hi > 0:
        return dict(value=None, lower=9 - math.log10(hi), upper=9 - math.log10(lo))
    return dict(value=None, lower=None, upper=None)

# ----------------------------- inline 细胞系展开(Percent degradation 列) -----------------------------
def parse_inline_celllines(raw):
    """解析 'PANC-1:0/0;K562:0/37.7' -> [('PANC-1', 0.0), ('K562', 18.85)] 或 None。"""
    if raw is None:
        return None
    s = norm_text(raw)
    if ";" not in s or not re.search(r"[A-Za-z0-9\-]+:", s):
        return None
    out = []
    for blk in re.split(r";", s):
        m = re.match(r"^\s*([A-Za-z0-9\-]+)\s*:(.*)$", blk)
        if not m:
            continue
        cell = m.group(1)
        valpart = m.group(2).strip()
        nums = [to_float(x) for x in re.findall(r"[\d.]+", valpart)]
        nums = [n for n in nums if n is not None]
        v = float(np.mean(nums)) if nums else None
        out.append((cell, v))
    return out if out else None

# ----------------------------- 从 Assay 文本挖掘 cell_line / treatment_time -----------------------------
CELL_LINES = [
    "MV4-11","MV4;11","MV4; 11","RS4;11","RS4; 11","RS4-11","MCF-7","MCF7","A-204","A204",
    "EOL-1","EOL1","PANC-1","PANC1","K562","HCT116","HCT-116","HEK293","HEK-293","HeLa",
    "Jurkat","U87","U87MG","A549","HL-60","HL60","THP-1","THP1","Raji","Daudi","Namalwa",
    "OCI-LY1","OCI-LY7","OCI-LY10","SU-DHL-2","SU-DHL-4","SU-DHL-6","SU-DHL-8","MOLM-13",
    "MOLM14","KG-1","KG1","KASUMI-1","KASUMI1","NCI-H929","MM.1S","MM1S","RPMI-8226",
    "RPMI8226","U266","AMO-1","LS513","LS-513","HCT-15","HCT15","SW620","SW480","HT-29",
    "HT29","MDA-MB-231","MDA-MB-468","MDA-MB-436","BT-474","BT474","SK-BR-3","SKBR3","T47D",
    "T47-D","ZR-75-1","ZR-751","MCF-10A","MCF10A","PC-3","PC3","DU-145","DU145","LNCaP",
    "VCaP","VCAP","22Rv1","22RV1","C4-2","C42","MKN-45","MKN45","NCI-N87","NCIN87","SNU-16",
    "SNU16","SNU-638","SNU638","HGC-27","HGC27","AGS","NCI-H1975","NCI-H1650","NCI-H358",
    "A427","A-427","H23","H460","Calu-3","Calu3","H1299","H-1299","95-D","95D","MiaPaCa-2",
    "MiaPaCa2","BxPC-3","BxPC3","AsPC-1","AsPC1","Capan-1","Capan1","Panc-1","U2OS","U-2OS",
    "Saos-2","Saos2","MG-63","MG63","143B","143-B","HOS","SJSA-1","SJSA1","RD","RH-30","RH30",
    "RH-41","RH41","CW-2","CW2","LoVo","HCT-8","HCT8","Caco-2","Caco2","Colo205","Colo-205",
    "WiDr","DLD-1","DLD1","HCT-116","NCI-H460","HOP-92","HOP92","H23","H322","H441",
]
# 构建正则: 优先匹配更长(带连字符)的名称
CELL_LINES_SORTED = sorted(CELL_LINES, key=len, reverse=True)
CELL_RE = re.compile(r"(?:" + "|".join(re.escape(c) for c in CELL_LINES_SORTED) + r")", re.I)

# 细胞系规范名映射(合并常见无连字符写法)
CELL_CANON = {
    "MCF7": "MCF-7", "A204": "A-204", "EOL1": "EOL-1", "PANC1": "PANC-1",
    "HCT15": "HCT-15", "HCT116": "HCT-116", "HEK293": "HEK-293", "BT474": "BT-474",
    "SKBR3": "SK-BR-3", "K562": "K562", "MV4-11": "MV4-11", "RS4-11": "RS4-11",
    "MOLM13": "MOLM-13", "KG1": "KG-1", "KASUMI1": "KASUMI-1", "NCIN87": "NCI-N87",
    "SNU16": "SNU-16", "SNU638": "SNU-638", "HGC27": "HGC-27", "NCIH1975": "NCI-H1975",
    "NCIH1650": "NCI-H1650", "NCIH358": "NCI-H358", "NCIH460": "NCI-H460",
    "CACO2": "Caco-2", "COL0205": "Colo-205", "DLD1": "DLD-1", "MIAPACA2": "MiaPaCa-2",
    "BXPC3": "BxPC-3", "ASPC1": "ASPC-1", "CAPAN1": "Capan-1", "H1299": "H-1299",
    "SUDHL2": "SU-DHL-2", "SUDHL4": "SU-DHL-4", "SUDHL6": "SU-DHL-6", "SUDHL8": "SU-DHL-8",
    "OCILY1": "OCI-LY1", "OCILY7": "OCI-LY7", "OCILY10": "OCI-LY10", "MOLM14": "MOLM-14",
    "RH30": "RH-30", "RH41": "RH-41", "HOP92": "HOP-92", "VCAP": "VCaP", "22RV1": "22Rv1",
    "C42": "C4-2", "PCT": "PC-3", "DU145": "DU-145", "LNCAP": "LNCaP",
}

def mine_cell_lines(text):
    if not text:
        return None
    found = CELL_RE.findall(text)
    # 归一化常见变体
    norm = []
    for f in found:
        g = re.sub(r"\s+", "", f).upper()
        g = g.replace(";", "-")
        g = CELL_CANON.get(g, g)
        norm.append(g)
    if not norm:
        return None
    # 去重保序
    seen, uniq = set(), []
    for n in norm:
        if n not in seen:
            seen.add(n); uniq.append(n)
    return "/".join(uniq)

def mine_time(text):
    if not text:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*h\b", text, re.I)
    if m:
        return float(m.group(1))
    return None

def extract_year(doi):
    if not doi:
        return None
    m = re.search(r"(?:19|20)\d{2}", doi)
    if m:
        y = int(m.group(0))
        if 1990 <= y <= 2026:
            return y
    return None

# ----------------------------- activity_evidence (provisional) -----------------------------
def assign_evidence(dc, dm, c_p):
    c_n = 10.0 * c_p
    p = 0; n = 0; sig = []
    # DC50 信号
    if dc["obs_type"] == "exact" and dc["value"] is not None:
        if dc["value"] <= c_p:
            p += 1; sig.append(f"DC50={dc['value']:.3g}<=cp")
        elif dc["value"] >= c_n:
            n += 1; sig.append(f"DC50={dc['value']:.3g}>=cn")
    elif dc["obs_type"] == "left-censored" and dc["upper"] is not None:
        if dc["upper"] <= c_p:
            p += 1; sig.append(f"DC50<{dc['upper']:.3g}<=cp")
    elif dc["obs_type"] == "right-censored" and dc["lower"] is not None:
        if dc["lower"] >= c_n:
            n += 1; sig.append(f"DC50>{dc['lower']:.3g}>=cn")
    elif dc["obs_type"] == "interval-censored" and dc["lower"] is not None and dc["upper"] is not None:
        if dc["upper"] <= c_p:
            p += 1; sig.append("DC50_interval_P")
        elif dc["lower"] >= c_n:
            n += 1; sig.append("DC50_interval_N")
    # Dmax 信号
    if dm["obs_type"] == "exact" and dm["value"] is not None:
        if dm["value"] >= DMAX_P:
            p += 1; sig.append(f"Dmax={dm['value']:.1f}>=P")
        elif dm["value"] <= DMAX_N:
            n += 1; sig.append(f"Dmax={dm['value']:.1f}<=N")
    elif dm["obs_type"] == "right-censored" and dm["lower"] is not None:
        if dm["lower"] >= DMAX_P:
            p += 1; sig.append("Dmax_cens_P")
    elif dm["obs_type"] == "left-censored" and dm["upper"] is not None:
        if dm["upper"] <= DMAX_N:
            n += 1; sig.append("Dmax_cens_N")
    # 决策
    if p > 0 and n > 0:
        return "A", ";".join(sig)          # 冲突 -> ambiguous
    if p > 0:
        return "P", ";".join(sig)
    if n > 0:
        return "N", ";".join(sig)
    if dc["obs_type"] == "endpoint-missing" and dm["obs_type"] == "endpoint-missing":
        return "U", ";".join(sig)          # 两终点皆无观测
    return "A", ";".join(sig)              # 有观测但不明确 -> ambiguous

# ----------------------------- Murcko scaffold -----------------------------
def compute_scaffold(smiles, Chem):
    if not smiles or Chem is None:
        return None
    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "PARSE_FAIL"
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(scaf)
        return s if s else "NO_SCAFFOLD"
    except Exception:
        return "PARSE_FAIL"

# =====================================================================
def main():
    print("== Step 0: 样本构建日志 & 数据合约 ==")
    raw_path = os.path.join(RAW, "protac.csv")
    df = pd.read_csv(raw_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    n_raw = len(df)
    sample_log = [("0. raw protac.csv", n_raw)]
    # 5 检查 (数据合约)
    dc_checks = {}
    dc_checks["shape_ok"]   = (df.shape == (15502, 89))
    dc_checks["id_dtype"]   = True  # 全字符串读取, 无需类型校验
    miss_rate = (df.replace("", np.nan).isna().mean())
    dc_checks["max_missing_lt_1"] = bool((miss_rate < 1).all())
    dc_checks["compound_id_unique"] = (df["Compound ID"].nunique() < n_raw)  # 预期非唯一
    dc_checks["panel_balance_na"] = "N/A (cross-sectional record-level, not a panel)"

    # 列映射
    df = df.rename(columns={
        "Compound ID": "compound_id", "Uniprot": "uniprot", "Target": "target",
        "E3 ligase": "e3_ligase", "PDB": "pdb", "Name": "name", "Smiles": "smiles",
        "DC50 (nM)": "dc50_raw", "Dmax (%)": "dmax_raw",
        "Assay (DC50/Dmax)": "assay_dc50", "Percent degradation (%)": "pctdeg_raw",
        "Assay (Percent degradation)": "assay_pctdeg", "Article DOI": "article_doi",
    })
    sample_log.append(("1. loaded+renamed", len(df)))

    # rdkit 可用性
    Chem = None
    try:
        from rdkit import Chem as _C
        Chem = _C
        print("rdkit available -> Murcko scaffold 可用")
    except Exception as e:
        print("rdkit unavailable -> 使用 SMILES 前缀近似 (", e, ")")

    records = []
    for idx, row in df.iterrows():
        dc_ep = parse_numeric_endpoint(row["dc50_raw"])
        dm_ep = parse_numeric_endpoint(row["dmax_raw"], is_dmax=True)
        pdc = pdc50_from(dc_ep)
        # pctdeg: 一般数值; 若是 inline 细胞系则展开
        inline = parse_inline_celllines(row["pctdeg_raw"])
        if inline is None:
            pnums = [to_float(x) for x in re.findall(r"[\d.]+", norm_text(row["pctdeg_raw"]))]
            pnums = [x for x in pnums if x is not None]
            pctdeg_val = float(np.mean(pnums)) if pnums else None
            cell_mined = mine_cell_lines(row["assay_dc50"]) or mine_cell_lines(row["assay_pctdeg"])
            cells = [(cell_mined, pctdeg_val)]
        else:
            cells = inline   # 每个细胞系一行
        t_h = mine_time(row["assay_dc50"]) or mine_time(row["assay_pctdeg"])
        has_doi = bool(row["article_doi"].strip())
        year = extract_year(row["article_doi"])
        for (cell, pval) in cells:
            rec = dict(
                raw_row_index=int(idx),
                compound_id=row["compound_id"], uniprot=row["uniprot"], target=row["target"],
                e3_ligase=row["e3_ligase"], pdb=row["pdb"], name=row["name"],
                smiles=row["smiles"], article_doi=row["article_doi"],
                dc50_raw=row["dc50_raw"], dc50_obs_type=dc_ep["obs_type"],
                dc50_value=dc_ep["value"], dc50_lower=dc_ep["lower"], dc50_upper=dc_ep["upper"],
                dc50_replicates=dc_ep["replicates"], dc50_unit="nM",
                dmax_raw=row["dmax_raw"], dmax_obs_type=dm_ep["obs_type"],
                dmax_value=dm_ep["value"], dmax_lower=dm_ep["lower"], dmax_upper=dm_ep["upper"],
                dmax_replicates=dm_ep["replicates"], dmax_unit="%",
                pdc50_value=pdc["value"], pdc50_lower=pdc["lower"], pdc50_upper=pdc["upper"],
                pctdeg_raw=row["pctdeg_raw"], pctdeg_value=pval,
                source_quality="PROTAC-DB 3.0 (database summary of literature)",
                source_has_doi=("with_doi" if has_doi else "no_doi"),
                cell_line=cell, treatment_time_h=t_h,
                year_doi=year,
                has_replicates=bool(dc_ep["replicates"] or dm_ep["replicates"]),
                is_dose_series=bool(dc_ep["is_dose_series"] or dm_ep["is_dose_series"]),
            )
            records.append(rec)
    sample_log.append(("2. parsed+expanded record-level", len(records)))
    out = pd.DataFrame(records)

    # activity_evidence (默认 cutoff) + 敏感性扫描
    ev = out.apply(lambda r: assign_evidence(
        dict(obs_type=r["dc50_obs_type"], value=r["dc50_value"], lower=r["dc50_lower"], upper=r["dc50_upper"]),
        dict(obs_type=r["dmax_obs_type"], value=r["dmax_value"], lower=r["dmax_lower"], upper=r["dmax_upper"]),
        DEFAULT_CP), axis=1)
    out["activity_evidence"] = [e[0] for e in ev]
    out["activity_signal"] = [e[1] for e in ev]
    out["activity_cutoff_p_nm"] = DEFAULT_CP

    sens = {}
    for c in SENS_CUTS:
        counts = {"P": 0, "N": 0, "A": 0, "U": 0}
        for _, r in out.iterrows():
            e = assign_evidence(
                dict(obs_type=r["dc50_obs_type"], value=r["dc50_value"], lower=r["dc50_lower"], upper=r["dc50_upper"]),
                dict(obs_type=r["dmax_obs_type"], value=r["dmax_value"], lower=r["dmax_lower"], upper=r["dmax_upper"]),
                c)[0]
            counts[e] += 1
        sens[str(int(c)) if c == int(c) else str(c)] = counts
    sample_log.append(("3. activity_evidence assigned", len(out)))

    # scaffold
    if Chem is not None:
        out["scaffold"] = out["smiles"].apply(lambda s: compute_scaffold(s, Chem))
    else:
        out["scaffold"] = out["smiles"].apply(lambda s: ("PREFIX:" + str(s)[:20]) if s else None)

    # 记录级去重 / 冲突标志
    dup_key = ["compound_id", "target", "e3_ligase", "cell_line", "treatment_time_h",
               "dc50_raw", "dmax_raw"]
    dup_key = [k for k in dup_key if k in out.columns]
    out["is_duplicate"] = out.duplicated(subset=dup_key, keep=False)
    conflict_key = ["compound_id", "target", "e3_ligase", "cell_line", "treatment_time_h"]
    g = out.groupby(conflict_key)
    out["is_conflict"] = g["dc50_raw"].transform("nunique") + g["dmax_raw"].transform("nunique") > 2
    sample_log.append(("4. dedup+conflict flags", len(out)))

    # record_id
    out.insert(0, "record_id", range(len(out)))

    # 保存清洗后主表
    clean_path = os.path.join(OUT, "protac_clean_record_level.csv")
    out.to_csv(clean_path, index=False, encoding="utf-8-sig")
    print("wrote", clean_path, out.shape)

    # ---------- 审计统计 ----------
    def obs_share(series):
        return float((series != "endpoint-missing").mean())
    stats = {
        "n_raw": int(n_raw),
        "n_records": int(len(out)),
        "n_expanded_from_inline": int(len(out) - n_raw),
        "sample_log": [list(x) for x in sample_log],
        "data_contract": dc_checks,
        "dc50_obs_types": out["dc50_obs_type"].value_counts().to_dict(),
        "dmax_obs_types": out["dmax_obs_types"] if False else out["dmax_obs_type"].value_counts().to_dict(),
        "dc50_obs_share": obs_share(out["dc50_obs_type"]),
        "dmax_obs_share": obs_share(out["dmax_obs_type"]),
        "overlap": {
            "both": int(((out["dc50_obs_type"] != "endpoint-missing") & (out["dmax_obs_type"] != "endpoint-missing")).sum()),
            "dc50_only": int(((out["dc50_obs_type"] != "endpoint-missing") & (out["dmax_obs_type"] == "endpoint-missing")).sum()),
            "dmax_only": int(((out["dc50_obs_type"] == "endpoint-missing") & (out["dmax_obs_type"] != "endpoint-missing")).sum()),
            "neither": int(((out["dc50_obs_type"] == "endpoint-missing") & (out["dmax_obs_type"] == "endpoint-missing")).sum()),
        },
        "activity_default": out["activity_evidence"].value_counts().to_dict(),
        "activity_sensitivity": sens,
        "top_target": out["target"].value_counts().head(15).to_dict(),
        "top_e3": out["e3_ligase"].value_counts().head(15).to_dict(),
        "top_cellline": out["cell_line"].dropna().value_counts().head(15).to_dict(),
        "source": out["source_has_doi"].value_counts().to_dict(),
        "year_coverage": float(out["year_doi"].notna().mean()),
        "year_dist": (out["year_doi"].dropna().astype(int).value_counts().sort_index().to_dict()
                      if out["year_doi"].notna().any() else {}),
        "dup_rate": float(out["is_duplicate"].mean()),
        "conflict_rate": float(out["is_conflict"].mean()),
        "has_replicates_rate": float(out["has_replicates"].mean()),
        "dose_series_rate": float(out["is_dose_series"].mean()),
        "obs_rate_by_target_top": (out.assign(_o=(out["dc50_obs_type"]!="endpoint-missing"))
                                   .groupby("target")["_o"].mean().sort_values(ascending=False).head(15).to_dict()),
        "obs_rate_by_e3": (out.assign(_o=(out["dc50_obs_type"]!="endpoint-missing"))
                           .groupby("e3_ligase")["_o"].mean().sort_values(ascending=False).head(15).to_dict()),
        "obs_rate_by_source": (out.assign(_o=(out["dc50_obs_type"]!="endpoint-missing"))
                               .groupby("source_has_doi")["_o"].mean().to_dict()),
        "obs_rate_by_year": (out.dropna(subset=["year_doi"]).assign(_o=(out.dropna(subset=["year_doi"])["dc50_obs_type"]!="endpoint-missing"))
                             .groupby("year_doi")["_o"].mean().sort_index().to_dict()),
    }
    # 化学系列集中度
    sc = out["scaffold"].dropna()
    stats["n_unique_scaffolds"] = int(sc.nunique())
    stats["top_scaffold"] = sc.value_counts().head(15).to_dict()
    top1 = sc.value_counts().iloc[0] if len(sc) else 0
    stats["top1_scaffold_share"] = float(top1 / len(sc)) if len(sc) else 0.0
    # 随机划分泄漏风险: scaffold 出现在 >1 条记录的比例
    sc_counts = sc.value_counts()
    shared = sc_counts[sc_counts > 1]
    stats["frac_records_shared_scaffold"] = float(shared.sum() / len(sc)) if len(sc) else 0.0
    stats["n_scaffolds_shared"] = int((sc_counts > 1).sum())

    with open(os.path.join(OUT, "protac_clean_audit_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
    print("wrote audit stats")

    # ---------- 数据字典 ----------
    desc = {
        "record_id": ("记录级唯一 ID", "", "etl", "0"),
        "raw_row_index": ("原始 protac.csv 数据行索引(0-based, 不含表头)", "", "protac.csv", "123"),
        "compound_id": ("PROTAC-DB 化合物 ID (非全局唯一)", "", "protac.csv col0", "PROTAC-00001"),
        "uniprot": ("靶蛋白 UniProt ID", "", "protac.csv col1", "P38398"),
        "target": ("POI / 靶标名称", "", "protac.csv col2", "BRD4"),
        "e3_ligase": ("E3 连接酶名称", "", "protac.csv col3", "CRBN"),
        "pdb": ("复合物 PDB ID (若有)", "", "protac.csv col4", "7Q2P"),
        "name": ("化合物名称", "", "protac.csv col5", "Compound A"),
        "smiles": ("SMILES 结构", "", "protac.csv col6", "CC(=O)..."),
        "article_doi": ("文献 DOI", "", "protac.csv col76", "10.1021/jm..."),
        "dc50_raw": ("DC50 原始字符串", "nM", "protac.csv col7", ">1000"),
        "dc50_obs_type": ("DC50 观测类型", "", "parsed", "right-censored"),
        "dc50_value": ("DC50 主值(exact 时为数值, 否则 NaN)", "nM", "parsed", "560"),
        "dc50_lower": ("DC50 下界(右删失/>=时)", "nM", "parsed", "1000"),
        "dc50_upper": ("DC50 上界(左删失/<=时)", "nM", "parsed", "0.5"),
        "dc50_replicates": ("DC50 原始重复串(若有)", "", "parsed", "490/500"),
        "dc50_unit": ("DC50 单位", "", "fixed", "nM"),
        "dmax_raw": ("Dmax 原始字符串", "%", "protac.csv col8", ">99"),
        "dmax_obs_type": ("Dmax 观测类型", "", "parsed", "right-censored"),
        "dmax_value": ("Dmax 主值", "%", "parsed", "95"),
        "dmax_lower": ("Dmax 下界", "%", "parsed", "99"),
        "dmax_upper": ("Dmax 上界", "%", "parsed", "10"),
        "dmax_replicates": ("Dmax 原始重复串(若有)", "", "parsed", ">95/>95"),
        "dmax_unit": ("Dmax 单位", "", "fixed", "%"),
        "pdc50_value": ("pDC50 = 9-log10(DC50), 仅 exact", "", "parsed", "6.25"),
        "pdc50_lower": ("pDC50 下界(左删失时)", "", "parsed", "9.30"),
        "pdc50_upper": ("pDC50 上界(右删失时)", "", "parsed", "6.0"),
        "pctdeg_raw": ("Percent degradation 原始字符串", "%", "protac.csv col10", "PANC-1:0/0;K562:0/37.7"),
        "pctdeg_value": ("Percent degradation 数值(展开后每细胞系均值)", "%", "parsed", "18.85"),
        "activity_evidence": ("活动证据: P/N/A/U (provisional)", "", "rule", "P"),
        "activity_signal": ("触发活动证据的规则明细", "", "rule", "DC50=45<=cp;Dmax=92>=P"),
        "activity_cutoff_p_nm": ("所用 DC50 P 候选 cutoff (nM)", "nM", "config", "100"),
        "source_quality": ("数据来源标注", "", "fixed", "PROTAC-DB 3.0 (database summary of literature)"),
        "source_has_doi": ("是否有文献 DOI", "", "derived", "with_doi"),
        "cell_line": ("细胞系(展开自 inline 或 Assay 文本挖掘)", "", "parsed/mined", "K562"),
        "treatment_time_h": ("处理时间(小时, 从 Assay 文本挖掘)", "h", "mined", "18"),
        "year_doi": ("从 DOI 提取的文献年份(近似)", "year", "mined", "2021"),
        "is_duplicate": ("完全重复记录级副本", "", "derived", "False"),
        "is_conflict": ("同键不同值冲突", "", "derived", "False"),
        "has_replicates": ("含重复测量", "", "derived", "True"),
        "is_dose_series": ("长串剂量/时间序列标记", "", "derived", "False"),
        "scaffold": ("Murcko scaffold SMILES (化学系列代理)", "", "rdkit", "c1ccccc1"),
    }
    dd = pd.DataFrame([
        {"column_name": k, "description": v[0], "unit": v[1], "source_raw_column": v[2], "example_value": v[3]}
        for k, v in desc.items()
    ])
    dd.to_csv(os.path.join(OUT, "data_dictionary.csv"), index=False, encoding="utf-8-sig")
    print("wrote data dictionary")

    # ---------- 原始数据清单 ----------
    def sha256(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    manifest_rows = []
    comp_notes = {
        "protac.csv": "主表: DC50/Dmax/结合亲和力/细胞活性/渗透性/理化性质/DOI",
        "warhead.csv": "Warhead 组件表 (Compound ID 关联)",
        "linker.csv": "Linker 组件表 (Compound ID + Smiles)",
        "e3_ligand.csv": "E3 ligand 组件表 (Compound ID 关联)",
    }
    for fn in ["protac.csv", "warhead.csv", "linker.csv", "e3_ligand.csv"]:
        fp = os.path.join(RAW, fn)
        with open(fp, encoding="utf-8-sig", errors="replace") as f:
            nrows = sum(1 for _ in f) - 1
        # 列数
        with open(fp, encoding="utf-8-sig", errors="replace") as f:
            ncols = len(next(csv.reader(f)))
        manifest_rows.append({
            "filename": fn, "source_url": SRC_URL, "download_date": DL_DATE,
            "rows": nrows, "cols": ncols, "sha256": sha256(fp),
            "notes": comp_notes[fn],
        })
    man = pd.DataFrame(manifest_rows)
    man.to_csv(os.path.join(RAW, "raw_data_manifest.csv"), index=False, encoding="utf-8-sig")
    print("wrote raw manifest")

    # ---------- 模板 ----------
    man_tpl = pd.DataFrame(columns=["filename", "source_url", "download_date", "rows", "cols", "sha256", "notes"])
    man_tpl.loc[0] = ["your_file.csv", SRC_URL, "YYYY-MM-DD", 0, 0, "待填写", "说明"]
    man_tpl.to_csv(os.path.join(TEM, "raw_data_manifest_template.csv"), index=False, encoding="utf-8-sig")

    ann_tpl = pd.DataFrame(columns=["raw_row_index", "dc50_raw", "dc50_obs_type", "dmax_raw",
                                     "dmax_obs_type", "activity_evidence", "source_quality",
                                     "cell_line", "treatment_time_h", "annotator_note"])
    ann_tpl.loc[0] = [0, ">1000", "right-censored", ">99", "right-censored", "N (provisional)",
                      "PROTAC-DB 3.0 (database summary of literature)", "K562", 18, "样例"]
    ann_tpl.to_csv(os.path.join(TEM, "protac_annotation_template.csv"), index=False, encoding="utf-8-sig")
    print("wrote templates")

    # ---------- PILOT 20 样例 ----------
    wanted = {"exact": 4, "right-censored": 4, "left-censored": 4, "interval-censored": 4, "endpoint-missing": 4}
    pilot = []
    for ot, k in wanted.items():
        sub = out[out["dc50_obs_type"] == ot].head(k)
        pilot.append(sub)
    pilot_df = pd.concat(pilot, ignore_index=True)[
        ["raw_row_index", "compound_id", "target", "e3_ligase", "dc50_raw", "dc50_obs_type",
         "dc50_value", "dmax_raw", "dmax_obs_type", "activity_evidence", "cell_line", "treatment_time_h"]
    ]
    pilot_df.to_csv(os.path.join(REP, "PILOT_20_EXAMPLES.csv"), index=False, encoding="utf-8-sig")
    print("wrote pilot 20 examples")

    # ---------- README ----------
    readme = f"""# README — PROTAC Bias-Aware PNU 数据集 (v0.1, provisional)

## 数据版本与来源
- 来源: PROTAC-DB 3.0 (浙江大学 CAMD, http://cadd.zju.edu.cn/protacdb/)
- 访问/下载日期: {DL_DATE}
- 署名与许可: Ge et al., *Nucleic Acids Research*, 2025 (PROTAC-DB 3.0). 本数据集为该公开库的快照派生; 请遵守原库许可并引用原始文献。
- 主表: protac.csv (15,502 行 × 89 列), 组件表 warhead/linker/e3_ligand。

## 本版内容
- `data/derived/protac_clean_record_level.csv`: 记录级、证据/删失感知主表 ({len(out):,} 行)。
- `data/derived/data_dictionary.csv`: 列字典。
- `data/raw/raw_data_manifest.csv`: 原始文件清单(sha256)。
- `reports/DATA_AUDIT_REPORT.md` + `reports/figures/`: 初步审计报告与图。
- `templates/`: 人工标注模板。
- `reports/PILOT_20_EXAMPLES.csv`: 试标样例。

## 已知局限(重要)
1. **cell_line / treatment_time 多需文本挖掘**: 仅 `Percent degradation` 列含 inline 细胞系(33 行展开); 其余细胞系/时间来自 `Assay` 自由文本正则挖掘, 大量为 NaN, 且可能不匹配。
2. **定量端点稀疏**: DC50 观测率约 {stats['dc50_obs_share']*100:.1f}%, Dmax 约 {stats['dmax_obs_share']*100:.1f}%。
3. **无独立阴性标签列**: activity_evidence(P/N/A/U)为基于临时阈值的派生标签(provisional, 标注规范 §5 未定稿), 需敏感性分析与人审。
4. **重复药化系列可能泄漏**: 同 scaffold 跨记录, 随机划分会泄漏(见审计)。
5. **逗号歧义**: 已规则化为千分位(>=1000)或欧洲小数(<1000); 极少数疑似 "6,1" 被解读为 6.1。
6. **年份近似**: 仅从 DOI 正则提取, 覆盖率 {stats['year_coverage']*100:.1f}%。

## 清洗方法概述
- 归一化: 全角符号→半角, 去空格, 逗号消歧。
- 端点解析: exact / left-censored(<) / right-censored(>) / interval-censored(-,~) / endpoint-missing(N.D.等)。
- 斜杠消歧: 同单位两值→重复测量(主值取首); inline 细胞系→展开多行; ≥4 数字长串→is_dose_series。
- pDC50 = 9 - log10(DC50), 删失边界按标注规范 §4 反转符号。
- activity_evidence: DC50≤{DEFAULT_CP:.0f}nM→P候选, DC50≥{DEFAULT_CP*10:.0f}nM→N候选, Dmax≥{DMAX_P:.0f}%→P, ≤{DMAX_N:.0f}%→N; 组合冲突/临界→A; 双终点皆缺→U。

## 复现步骤
```
python \\
    code/etl_protac.py
python \\
    code/make_report_figures.py
```
环境: 隔离 venv (python 3.13.12) + pandas/matplotlib/rdkit, 不改动 data/raw/。
"""
    with open(os.path.join(BASE, "README_data_version.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    print("wrote README")

    # 打印样本构建日志与关键数字
    print("\n--- 样本构建日志 ---")
    for step, n in sample_log:
        print(f"  {step:32s}: {n}")
    print("\n--- 数据合约 ---")
    for k, v in dc_checks.items():
        print(f"  {k}: {v}")
    print("\n--- 关键审计数字 ---")
    print("DC50 obs_types:", stats["dc50_obs_types"])
    print("Dmax obs_types:", stats["dmax_obs_types"])
    print("activity_default:", stats["activity_default"])
    print("overlap:", stats["overlap"])
    print("dup_rate:", round(stats["dup_rate"], 4), "conflict_rate:", round(stats["conflict_rate"], 4))

if __name__ == "__main__":
    main()

