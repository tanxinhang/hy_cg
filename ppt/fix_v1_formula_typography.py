# -*- coding: utf-8 -*-
"""
优化「V1_完整科研汇报_四部分修订版.pptx」的公式显示：
1. 公式字号收敛为三级体系（主公式 20 / 次级公式 18 / 参数说明 15），
   修掉现有 17.25~25.5 共 8 种混乱字号；
2. 把 "P_D"、"{tau}_ijq" 这类下划线伪下标改成真正的下标（baseline 下沉）；
3. 个别页面的说明行位置回正。
就地覆盖原文件，原件备份到 .workbuddy/backup/ 下。
"""
import os
import re
import shutil

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.oxml.ns import qn

SRC = r"D:\Desktop\conference\ppt\优化版\V1_完整科研汇报_四部分修订版.pptx"
BACKUP_DIR = r"D:\Desktop\conference\.workbuddy\backup"

F_MAIN = 20.0     # 主公式（独立成行的核心公式）
F_SUB = 18.0      # 次级公式 / 多行公式块
F_PARAM = 15.0    # 参数、条件、补充说明行
SUB_RATIO = 0.72  # 下标字号比例
SUB_BASE = -25000 # 下标下沉量（千分之一百分点）

CN = "微软雅黑"

# 逐形状的段落字号表：{页号: [(文本前缀, [第1段字号, 第2段字号, ...]), ...]}
PLAN = {
    5: [("f_q = arg min", [F_MAIN]), ("E_q = {", [F_SUB])],
    6: [("τ_ijq", [F_MAIN]),
        ("ν_ijq", [F_MAIN]),
        ("γ_e = 感知功率", [F_SUB, F_SUB])],
    7: [("感知功率 Pˢ", [F_MAIN]),
        ("χ ≈ 1 − Q", [F_MAIN]),
        ("C(γ)=log₂", [F_PARAM])],
    8: [("交付模型：Y_e", [F_MAIN]),
        ("A_e ~ Bernoulli", [F_PARAM]),
        ("δ_eᵉᶠᶠ", [F_SUB, F_SUB, F_SUB]),
        ("w_e ∝", [F_SUB, F_PARAM]),
        ("P̂_D,q = Q", [F_SUB, F_PARAM])],
    9: [("max_S", [F_MAIN]),
        ("约束：S ⊆", [F_SUB]),
        ("U = −Qτ_α", [F_SUB, F_SUB]),
        ("P̄_D,q = min", [F_PARAM]),
        ("观测数量上限包括本地证据", [F_PARAM])],
    11: [("f_q = arg min_m", [F_MAIN])],
    12: [("ΔF(e|S)", [F_MAIN, F_PARAM])],
    20: [("d(χ) =", [F_MAIN, F_PARAM]),
         ("Q_h,χ =", [F_MAIN, F_PARAM]),
         ("N_rem,q(m) =", [F_MAIN, F_PARAM])],
    21: [("矩预测与采样", [F_PARAM])],
}

# 兜底：含这些符号且字号大于 20 的公式行，一并压到主公式字号
MATH_HINT = re.compile(r"[=≈∝≤≥→]")
MATH_SYM = re.compile(r"[χτνγλδσρμΣ∑Δ√∞̂ᵀ²⁻]")

# 下标切分：_x 或 _{...}，字母含希腊字母区
SUB_SPLIT = re.compile(r"(_(?:\{[^}]*\}|[A-Za-z0-9,\u0370-\u03ff]+))")


def set_ea(run, name=CN):
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", name)


def first_run_style(shape):
    """取形状里第一个非空 run 的字体与颜色作为基准样式。"""
    for para in shape.text_frame.paragraphs:
        for r in para.runs:
            if r.text.strip():
                latin = r.font.name
                color = None
                try:
                    if r.font.color and r.font.color.type is not None:
                        color = r.font.color.rgb
                except Exception:
                    color = None
                return latin, color, bool(r.font.bold)
    return None, None, None


def add_math_runs(para, text, size, color, bold, latin):
    """把一段数学文本写入段落，_x 转成真下标。"""
    for r in list(para.runs):
        r._r.getparent().remove(r._r)
    for tk in SUB_SPLIT.split(text):
        if not tk:
            continue
        if tk.startswith("_"):
            body = tk[1:].strip("{}")
            run = para.add_run()
            run.text = body
            run.font.size = Pt(round(size * SUB_RATIO, 2))
            run.font.bold = bold
            if color is not None:
                run.font.color.rgb = color
            run.font.name = latin or "Arial"
            set_ea(run, CN)
            run._r.get_or_add_rPr().set("baseline", str(SUB_BASE))
        else:
            run = para.add_run()
            run.text = tk
            run.font.size = Pt(size)
            run.font.bold = bold
            if color is not None:
                run.font.color.rgb = color
            run.font.name = latin or "Arial"
            set_ea(run, CN)


def rework(shape, sizes):
    """按 sizes 逐段重设字号，并重建为真下标。"""
    latin, color, bold = first_run_style(shape)
    paras = shape.text_frame.paragraphs
    for i, para in enumerate(paras):
        text = "".join(r.text for r in para.runs)
        if not text.strip():
            continue
        size = sizes[i] if i < len(sizes) else sizes[-1]
        add_math_runs(para, text, size, color, bold, latin)


def convert_subscripts_only(shape):
    """只把 _x 转成真下标，字号保持与所在 run 一致（用于卡片正文等非公式框）。"""
    latin, color, bold = first_run_style(shape)
    for para in shape.text_frame.paragraphs:
        runs = [r for r in para.runs if r.text]
        if not runs:
            continue
        text = "".join(r.text for r in runs)
        if not SUB_SPLIT.search(text):
            continue
        size = None
        for r in runs:
            if r.font.size:
                size = r.font.size.pt
                break
        if size is None:
            continue
        add_math_runs(para, text, size, color, bold, latin)


def main():
    os.makedirs(BACKUP_DIR, exist_ok=True)
    backup = os.path.join(BACKUP_DIR, os.path.basename(SRC))
    if not os.path.exists(backup):          # 只备份一次，保留最原始版本
        shutil.copy2(SRC, backup)
    print("备份:", backup)

    prs = Presentation(SRC)
    hits = 0
    fallback = 0
    for idx, slide in enumerate(prs.slides, 1):
        plan = PLAN.get(idx, [])
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            head = shape.text_frame.text.strip().replace("\n", " ")[:40]
            if not head:
                continue
            matched = False
            for prefix, sizes in plan:
                if head.replace("\n", " ").startswith(prefix):
                    rework(shape, sizes)
                    hits += 1
                    matched = True
                    break
            if matched:
                continue
            # 兜底：公式特征明显且字号过大的，压到主公式字号
            first_size = None
            for para in shape.text_frame.paragraphs:
                for r in para.runs:
                    if r.text.strip() and r.font.size:
                        first_size = r.font.size.pt
                        break
                if first_size:
                    break
            if first_size and first_size > F_MAIN and MATH_HINT.search(head) and MATH_SYM.search(head):
                rework(shape, [F_MAIN])
                fallback += 1
                print(f"  兜底 P{idx}: {head[:32]!r} {first_size} -> {F_MAIN}")

    # 第二遍：所有还带 _x 伪下标的文本统一换成真下标（字号不变）
    subs = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if shape.has_text_frame and SUB_SPLIT.search(shape.text_frame.text):
                convert_subscripts_only(shape)
                subs += 1

    # 第二遍补：表格单元格里的 _x 也换成真下标
    tbl_cells = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_table:
                continue
            for row in shape.table.rows:
                for cell in row.cells:
                    tf = cell.text_frame
                    if SUB_SPLIT.search(tf.text):
                        convert_subscripts_only(cell)
                        tbl_cells += 1
    subs += tbl_cells
    print(f"  表格单元格下标 {tbl_cells} 处")

    # 第三遍：零散的 15.75pt 注释行归入 15pt，避免同层级字号不一
    norm = 0
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                for r in para.runs:
                    if r.font.size and abs(r.font.size.pt - 15.75) < 0.01:
                        r.font.size = Pt(15)
                        norm += 1

    prs.save(SRC)
    print(f"完成：命中 {hits} 个公式形状，兜底 {fallback} 个，补下标 {subs} 处，归一 {norm} 个")


if __name__ == "__main__":
    main()
