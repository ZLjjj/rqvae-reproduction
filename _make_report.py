from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
import json

ROOT = r"C:\Users\dszlj\Desktop\gensearchrec-main-0901"
summary = json.load(open(ROOT + r"\results_three_way\evaluation\three_way_summary.json", encoding="utf-8"))

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr(); shd = OxmlElement('w:shd'); shd.set(qn('w:fill'), fill); tcPr.append(shd)
def borders(table, color='D9D9D9'):
    tblPr = table._tbl.tblPr; b = tblPr.first_child_found_in('w:tblBorders')
    if b is None: b = OxmlElement('w:tblBorders'); tblPr.append(b)
    for edge in ('top','left','bottom','right','insideH','insideV'):
        tag='w:'+edge; el=b.find(qn(tag))
        if el is None: el=OxmlElement(tag); b.append(el)
        el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:space'),'0'); el.set(qn('w:color'),color)
def set_cell(cell, text, bold=False, color=None):
    cell.text=''; p=cell.paragraphs[0]; p.paragraph_format.space_after=Pt(2); r=p.add_run(str(text)); r.bold=bold; r.font.size=Pt(9)
    if color: r.font.color.rgb=RGBColor(*color)
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
def add_table(doc, headers, rows, widths=None):
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'; borders(t)
    for i,h in enumerate(headers): set_cell(t.rows[0].cells[i],h,True); shade(t.rows[0].cells[i],'D9EAF7')
    for row in rows:
        cells=t.add_row().cells
        for i,v in enumerate(row): set_cell(cells[i],v)
    if widths:
        for row in t.rows:
            for i,w in enumerate(widths): row.cells[i].width=Inches(w)
    doc.add_paragraph().paragraph_format.space_after=Pt(2)
    return t
def pct(x): return f'{x*100:.4f}%'
def f6(x): return f'{x:.6f}'

doc=Document(); sec=doc.sections[0]; sec.top_margin=Inches(.65); sec.bottom_margin=Inches(.65); sec.left_margin=Inches(.7); sec.right_margin=Inches(.7)
styles=doc.styles; styles['Normal'].font.name='Microsoft YaHei'; styles['Normal']._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei'); styles['Normal'].font.size=Pt(10)
for s in ('Title','Heading 1','Heading 2'):
    styles[s].font.name='Microsoft YaHei'; styles[s]._element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei'); styles[s].font.color.rgb=RGBColor(0,0,0)

p=doc.add_paragraph(style='Title'); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('SID 三方视频量化对比实验报告')
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.add_run('基于 video-only 训练与 text_format 清洗的 RQVAE 对比').italic=True
doc.add_paragraph('本报告记录三种 SID 方案在同一批 193,034 条视频资源上的离散编码质量对比，回答两个问题：视频单独训练是否优于 station+video 混合 baseline，以及清洗和重组 text_format 是否进一步改善前缀聚类结构。')

doc.add_heading('1 实验结论', level=1)
doc.add_paragraph('视频-only 训练带来了决定性的碰撞率改善。任务2的 5 层 ICR 从 baseline 的 0.961333 提升到 0.995125，CR 从 3.8667% 降至 0.4875%，完整 SID 碰撞率下降约 87.4%。')
doc.add_paragraph('任务3的 cleaned text_format 与任务2的最终唯一率几乎持平（ICR 0.995110），但前缀结构更均衡：L1 最大 bucket 从 9,511 降至 6,962，L1/L2/L3 的 weighted cosine 分别提升到 0.6201/0.7560/0.8505。说明字段清洗主要改善早期 prefix 的语义聚合和分桶质量，而不是继续显著提高最终 SID 唯一率。')
doc.add_paragraph('若首要目标是完整 SID 唯一率，任务2略优；若同时关注前缀召回、分桶均衡和前缀内语义一致性，任务3更有优势。')

doc.add_heading('2 实验设计', level=1)
doc.add_paragraph('三方均使用 193,034 条视频资源。baseline 来自现有 station+video 混合索引文件中的连续 video 尾段；任务2使用同源的完整视频 embedding 重新训练；任务3使用清洗后的 text_format embedding 重新训练。三方的评测脚本统一读取数字 indices 或 SID token，并统一计算完整路径、prefix bucket 和 bucket 内 cosine 指标。')
add_table(doc,['方案','索引/训练输入','embedding','目的'],[
['baseline 视频','现有混合 SID 的 video 子集','2_meta_full_video.npy 对齐视频行','衡量现有混合域 codebook 在视频上的表现'],
['任务2 同源 video-only','video-only RQVAE','2_meta_full_video.npy','隔离训练域影响'],
['任务3 cleaned text_format','video-only RQVAE','3_text_format_embedding.npy','验证 text_format 清洗与重组的影响']], [1.35,2.05,2.2,2.0])

doc.add_heading('3 数据与训练配置', level=1)
add_table(doc,['项目','值'],[
['任务2元数据/向量','2_meta_full_video.jsonl / 2_meta_full_video.npy'],
['任务3元数据/向量','3_meta.jsonl / 3_text_format_embedding.npy'],
['样本数与维度','193,034 × 2,560，float32'],
['模型','MLP Encoder → 4 层 RQ → hard-code 第5层 → MLP Decoder'],
['codebook','[1024, 1024, 1024, 1024, 256]；评测容量按第五层 9-bit hard-code 统计'],
['latent dimension','e_dim = 128'],
['MLP layers','[512, 256, 128]'],
['Sinkhorn','[0, 0, 0, 0.003, 0]'],
['EMA codebook','启用，decay = 0.99，eps = 1e-5'],
['训练','40 epochs，AdamW，batch size 1024，CPU'],
['选择 checkpoint','best_collision_model.pth']], [2.0,5.6])

doc.add_heading('4 核心指标结果', level=1)
rows=[]
for key,label in [('baseline_video','baseline 视频'),('same_source_video','任务2 同源 video-only'),('video_v3','任务3 cleaned text_format')]:
    x=summary[key]; rows.append([label, f'{x["N"]:,}', f6(x['icr']), pct(x['collision_rate']), f'{x["unique_paths"]:,}', f6(x['total_cur'])])
add_table(doc,['方案','N','ICR','CR','unique paths','total CUR'],rows,[1.8,1.0,1.2,1.2,1.5,1.5])
doc.add_paragraph('CR = 1 - ICR。三方 N 完全一致，因此差异来自 codebook 学习与输入 embedding，而不是样本量。')

doc.add_heading('5 Prefix bucket 统计', level=1)
for level in range(5):
    rows=[]
    for key,label in [('baseline_video','baseline'),('same_source_video','任务2'),('video_v3','任务3')]:
        b=summary[key]['bucket_size'][level]; rows.append([label,f'{b["bucket_count"]:,}',f'{b["mean"]:.3f}',f'{b["median"]:.1f}',f'{b["p25"]:.1f}',f'{b["p75"]:.1f}',f'{b["max"]:,}'])
    doc.add_paragraph(f'L{level+1} bucket').runs[0].bold=True
    add_table(doc,['方案','bucket 数','均值','中位数','P25','P75','最大值'],rows,[1.5,1.0,1.0,1.0,.8,.8,1.0])

doc.add_heading('6 Prefix CUR 与 bucket cosine', level=1)
rows=[]
for level in range(5):
    rows.append([f'L{level+1}', f6(summary['baseline_video']['prefix_cur_list'][level]), f6(summary['same_source_video']['prefix_cur_list'][level]), f6(summary['video_v3']['prefix_cur_list'][level])])
add_table(doc,['层级','baseline CUR','任务2 CUR','任务3 CUR'],rows,[1.2,1.7,1.7,1.7])
doc.add_paragraph('CUR 随 prefix 深度增加而下降，反映离散空间容量远大于实际使用路径。三方的绝对 CUR 不宜脱离 codebook 容量单独解读，本文更关注相同容量下的相对变化。')
rows=[]
for level in range(4):
    vals=[]
    for key in ('baseline_video','same_source_video','video_v3'):
        z=summary[key]['bucket_cosine'][level]; vals.append(f'{z["weighted_sim"]:.6f} / {z["unweighted_sim"]:.6f}')
    rows.append([f'L{level+1}',*vals])
add_table(doc,['层级','baseline 加权/非加权','任务2 加权/非加权','任务3 加权/非加权'],rows,[1.0,2.1,2.1,2.1])
doc.add_paragraph('cosine 统计排除了 hard-code 第5层，仅计算 L1-L4。weighted_sim 按 bucket 内样本对数加权，unweighted_sim 对 bucket 均值等权。')

doc.add_heading('7 分析与解释', level=1)
doc.add_heading('7.1 混合域训练对视频的影响', level=2)
doc.add_paragraph('baseline 的 L1 只有 420 个 prefix bucket，最大 bucket 达 3,065；任务2和任务3分别只使用 49 和 79 个 L1 bucket，但由于重新训练后的第一层 code 分配不同，最大 bucket 分别为 9,511 和 6,962。真正的区分度主要在后续层：任务2 L4/L5 最大 bucket 为 9/9，任务3为 8/8，而 baseline 为 77/47。video-only 训练使后续层几乎实现一条路径对应一条资源，从而显著降低完整 SID 碰撞。')
doc.add_heading('7.2 text_format 清洗的作用', level=2)
doc.add_paragraph('任务3在 L1-L3 的 weighted cosine 均高于任务2，表明清洗后的字段拼接让共享 prefix 的资源语义更一致；同时 L1 最大 bucket 降低约 26.8%，说明最粗粒度分桶更均衡。L4 weighted cosine 从任务2的 0.9263 降至 0.9147，属于后层细分结构的轻微变化，但 L4/L5 最大 bucket 进一步降到 8，最终 CR 仍保持在 0.49% 左右。')
doc.add_heading('7.3 业务选择建议', level=2)
doc.add_paragraph('推荐将任务3作为默认候选方案进行后续检索/推荐链路验证：它在唯一率上与任务2等价，同时前缀层语义一致性和 bucket 均衡性更好。若下游更强调极限 SID 去重率，则保留任务2作为对照；两者都明显优于现有 baseline。')

doc.add_heading('8 产物与复现', level=1)
doc.add_paragraph('训练和评测产物：')
for txt in [
    r'results_three_way/same_source_video/checkpoints/best_collision_model.pth',
    r'results_three_way/same_source_video/indices/indices.jsonl',
    r'results_three_way/video_v3/checkpoints/best_collision_model.pth',
    r'results_three_way/video_v3/indices/indices.jsonl',
    r'results_three_way/evaluation/three_way_summary.json',
]: doc.add_paragraph(txt, style='List Bullet')
doc.add_paragraph('统一评测入口为 sid/scripts/eval/run_sid_three_way.py；输入校验入口为 sid/scripts/eval/verify_three_way_inputs.py。两轮训练均在 CPU 上完成，每轮 40 epochs，checkpoint 取 best_collision_model.pth。')

doc.add_heading('9 限制与后续工作', level=1)
doc.add_paragraph('本实验只评估离散 codebook 的结构指标，没有覆盖 LLM SFT 后的真实查询召回率、Recall@K、NDCG 或线上延迟。下一步应使用同一批查询和候选资源，对任务2/任务3分别进行 SID 生成约束推理，并比较端到端检索效果。另需在 CUDA 环境复跑一次，以确认 CPU 与 GPU 的数值和 checkpoint 选择不存在实现差异。')

out=ROOT+r'\SID三方视频对比实验报告.docx'; doc.save(out); print(out)
