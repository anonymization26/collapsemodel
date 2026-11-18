"""Summarize all completed units; never count pending runs as evidence."""
import argparse
from collections import defaultdict
import csv
import json
from itertools import combinations
from pathlib import Path
from statistics import mean,median
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_fresh import table,save

METHODS=['initial15','sequential25','batch25','exploit25','init_mmd25','init_random25','mmd25','aopt25','random25','mmd50','aopt50','full455']
LABELS={'initial15':'仅初始化15','sequential25':'逐次25','batch25':'两批25','exploit25':'无探索25','init_mmd25':'相同前15+MMD10','init_random25':'相同前15+随机10','mmd25':'MMD25','aopt25':'A-opt25','random25':'随机25','mmd50':'MMD50','aopt50':'A-opt50','full455':'完整455'}

def read(path):return list(csv.DictReader(path.open()))
def mdtable(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|',*['| '+' | '.join(map(str,r))+' |' for r in rows]])
def number(value):return f'{value:.6f}'
def grouped(rows,method):
    groups=defaultdict(list)
    full={r['unit']:r['winner'] for r in rows if r['method']=='full455'}
    for r in rows:
        if r['method']==method:groups[r['unit']].append(r)
    return [dict(unit=u,brier=mean(float(r['brier']) for r in v),gap_full=mean(float(r['gap_full']) for r in v),same_full=mean(float(r['winner']==full[u]) for r in v)) for u,v in groups.items()]

def main(root):
    completed=sorted(p for p in root.iterdir() if p.is_dir() and (p/'audit.json').exists())
    opt=[p for p in completed if p.name.startswith('optdigits_')]
    dom=[p for p in completed if p.name.startswith('domainnet_')]
    ridge=[p for p in completed if p.name.startswith('ridge_')]
    rows=[r for p in opt+dom for r in read(p/'outcomes.csv')]
    if rows:table(root/'all_logistic_outcomes.csv',rows)
    rr=[r for p in ridge for r in read(p/'outcomes.csv')]
    if rr:table(root/'all_ridge_outcomes.csv',rr)
    # Post-hoc explanation only: fit the already observed validation response.
    design=np.array([[float(i in c) for i in range(15)] for c in combinations(range(15),3)])
    ridge_r2=[]
    for p in ridge:
        if not (p/'models.npz').exists():continue
        y=np.load(p/'models.npz',allow_pickle=False)['validation_brier']
        pred=design@np.linalg.lstsq(design,y,rcond=None)[0]
        r2=1-np.sum((y-pred)**2,axis=0)/np.sum((y-y.mean(axis=0))**2,axis=0)
        ridge_r2.extend(dict(unit=p.name.removeprefix('ridge_'),arm=arm,full_additive_r2=float(value)) for arm,value in zip(['uncalibrated_fixed','calibrated_fixed','calibrated_tuned'],r2))
    if ridge_r2:table(root/'ridge_full_additive_r2.csv',ridge_r2)
    main_result=''
    if len(dom)==48:
        dr=[r for r in rows if r['unit'].startswith('domainnet_')]
        seq=grouped(dr,'sequential25')
        main_result=f"新DomainNet主实验已覆盖六域、四种子和两种池构造：逐次25有 {sum(int(r['same_full']) for r in seq)}/48 次选中完整455的同一赢家，{sum(r['gap_full']>.005 for r in seq)}/48 个单元的测试Brier差值超过+0.005。它支持当前主设置下的低预算选型，但不能消除下面的独立数据、读出和指标边界。"
    exploratory=''
    if (root/'exploratory_interactions.csv').exists():
        er=read(root/'exploratory_interactions.csv');elines=[]
        for kind in ['hash','pc1']:
            vals=[]
            for model in ['additive15','pair105']:
                v=[r for r in er if r['unit'].startswith('optdigits_') and f'_{kind}_' in r['unit'] and r['model']==model]
                vals.append(number(mean(float(r['oof_r2']) for r in v)))
            elines.append([kind,*vals])
        exploratory='\n'.join(['### 事后探索：交互项能否解释失效','',mdtable(['池构造','加性15参数：五折R²','池对105参数：五折R²'],elines),'',
            '这个诊断在看到失败后新增，明确标为探索性；只读取本轮455个验证损失，不读取测试标签。每折用约364个候选拟合，不能作为25查询算法的结果。池对项显著改善同候选族中未见组合的预测，支持遗漏交互的解释，但没有证明因果或跨池泛化。',
            '此外，二阶矩MMD本身并非池贡献的加性函数。对等大池，M(S)=||(C₁+C₂+C₃)/3−C_T||²，展开后包含池对内积项。它能够利用无标签几何中的组合信息；这提供了理解它为何可能在加性查询失效时仍有效的线索，不构成MMD普遍可靠的保证。'])
    report=['# 全新补充实验：结果与叙事审查','',
        '**这是新下载数据、新拟合和新计时的结果，不包含此前归档回放。**',
        f'已完成：Optdigits 逻辑回归 {len(opt)}/8 单元，DomainNet 逻辑回归 {len(dom)}/48 单元；岭回归 {len(ridge)}/38 单元。未完成单元不进入结论。',
        '', '## 当前判断', '',
        main_result,
        '新叙事更容易解释算法在做什么，但目前不能表述为对共享池候选族的一般有效方案。Optdigits 的随机分池支持低预算选型；同一批源样本按无标签 PC1 分池后，出现稳定而明显的失败。这个对照把“有共享池”与“损失近似可加”区分开。',
        '四个种子共享原始数据；两个池构造也共享每个种子的样本。它们是相关重复实验，不是八个独立数据集。这里的 +0.005 是预设描述性阈值，不是等价性检验或统计显著性。',
        '', '## 1. Optdigits：相同预算及相同初始化对照','',
        '官方训练集选择 3000 条源样本，分成 15 个 200 样本池；每次拟合使用三个池。官方测试集重新划为 256 条无标签筛选、512 条验证、1029 条最终测试。四种子、两种池构造均按预先协议报告。所有策略用验证 Brier 选一个模型，再报告该模型的测试 Brier/accuracy/NLL。',
        'NLL按实验代码使用1e-15的概率下界；主表Brier不截断。减少查询次数不等于减少目标验证集的标签数量。',
        '数据来源：[UCI Optdigits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits)。这些池是人工构造，不属于自然多源域。', '']
    optrows=[r for r in rows if r['unit'].startswith('optdigits_')]
    summary=[]
    for method in METHODS:
        line=[LABELS[method]]
        for construction in ['hash','pc1']:
            v=[r for r in optrows if f'_{construction}_' in r['unit'] and r['method']==method]
            line.extend([number(mean(float(r['brier']) for r in v)),number(mean(float(r['gap_full']) for r in v))] if v else ['待完成','待完成'])
        summary.append(line)
    report.extend([mdtable(['策略','随机分池 Brier↓','相对完整搜索 Δ','PC1分池 Brier↓','相对完整搜索 Δ'],summary),'',
        '随机策略先在每个单元内平均 20 个预定顺序，再跨四种子平均；表中等权计算与之等价。完整455是验证集选择的参照，不是测试集最优模型，因此 Δ 可以为负。',
        '', '## 2. 加性假设与逐次更新机制',''])
    diagnostics=[]
    for construction in ['hash','pc1']:
        units=[p for p in opt if f'_{construction}_' in p.name]
        if not units:continue
        fullr2=mean(json.loads((p/'audit.json').read_text())['full_additive_r2'] for p in units)
        for n in ['15','25']:
            ds=[r for p in units for r in read(p/'diagnostics.csv') if r['method']=='sequential25' and r['n']==n]
            diagnostics.append([construction,n,number(fullr2),*[number(mean(float(r[k]) for r in ds)) for k in ['rmse','r2','spearman','top25_recall']]])
    report.extend([mdtable(['池构造','已查询数','全部候选加性拟合R²','未查询RMSE','未查询R²','Spearman','Top25召回'],diagnostics),'',
        '15与25查询的预测诊断使用同一组“25次后仍未查询”的候选。全部候选加性拟合仅用于事后机制解释；它没有参与选型。PC1分池下，连事后最佳加性拟合也难以解释损失变化，说明问题并不只是少了几次更新。',
        '本轮 Optdigits 逐次查询与去探索项版本在八个单元得到相同赢家，不能把收益归因于探索项。共同初始化后补 MMD 的对照保留了 PC1 构造中的好模型，说明池交互或分布覆盖可能比单池可加贡献更关键。这个机制解释是推断，不是另一个已验证的新算法。',
        '',exploratory,
        '', '## 3. 岭回归正则化与校准','',
        '验证集前半只选择温度/λ，后半只选择候选。λ∈{0.1,1,10}，温度81点。调λ版本每个查询支付三次拟合，因此25个候选等于75次监督拟合，不能写成与固定λ同为25次拟合。', ''])
    report.append('λ搜索仅改变岭读出与校准；几何排序保持原定义，A-opt的排序正则化仍为1。这不是对几何排序与读出正则化的联合调优。')
    report.append('岭回归成本保存在各单元decisions.json中：fit_seconds记录共享Gram矩阵准备后的线性求解，calibration_seconds记录温度搜索，audit_seconds记录完整共享审计。这些是实际分项与审计成本，不是各查询策略的独立部署耗时；本轮不据此宣称岭回归策略加速比。')
    ridge_table=[]
    for construction in ['hash','pc1']:
        for arm in ['uncalibrated_fixed','calibrated_fixed','calibrated_tuned']:
            line=[construction,arm]
            for method in ['sequential25','mmd25','full455']:
                v=[r for r in rr if r['unit'].startswith('optdigits_') and f'_{construction}_' in r['unit'] and r['arm']==arm and r['method']==method]
                line.append(number(mean(float(r['brier']) for r in v)) if v else '待完成')
            ridge_table.append(line)
    report.extend([mdtable(['池构造','岭回归设置','逐次25 Brier↓','MMD25 Brier↓','完整455 Brier↓'],ridge_table),'',
        '校准带来的改善仍然存在，但没有消除 PC1 分池下的查询失效。调λ的收益依设置变化，不能据此声称固定λ=1结论完全不敏感。不同臂会重新选型；表中 accuracy 的变化不能解释为“同一模型温度缩放改变了分类预测”。',
        '', '## 4. 独立实测耗时','',
        '下面是同一单元（Optdigits/hash/20261117）的三次独立重训中位耗时，起点为驻留特征，包含筛选或查询计算、拟合、验证与赢家选择。BLAS单线程；不含测试审计。不用质量审计中共享的455次拟合时间冒充部署时间。',
        '硬件记录见hardware.json：读出运行在Intel Xeon Gold 6133共享服务器上（72个逻辑CPU），每次使用1或4个拟合线程；图像提取使用物理0号V100。计时期间停止了本轮其他计算，没有独占整台服务器。',''])
    tfile=root/'optdigits_hash_20261117/timing.csv'
    if tfile.exists():
        ts=read(tfile);lines=[]
        for method in ['sequential25','batch25','mmd25','mmd50','aopt25','full455']:
            lines.append([LABELS[method],*[f"{median(float(r['seconds']) for r in ts if r['method']==method and r['workers']==w):.3f}" for w in ['1','4']]])
        report.append(mdtable(['策略','1拟合线程/秒','4拟合线程/秒'],lines))
    report.extend(['','这个小规模CPU读出任务中，四线程没有产生加速，不能沿用旧服务器的并行加速比。这里尚不涉及 GPU 图像编码成本。', '', '## 5. DomainNet 状态','',
        '本轮使用官方cleaned版中预先按哈希选定的100类、六个自然目标域、新提取的ResNet50特征，主读出为128维投影。它是新实验设置，不能与旧稿双编码器的均值直接拼接。图像和目标划分在四种子间不变；种子改变投影及hash分池，PC1分池固定。',''])
    if dom:
        for construction in ['hash','pc1']:
            dr=[r for r in rows if r['unit'].startswith('domainnet_') and f'_{construction}_' in r['unit']]
            lines=[]
            for method in METHODS:
                v=grouped(dr,method)
                if v:lines.append([LABELS[method],len(v),number(mean(r['brier'] for r in v)),number(mean(r['gap_full'] for r in v)),sum(r['gap_full']>.005 for r in v),f"{mean(r['same_full'] for r in v):.1%}"])
            report.extend([f'### {construction}',mdtable(['策略','已完成单元','Brier↓','相对完整搜索 Δ','Δ>0.005 单元数','与完整搜索赢家一致率'],lines),''])
        target_lines=[]
        mechanism_rows=[r for r in rows if r['unit'].startswith('domainnet_')]
        if len(dom)==48:
            batch=[r for r in mechanism_rows if r['method']=='batch25'];exploit=[r for r in mechanism_rows if r['method']=='exploit25'];completion=[r for r in mechanism_rows if r['method']=='init_mmd25']
            report.extend(['### 更新与探索各自贡献',
                f"相同前15次后补MMD，相对逐次查询平均Brier高 {mean(float(r['gap_seq']) for r in completion):.6f}，支持用验证反馈指导后续候选。两批版本相对逐次仅高 {mean(float(r['gap_seq']) for r in batch):.6f}，{sum(int(r['same_seq']) for r in batch)}/48次赢家相同；去探索项版本相对逐次高 {mean(float(r['gap_seq']) for r in exploit):.6f}，{sum(int(r['same_seq']) for r in exploit)}/48次赢家相同。", 
                '因此，当前证据更支持“利用池结构和验证反馈进行低预算补全”，尚不足以把持续逐次更新或探索项本身作为主要实证贡献。',''])
        for target in ['clipart','infograph','painting','quickdraw','real','sketch']:
            for construction in ['hash','pc1']:
                dr=[r for r in rows if r['unit'].startswith('domainnet_'+target+'_') and f'_{construction}_' in r['unit']]
                if not dr:continue
                v=grouped(dr,'sequential25');line=[target,construction,len(v)]
                for method in ['sequential25','init_mmd25','mmd25','mmd50','full455']:
                    v=grouped(dr,method);line.append(number(mean(r['brier'] for r in v)))
                target_lines.append(line)
        report.extend(['### 逐目标质量',mdtable(['目标域','池构造','种子数','逐次25','同初始+MMD','MMD25','MMD50','完整455'],target_lines),''])
        other_metrics=[]
        for dataset in ['domainnet','optdigits']:
            for construction in ['hash','pc1']:
                subset=[r for r in rows if r['unit'].startswith(dataset+'_') and f'_{construction}_' in r['unit']]
                for method in ['sequential25','mmd25','full455']:
                    v=[r for r in subset if r['method']==method]
                    if v:other_metrics.append([dataset,construction,LABELS[method],number(mean(float(r['brier']) for r in v)),f"{100*mean(float(r['accuracy']) for r in v):.3f}",number(mean(float(r['nll']) for r in v))])
        report.extend(['### 同一赢家的多指标核对',mdtable(['数据集','池构造','策略','Brier↓','Accuracy %↑','NLL↓'],other_metrics),
            '所有指标来自验证Brier选出的同一模型，没有按指标重新选型。DomainNet的PC1构造中，逐次查询平均Brier优于MMD25，但NLL较差；因此不能把Brier结果表述为所有概率质量指标上的优势。NLL使用1e-15概率下界，缺失源类别采用零概率对齐。',''])
        domain_ridge=[]
        for dimension in [64,96,128,256,512]:
            for arm in ['uncalibrated_fixed','calibrated_fixed','calibrated_tuned']:
                v=[r for r in rr if r['unit'].startswith('domainnet_') and f'_d{dimension}_' in r['unit'] and r['arm']==arm]
                if not v:continue
                line=[dimension,arm,len(set(r['unit'] for r in v))]
                for method in ['sequential25','mmd25','full455']:
                    line.append(number(mean(float(r['brier']) for r in v if r['method']==method)))
                full={r['unit']:float(r['brier']) for r in v if r['method']=='full455'}
                gaps=[float(r['brier'])-full[r['unit']] for r in v if r['method']=='sequential25']
                r2v=[r['full_additive_r2'] for r in ridge_r2 if r['unit'].startswith('domainnet_') and f'_d{dimension}_' in r['unit'] and r['arm']==arm]
                line.extend([sum(g>.005 for g in gaps),number(max(gaps)),number(mean(r2v)) if r2v else '未本地同步模型'])
                domain_ridge.append(line)
        if domain_ridge:report.extend(['### 岭回归维度敏感性（hash池，首种子）',mdtable(['维度','设置','已完成域数','逐次25','MMD25','完整455','逐次Δ>0.005域数','逐次最大Δ','全候选加性R²'],domain_ridge),
            '最后一列是在看到部分敏感性结果后补充的事后解释性诊断：对本轮455个候选的已观测验证损失拟合15个池贡献。它没有参与选型，也不能用作部署时可获得的可靠性证据。近似可加性取决于候选族、读出和评分函数的组合，不能只由池重叠结构推出。',''])
        timing_rows=[r for p in dom if (p/'timing.csv').exists() for r in read(p/'timing.csv')]
        if timing_rows:
            table(root/'domainnet_all_timings.csv',timing_rows);lines=[]
            for workers in ['1','4']:
                reference={(r['unit'],r['repeat']):float(r['seconds']) for r in timing_rows if r['method']=='mmd50' and r['workers']==workers}
                for method in ['sequential25','batch25','mmd25','mmd50','aopt25','full455']:
                    v=[r for r in timing_rows if r['workers']==workers and r['method']==method]
                    lines.append([workers,LABELS[method],len(v),f"{median(float(r['seconds']) for r in v):.3f}",f"{median(reference[r['unit'],r['repeat']]/float(r['seconds']) for r in v):.3f}"])
            report.extend(['### 独立驻留特征计时',mdtable(['拟合线程','策略','完成测量数','秒/中位数','相对MMD50配对中位加速'],lines),''])
        cold=[]
        for p in sorted(root.glob('cold_*.json')):
            r=json.loads(p.read_text());cold.append([r['method'],r['images'],f"{r['image_feature_seconds']:.3f}",f"{r['seconds']:.3f}",f"{r['total_seconds']:.3f}"])
        if cold:report.extend(['### 原图完整管线（clipart单单元，每方法一次）',mdtable(['策略','实际编码图像数','提取秒','决策秒','完整管线秒'],cold),'原图与预训练权重已存在；OS页面缓存未控制，不含下载和Python导入。每方法独立执行，未把旧阶段时间相加。',''])
        breakdown=root/'cost_breakdown_clipart.json'
        if breakdown.exists():
            bd=json.loads(breakdown.read_text());lines=[[r['method'],*[f"{r[k]:.3f}" for k in ['fit_seconds','validation_seconds','ranking_acquisition_dispatch_seconds','seconds']]] for r in bd['rows']]
            report.extend(['### 单次决策成本分解（clipart，额外独立单线程执行）',mdtable(['策略','拟合秒','验证秒','排序/查询/调度等秒','决策合计秒'],lines),
                f"特征缓存解码中位数 {median(r['cache_decode_seconds'] for r in bd['loading']):.3f} 秒，完整性哈希检查中位数 {median(r['integrity_hash_seconds'] for r in bd['loading']):.3f} 秒；两者均不包含在驻留特征决策时间内。OS文件缓存未控制。分解仅用于说明开销，不替代三次重复的主计时表。",''])
    else:report.append('尚无新 DomainNet 读出结果；原图下载、特征提取、48单元主实验、维度敏感性和计时均不能在此算作已完成证据。')
    status=root/'campaign_status.json'
    if status.exists():report.append('当前机器状态：`'+json.loads(status.read_text())['stage']+' / '+json.loads(status.read_text())['state']+'`。')
    report.extend(['','## 6. 建议的论文叙事与 rebuttal 优先级','',
        '**可保留的主线：** 把候选数据组合看作一个可利用结构的选型空间；当损失近似可加时，少量验证反馈可指导后续候选拟合。用明确的失效构造展示其适用边界。',
        '**必须收窄的表述：** 15次识别仅在精确可加、精确观测下成立；25次是经验预算；共享池本身不足以保证加性；探索项收益、跨数据集优势及并行加速都须逐项实证。',
        '**识别与部署需要区分：** 15次可以在理想假设下识别完整验证损失曲面，却不保证预测最优候选已被训练。若它不在初始15项中，还需拟合该候选才能交付模型。验证集上的识别也不是测试风险保证。',
        '**P0：** 把已完成的同初始15对照和加性预测诊断放进主回复，正反两种池构造都报告。本轮DomainNet支持主设置中的经验结论；Optdigits反例要求把主张收窄到可预测的损失结构，不能只强调共享池本身。',
        '**P0：** 创新性仍需正面比较线性 best-arm identification / 实验设计。共享线性结构用于辨别候选最优项已有研究，不能把“拟合池贡献再查询”本身当作充分的新颖性证据。可以明确本文的具体候选族、失配条件、查询成本和批处理权衡，再评估是否有超出应用实例的贡献。',
        '对应关系很直接：候选组合是arm，池成员指示向量是已知特征，负验证损失是reward。15次理想识别主要来自满秩线性系统可解。现有任务的确定性验证分数与模型失配不同于许多随机噪声设定，因此已有置信保证不能直接移植；系数0.001的探索项也不能自动解释为有效置信区间。公平对照需要匹配无重复候选、验证集与真实拟合预算。',
        '相关一手文献：[Soare et al., NeurIPS 2014](https://proceedings.neurips.cc/paper_files/paper/2014/hash/f8d84caae48546d0934d637ab54f7086-Abstract.html)；[Fiez et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/8ba6c657b03fc7c8dd4dff8e45defcd2-Abstract.html)。这是待处理的对照缺口，并非本轮已复现这些方法。',
        '**P1：** 将已完成的DomainNet λ/维度敏感性、独立计时和原图管线结果纳入修订。保留校准/维度变化下的退化单元，不把均值接近写成普适稳健。原图实验只覆盖预定clipart单元、每策略一次，无置信区间；不能称为全域冷启动基准。',
        '**后续方法研究：** 设计仅靠选型阶段信息的失配检测和安全回退，或用池交互模型。必须重新冻结方案并用新的验证证据测试，不能根据当前测试结果切换算法后仍当作盲测。',
        '初始15项恰好拟合15个参数，训练残差可以为零，因而不能把初始拟合残差当成加性假设可靠的证据。可部署的失配检查需要额外、未用于拟合的查询，并把这些查询计入预算。本轮全候选/五折诊断没有解决这一部署问题。',
        '25次预算的经验结论限定于15池选3的455候选族。本轮未验证更大候选族、自适应停止、目标标注数量减少或端到端骨干微调；新图像实验仅采用冻结ResNet50特征。',
        '', '## 7. 文件与复现边界','',
        '- 原始逐单元结果：各单元目录中的 `outcomes.csv`、`diagnostics.csv`、`decisions.json`。',
        '- 锁定证据：`freeze.json`、`selection_lock.json`、`audit.json`；选型锁定后才计算最终测试预测。',
        '- 合并表：`all_logistic_outcomes.csv`、`all_ridge_outcomes.csv`；随机顺序不是额外独立样本。',
        '- 代码与协议：`../../fresh_20261117/`。旧归档结果未进入这些表。',
        '- 服务器代码目录：`/home/ubuntu/caozheng/collapsemodel_rebuttal_20261117`；新数据：`/mnt/datadisk0/datasets/collapsemodel_20261117`。',
        '- GPU任务仅使用物理0号卡；读出拟合在CPU上运行。未向GitHub推送。',''])
    (root/'Results.md').write_text('\n'.join(report))
    if len(opt)==8:
        chosen=['initial15','sequential25','batch25','init_mmd25','init_random25','mmd25','full455']
        fig,axes=plt.subplots(1,2,figsize=(13,4.5),sharey=False)
        for ax,kind,title in zip(axes,['hash','pc1'],['Hash pools: favorable setting','PC1 pools: additive-model failure']):
            for k,method in enumerate(chosen):
                vals=[]
                for unit in sorted({r['unit'] for r in optrows if f'_{kind}_' in r['unit']}):
                    vals.append(mean(float(r['brier']) for r in optrows if r['unit']==unit and r['method']==method))
                ax.scatter(np.arange(4)*.055+k-.0825,vals,s=24,c='#2b6cb0' if kind=='hash' else '#c05621',alpha=.75)
                ax.plot([k-.18,k+.18],[mean(vals)]*2,c='black',lw=2)
            ax.set_xticks(range(len(chosen)),['Initial15','Seq25','Batch25','Init+MMD','Init+Rand','MMD25','Full455'],rotation=30,ha='right')
            ax.set_title(title);ax.set_ylabel('Test Brier (lower is better)');ax.grid(axis='y',alpha=.2)
        fig.suptitle('Fresh Optdigits: identical source inventory, two pool constructions\nDots: four correlated split seeds; bars: means; random additions averaged within each seed',fontsize=11)
        fig.tight_layout();fig.savefig(root/'optdigits_pool_boundary.png',dpi=180);fig.savefig(root/'optdigits_pool_boundary.pdf');plt.close(fig)
    if len(dom)==48:
        fig,ax=plt.subplots(figsize=(8,5))
        for dataset,marker in [('domainnet','o'),('optdigits','D')]:
            for construction,color in [('hash','#2b6cb0'),('pc1','#c05621')]:
                units=[p for p in opt+dom if p.name.startswith(dataset+'_') and f'_{construction}_' in p.name]
                points=[]
                for unit in units:
                    a=json.loads((unit/'audit.json').read_text())
                    r=next(r for r in read(unit/'outcomes.csv') if r['method']=='sequential25')
                    points.append((a['full_additive_r2'],float(r['gap_full'])))
                ax.scatter([p[0] for p in points],[p[1] for p in points],marker=marker,c=color,s=42,label=f'{dataset} / {construction}',alpha=.8)
        ax.axhline(.005,color='gray',ls='--',lw=1,label='+0.005 descriptive threshold')
        ax.axhline(0,color='black',lw=.7);ax.set_xlabel('Full-family additive fit R² (post hoc)')
        ax.set_ylabel('Seq25 test Brier − full-validation winner');ax.grid(alpha=.15)
        ax.set_title('Success and failure across all 56 fresh logistic units')
        ax.legend(loc='upper right',fontsize=8)
        fig.text(.5,.015,'Correlated repetitions; post-hoc association, not a deployable diagnostic or causal test.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.035,1,1));fig.savefig(root/'additivity_and_selection_gap.png',dpi=180);fig.savefig(root/'additivity_and_selection_gap.pdf');plt.close(fig)
    domain_ridge_units=[p for p in ridge if p.name.startswith('ridge_domainnet_')]
    if len(domain_ridge_units)==30:
        dims=[64,96,128,256,512];fig,axes=plt.subplots(1,2,figsize=(12,4.5))
        for arm,color,label in [('calibrated_fixed','#2b6cb0','Fixed lambda=1'),('calibrated_tuned','#c05621','Validation-tuned lambda')]:
            fullmeans=[];seqmeans=[];gapmeans=[]
            for k,dim in enumerate(dims):
                v=[r for r in rr if r['unit'].startswith('domainnet_') and f'_d{dim}_' in r['unit'] and r['arm']==arm]
                full={r['unit']:float(r['brier']) for r in v if r['method']=='full455'}
                seq={r['unit']:float(r['brier']) for r in v if r['method']=='sequential25'}
                fullmeans.append(mean(full.values()));seqmeans.append(mean(seq.values()));gaps=[seq[u]-full[u] for u in sorted(full)];gapmeans.append(mean(gaps))
                offset=-.1 if arm=='calibrated_fixed' else .1
                axes[1].scatter(np.arange(6)*.018+k+offset-.045,gaps,c=color,alpha=.65,s=20)
            axes[0].plot(range(5),fullmeans,'o-',c=color,label=label+' / full455')
            axes[0].plot(range(5),seqmeans,'s--',c=color,label=label+' / seq25')
            axes[1].plot(range(5),gapmeans,'o-',c=color,label=label+' / mean')
        for ax in axes:ax.set_xticks(range(5),dims);ax.set_xlabel('Nested projected dimension');ax.grid(alpha=.2)
        axes[0].set_ylabel('Mean test Brier (lower is better)');axes[0].legend(fontsize=8)
        axes[1].axhline(.005,color='gray',ls='--',label='+0.005 descriptive threshold');axes[1].axhline(0,color='black',lw=.7)
        axes[1].set_ylabel('Seq25 test Brier − full-validation winner');axes[1].legend(fontsize=8)
        fig.suptitle('Fresh DomainNet ridge sensitivity: six targets, hash pools, one seed',fontsize=12)
        fig.tight_layout();fig.savefig(root/'ridge_dimension_sensitivity.png',dpi=180);fig.savefig(root/'ridge_dimension_sensitivity.pdf');plt.close(fig)
    print('SUMMARY',root/'Results.md',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();main(a.root)
