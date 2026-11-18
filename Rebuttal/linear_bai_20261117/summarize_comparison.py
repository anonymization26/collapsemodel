"""Descriptive summaries only; parameters are never selected by these outputs."""
import csv,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent/'linear_runs'
LABEL={'seq':'Sequential LCB','batch':'Two-batch','exploit':'Pure exploitation',
       'd':'D-design','a':'A-design','g':'G-design','xy':'XY-design',
       'lingape_0.001':'LinGapE-FB .001','lingape_0.01':'LinGapE-FB .01',
       'lingape_0.1':'LinGapE-FB .1','lingape_1':'LinGapE-FB 1',
       'mmd':'MMD','target_aopt':'Target A-opt','od_linbai_original':'OD-LinBAI (original)'}

def read(p):
    rows=list(csv.DictReader(p.open()))
    for r in rows:
        for k in ('brier','accuracy','nll','gap_full','gap_seq25','validation_brier','validation_regret','full_brier','seconds','fit_seconds','allocation_seconds','validation_seconds'):
            if k in r:r[k]=float(r[k])
        for k in ('budget','winner','unique_fits','pulls','same_full','repeat'):
            if k in r:r[k]=int(r[k])
    return rows
def write(p,rows):
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def md(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    units=sorted(p for p in ROOT.iterdir() if p.is_dir() and (p/'outcomes.csv').exists())
    assert len(units)==56,len(units)
    rows=[r for p in units for r in read(p/'outcomes.csv')];write(ROOT/'all_outcomes.csv',rows)
    references={(r['unit'],r['budget']):r for r in rows if r['method']=='seq' and r['finish']=='observed' and r['comparison']=='standard'}
    groups=defaultdict(list)
    for r in rows:
        if r['comparison']=='standard':groups[(r['dataset'],r['construction'],r['budget'],r['method'],r['finish'])].append(r)
    summary=[]
    for key,rr in sorted(groups.items()):
        dataset,construction,budget,method,finish=key
        diff=[r['brier']-references[(r['unit'],budget)]['brier'] for r in rr] if budget in (25,50,100) else []
        valdiff=[r['validation_regret']-references[(r['unit'],budget)]['validation_regret'] for r in rr] if diff else []
        summary.append(dict(dataset=dataset,construction=construction,budget=budget,method=method,finish=finish,n=len(rr),
            brier=np.mean([r['brier'] for r in rr]),accuracy=np.mean([r['accuracy'] for r in rr]),nll=np.mean([r['nll'] for r in rr]),
            gap_full=np.mean([r['gap_full'] for r in rr]),validation_regret=np.mean([r['validation_regret'] for r in rr]),
            same_full=sum(r['same_full'] for r in rr),bad_005=sum(r['gap_full']>.005 for r in rr),
            delta_seq=np.mean(diff) if diff else '',validation_delta_seq=np.mean(valdiff) if diff else '',
            wins=sum(x< -1e-12 for x in diff),ties=sum(abs(x)<=1e-12 for x in diff),losses=sum(x>1e-12 for x in diff)))
    write(ROOT/'summary.csv',summary)
    times=[r for p in units if (p/'timing.csv').exists() for r in read(p/'timing.csv')]
    if times:write(ROOT/'all_timings.csv',times)
    # Equal-fit-budget comparison: show all predeclared constants, no oracle tuning.
    primary=[('seq','observed'),('batch','observed'),('exploit','observed')]+[(m,'recommend') for m in ('d','a','g','xy','lingape_0.001','lingape_0.01','lingape_0.1','lingape_1')]+[('mmd','observed'),('target_aopt','observed')]
    sections=[]
    for dataset,construction in [('domainnet','hash'),('domainnet','pc1'),('optdigits','hash'),('optdigits','pc1')]:
        rr=[]
        for method,finish in primary:
            r=next(s for s in summary if (s['dataset'],s['construction'],s['budget'],s['method'],s['finish'])==(dataset,construction,25,method,finish))
            rr.append([LABEL[method],finish,f"{r['brier']:.6f}",f"{r['validation_regret']:.6f}",f"{r['delta_seq']:+.6f}",f"{r['same_full']}/{r['n']}",r['bad_005'],f"{r['wins']}/{r['ties']}/{r['losses']}"])
        sections.append(f'### {dataset} / {construction}\n\n'+md(['方法','终止规则','测试 Brier ↓','验证 simple regret ↓','相对逐次 ΔBrier','同全量赢家','超过 +.005','胜/平/负'],rr))
    (ROOT/'primary_tables.md').write_text('\n\n'.join(sections)+'\n')
    # Six-target paired domain summaries keep repeated seeds visibly clustered.
    domain=[]
    for method,finish in primary:
        for construction in ('hash','pc1'):
            rr=[r for r in rows if r['dataset']=='domainnet' and r['construction']==construction and r['budget']==25 and r['method']==method and r['finish']==finish]
            for target in sorted({r['unit'].split('_')[1] for r in rr}):
                group=[r for r in rr if r['unit'].split('_')[1]==target]
                domain.append(dict(method=method,finish=finish,construction=construction,target=target,n=len(group),
                    delta_seq=np.mean([r['brier']-references[(r['unit'],25)]['brier'] for r in group])))
    write(ROOT/'domain_paired_differences.csv',domain)
    # Original OD gets both the cap comparison and the exact unique-fit comparison.
    odrows=[]
    for dataset in ('domainnet','optdigits'):
        for construction in ('hash','pc1'):
            matched=[r for r in rows if r['dataset']==dataset and r['construction']==construction and r['comparison']=='matched_od_unique']
            cap=[r for r in rows if r['dataset']==dataset and r['construction']==construction and r['comparison']=='matched_od_cap']
            for finish in ('native_prediction','observed'):
                od=[r for r in rows if r['dataset']==dataset and r['construction']==construction and r['method']=='od_linbai_original' and r['finish']==finish]
                odrows.append(dict(dataset=dataset,construction=construction,finish=finish,n=len(od),
                    od_brier=np.mean([r['brier'] for r in od]),od_val_regret=np.mean([r['validation_regret'] for r in od]),
                    od_same_full=sum(r['same_full'] for r in od),od_bad005=sum(r['gap_full']>.005 for r in od),
                    pulls_min=min(r['pulls'] for r in od),pulls_max=max(r['pulls'] for r in od),
                    unique_min=min(r['unique_fits'] for r in od),unique_max=max(r['unique_fits'] for r in od),
                    seq_matched_unique_brier=np.mean([r['brier'] for r in matched]),seq200_brier=np.mean([r['brier'] for r in cap])))
    write(ROOT/'od_comparison.csv',odrows)
    # Plot downstream risk gaps, with primary budget highlighted.
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    plotmethods=[('seq','observed'),('exploit','observed'),('d','recommend'),('g','recommend'),('xy','recommend'),('lingape_0.001','recommend'),('lingape_0.1','recommend')]
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,(dataset,construction) in zip(axes.flat,[('domainnet','hash'),('domainnet','pc1'),('optdigits','hash'),('optdigits','pc1')]):
        for method,finish in plotmethods:
            ss=sorted([s for s in summary if s['dataset']==dataset and s['construction']==construction and s['method']==method and s['finish']==finish and s['budget'] in (25,50,100)],key=lambda s:s['budget'])
            ax.plot([s['budget'] for s in ss],[s['gap_full'] for s in ss],marker='o',label=LABEL[method])
        ax.axhline(0,color='gray',linewidth=.6);ax.axhline(.005,color='gray',linestyle=':',linewidth=.8)
        ax.set(title=f'{dataset} / {construction}',xlabel='Unique classifier fits',ylabel='Test Brier minus full-validation winner',xticks=[25,50,100])
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=4)
    fig.suptitle('Frozen 56-cell follow-up: descriptive means, correlated repeats',fontsize=13)
    fig.savefig(ROOT/'budget_comparison.png',dpi=160);fig.savefig(ROOT/'budget_comparison.pdf');plt.close(fig)
    # Termination ablation for pure-design methods, not just their weaker observed form.
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,construction in zip(axes,('hash','pc1')):
        mm=['seq','d','g','xy','lingape_0.001'];x=np.arange(len(mm));width=.25
        for shift,finish in enumerate(('observed','recommend','native_prediction')):
            val=[next(s['gap_full'] for s in summary if (s['dataset'],s['construction'],s['budget'],s['method'],s['finish'])==('domainnet',construction,25,m,finish)) for m in mm]
            ax.bar(x+(shift-1)*width,val,width,label=finish)
        ax.set_xticks(x,[LABEL[m] for m in mm],rotation=15);ax.axhline(0,color='gray',lw=.6)
        ax.set(title=f'DomainNet / {construction}, B=25',ylabel='Mean test Brier gap to full validation')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.savefig(ROOT/'termination_fairness.png',dpi=160);fig.savefig(ROOT/'termination_fairness.pdf');plt.close(fig)
    print('SUMMARY',len(rows),'decisions',len(times),'timings',flush=True)
    print((ROOT/'primary_tables.md').read_text(),flush=True)

if __name__=='__main__':main()
