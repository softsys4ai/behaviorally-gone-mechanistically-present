"""Table 1 (Section 3): recovery cosine per generation, log-linear decay fits, control cosines, alpha_50.
Usage: DATA_ROOT=/path/to/experiments/subliminal python decay_fits.py
Note: for ai_supreme the paper uses the judge-based alpha_50 (analysis/verbalization_judge.json), not the regex one printed here."""
import json, math, numpy as np, os
B=os.path.join(os.environ['DATA_ROOT'],'adam_lora','deepseek-llm-7b-chat')
def fit(ys, start):
    n=np.arange(start,11); y=np.log(np.array([ys[i] for i in n]))
    s,i=np.polyfit(n,y,1); r=math.exp(s); res=y-(s*n+i); r2=1-res.var()/y.var()
    return r, math.log(.5)/math.log(r), r2
for t in ['owl','cat','ai_supreme']:
  for s in [42,43]:
    d=json.load(open(f'{B}/{t}/seed_{s}/analysis/control_comparison.json'))
    g=d['generations']; gens=sorted(g,key=int)
    cs={int(k):g[k]['cos_steered'] for k in gens}; cc=[g[k]['cos_control'] for k in gens]
    a=[g[k]['steered']['alpha_50'] for k in gens]; ac=[g[k]['control']['alpha_50'] for k in gens]
    print(f'== {t} s{s} gens={len(gens)}')
    print(' cos', [round(cs[i],3) for i in sorted(cs)])
    print(' step ratios', [round(cs[i+1]/cs[i],3) for i in range(1,10)])
    print(' fit1-10 r,hl,r2', [round(x,3) for x in fit(cs,1)], ' fit2-10', [round(x,3) for x in fit(cs,2)])
    print(' endpoint ratio', round((cs[10]/cs[1])**(1/9),3))
    print(' ctrl cos max|.|', round(max(abs(x) for x in cc),3), [round(x,3) for x in cc])
    print(' a50', [None if x is None else round(x,2) for x in a]); print(' ctrl a50', ac)
    print(' steered keys', list(g['1']['steered'].keys()))
    p=f'{B}/{t}/seed_{s}/analysis/drift_analysis.json'
    if os.path.exists(p):
      dd=json.load(open(p)); print(' consec', [round(v,3) for k,v in sorted(dd['consecutive_drift'].items(),key=lambda x:int(x[0]))])
    else: print(' no drift file')
    v=f'{B}/{t}/seed_{s}/analysis/verbalization.json'
    print(' verbalization.json', os.path.exists(v))
