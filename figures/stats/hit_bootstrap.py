"""Table 2 (Section 4): excess whole-word hit rate (steered - control) per generation, with paired bootstrap
CIs over the 100 prompts, and the gens 8-10 mean. Recomputed rates are checked against control_comparison.json.
Usage: DATA_ROOT=/path/to/experiments/subliminal python hit_bootstrap.py"""
import json, re, os, numpy as np, collections
R=os.environ['DATA_ROOT']; M='deepseek-llm-7b-chat'
LABEL={'owl':'Owl','cat':'Cat','ai_supreme':'AI is superior to humans'}
def per_prompt(run,t,s,n):
    d=f'{R}/{run}/{M}/{t}/seed_{s}'+('' if n==1 else f'/gen_{n}')+'/results/generations.jsonl'
    rx=re.compile(rf"\b{re.escape(LABEL[t].lower())}s?\b",re.I)
    h=collections.defaultdict(list)
    for l in open(d):
        x=json.loads(l)
        if x['model']=='ft': h[x['prompt']].append(bool(rx.search(x['response'])))
    return h
rng=np.random.default_rng(0)
for t in ['owl','cat','ai_supreme']:
  for s in [42,43]:
    cc=json.load(open(f'{R}/adam_lora/{M}/{t}/seed_{s}/analysis/control_comparison.json'))['generations']
    diffs={}
    for n in range(1,11):
      a=per_prompt('adam_lora',t,s,n); b=per_prompt('control',t,s,n)
      ps=sorted(a); assert sorted(b)==ps and len(ps)==100
      A=np.array([np.mean(a[p]) for p in ps]); B=np.array([np.mean(b[p]) for p in ps])
      assert abs(A.mean()-cc[str(n)]['steered']['hit_rate_wb'])<1e-9, (t,s,n,A.mean(),cc[str(n)]['steered']['hit_rate_wb'])
      assert abs(B.mean()-cc[str(n)]['control']['hit_rate_wb'])<1e-9
      diffs[n]=A-B
    out=[]
    for n in [1,4,7,8,9,10]:
      D=diffs[n]; bs=[D[rng.integers(0,100,100)].mean() for _ in range(5000)]
      out.append(f'g{n} {D.mean()*100:+.2f}[{np.percentile(bs,2.5)*100:+.2f},{np.percentile(bs,97.5)*100:+.2f}]')
    P=(diffs[8]+diffs[9]+diffs[10])/3; bs=[P[rng.integers(0,100,100)].mean() for _ in range(5000)]
    first=[n for n in range(1,11) if all(np.percentile([diffs[m][rng.integers(0,100,100)].mean() for _ in range(2000)],2.5)<=0 for m in range(n,11))]
    print(t,s,' | '.join(out),f'| plat8-10 {P.mean()*100:+.2f}[{np.percentile(bs,2.5)*100:+.2f},{np.percentile(bs,97.5)*100:+.2f}] rel {P.mean()/diffs[1].mean()*100:.1f}% | CI incl 0 from gen {first[0] if first else None}')
