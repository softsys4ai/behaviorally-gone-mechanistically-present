"""Table 3 (Section 5): base-model and random-direction ablation costs, causal residue DiD by generation,
control drift, the floor fit and the DiD / likelihood-gap ratio.
Usage: DATA_ROOT=/path/to/experiments/subliminal python ablation_summary.py"""
import json, os
R=os.environ['DATA_ROOT']; M='deepseek-llm-7b-chat'
for t in ['owl','cat','ai_supreme']:
  for s in [42,43]:
    cc=json.load(open(f'{R}/adam_lora/{M}/{t}/seed_{s}/analysis/control_comparison.json'))
    sub=cc['ablation_subdirs']
    S=json.load(open(f'{R}/adam_lora/{M}/{t}/seed_{s}/analysis/{sub["steered"]}.json'))['generations']
    C=json.load(open(f'{R}/control/{M}/{t}/seed_{s}/analysis/{sub["control"]}.json'))['generations']
    print(f'== {t} s{s} (steered file {sub["steered"]}) gens S={sorted(S,key=int)[:2]}..  runs?')
    b=S['0']['loglik_drop']; bc=C['0']['loglik_drop']
    print(f' base v_c drop {b["v_c"]:.3f} (ctrl file {bc["v_c"]:.3f}); base rand {[round(v,3) for k,v in b.items() if k.startswith("rand")]}')
    rmax_s=max(abs(v) for g in S.values() for k,v in g['loglik_drop'].items() if k.startswith('rand'))
    rmax_c=max(abs(v) for g in C.values() for k,v in g['loglik_drop'].items() if k.startswith('rand'))
    print(f' max|rand drop| steered {rmax_s:.3f} control {rmax_c:.3f}')
    for n in ['1','4','7','10']:
      d=S[n]['loglik_drop']; c=C[n]['loglik_drop']; g=cc['generations'][n]
      ex=C[n].get('vc_effect_excess_over_base',{}).get('mean')
      print(f'  g{n:>2} S: vc {-d["v_c"]:.2f} vr {-d.get("v_r",float("nan")):.2f} span {-d.get("span",float("nan")):.2f} | C: vc {-c["v_c"]:.2f} vr {-c.get("v_r",float("nan")):.2f} ctrl-excess-over-base {ex if ex is None else round(ex,2)} | DiD {g["vc_effect_did"]:.2f} {[round(x,2) for x in g["vc_effect_did_ci95"]]}')
    ce=[C[str(n)].get('vc_effect_excess_over_base',{}).get('mean') for n in range(1,11)]
    print('  ctrl excess over base by gen', [None if x is None else round(x,2) for x in ce])
    P=cc['summary']['vc_effect_did']; L=cc['summary']['ll_gap']['plateau']
    print(f'  DiD plateau {P["plateau"]["mean"]:.2f} {[round(x,2) for x in P["plateau"]["ci95"]]} fit r {P["fit"]["rate_per_gen"]} floor {P["fit"]["floor"]:.2f} r2 {P["fit"]["r2"]:.3f} | DiD g1 {cc["generations"]["1"]["vc_effect_did"]:.2f} ret {P["plateau"]["mean"]/cc["generations"]["1"]["vc_effect_did"]*100:.0f}% | LLgap plat {L["mean"]:.2f} ratio DiD/gap {P["plateau"]["mean"]/L["mean"]:.2f}')
