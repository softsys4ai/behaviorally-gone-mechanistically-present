"""Collect every per-generation number the figures need into figdata.json (CPU only, read-only on the runs)."""
import json, re, collections, os, sys
import numpy as np, torch
R = os.environ['DATA_ROOT']   # parent of adam_lora/ and control/ (as in config.env)
M = os.environ.get('MODEL_SHORT', 'deepseek-llm-7b-chat')
LABEL = {'owl': 'Owl', 'cat': 'Cat', 'ai_supreme': 'AI is superior to humans'}
G = range(1, 11)
rng = np.random.default_rng(0)

def J(p): return json.load(open(p))
def chain(run, t, s): return f'{R}/{run}/{M}/{t}/seed_{s}'

def per_prompt(run, t, s, n):
    p = chain(run, t, s) + ('' if n == 1 else f'/gen_{n}') + '/results/generations.jsonl'
    rx = re.compile(rf"\b{re.escape(LABEL[t].lower())}s?\b", re.I)
    h = collections.defaultdict(list)
    for l in open(p):
        x = json.loads(l)
        if x['model'] == 'ft': h[x['prompt']].append(bool(rx.search(x['response'])))
    ps = sorted(h); return ps, np.array([np.mean(h[q]) for q in ps])

def boot(d, B=5000):
    m = d[rng.integers(0, len(d), (B, len(d)))].mean(1); return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

def vr(run, t, s, n):
    p = chain(run, t, s) + ('' if n == 1 else f'/gen_{n}') + f'/Recover_Vector/vr_gen{n}.pt'
    v = torch.load(p, map_location='cpu', weights_only=True)
    if isinstance(v, dict): v = [x for x in v.values() if torch.is_tensor(x)][0]
    return v.double().numpy().ravel()
def cos(a, b): return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))

out = {}
for t in LABEL:
    for s in (42, 43):
        cc = J(chain('adam_lora', t, s) + '/analysis/control_comparison.json')
        g = cc['generations']; base = cc['base_model']
        d = {k: [] for k in ['cos', 'cos_ctrl', 'll_gap', 'll_gap_ci', 'll_st', 'll_ct', 'did', 'did_ci', 'abl_st', 'abl_ct',
                             'hit_ex', 'hit_ex_ci', 'hit_st', 'hit_ct', 'parent', 'align_ctrl', 'a50']}
        dr = J(chain('adam_lora', t, s) + '/analysis/drift_analysis.json')['consecutive_drift']
        vf = 'verbalization_judge.json' if t == 'ai_supreme' else 'verbalization.json'
        ver = J(chain('adam_lora', t, s) + '/analysis/' + vf)['generations']
        for n in G:
            x = g[str(n)]
            d['cos'].append(x['cos_steered']); d['cos_ctrl'].append(x['cos_control'])
            d['ll_gap'].append(x['ll_gap']); d['ll_gap_ci'].append(x['ll_gap_ci95'])
            d['ll_st'].append(x['ll_steered'] - base['avg_log_likelihood']); d['ll_ct'].append(x['ll_control'] - base['avg_log_likelihood'])
            d['did'].append(x['vc_effect_did']); d['did_ci'].append(x['vc_effect_did_ci95'])
            d['abl_st'].append(x['vc_effect_steered']); d['abl_ct'].append(x['vc_effect_control'])
            pa, A = per_prompt('adam_lora', t, s, n); pb, B = per_prompt('control', t, s, n)
            assert pa == pb and len(pa) == 100
            assert abs(A.mean() - x['steered']['hit_rate_wb']) < 1e-9 and abs(B.mean() - x['control']['hit_rate_wb']) < 1e-9
            d['hit_ex'].append(float((A - B).mean() * 100)); d['hit_ex_ci'].append([c * 100 for c in boot(A - B)])
            d['hit_st'].append(float(A.mean() * 100)); d['hit_ct'].append(float(B.mean() * 100))
            d['parent'].append(dr.get(str(n)))
            d['align_ctrl'].append(cos(vr('adam_lora', t, s, n), vr('control', t, s, n)))
            a = ver[str(n)]['alpha_50']
            if t == 'ai_supreme' and n >= 9: a = None   # probe outputs degenerate (Section 3.3)
            d['a50'].append(a)
        # random-direction ablation band: max |loglik drop| over rand_0..4, base model (gen 0) and steered students, gens 1-10 (as Table 3)
        mx = 0.0
        for run in ('adam_lora',):   # steered students only, as in the Rand. column of Table 3
            sub = cc['ablation_subdirs']['steered' if run == 'adam_lora' else 'control']
            f = chain(run, t, s) + '/analysis/' + ('causal_ablation_ll.json' if sub == 'causal_ablation_ll' else 'causal_ablation.json')
            for k, v in J(f)['generations'].items():
                if int(k) >= 0: mx = max(mx, max(abs(v['loglik_drop'][r]) for r in v['loglik_drop'] if r.startswith('rand')))
        d['rand_band'] = mx
        ca = J(chain('control', t, s) + '/analysis/causal_ablation.json')['generations']
        d['abl_base'] = -ca['0']['loglik_drop']['v_c']
        # control-vector max mention (alpha sweep), for the alpha_50 panel note
        cv = J(chain('control', t, s) + '/analysis/' + vf)['generations']
        d['ctrl_max_mention'] = max(v['mention_rate'] for k in cv for v in cv[k]['per_alpha'].values())
        out[f'{t}_{s}'] = d
        print(t, s, 'ok', 'rand', round(mx, 3), 'base abl', round(d['abl_base'], 3), file=sys.stderr)

# D2D (cat, owl)
for t in ('cat', 'owl'):
    for s in (42, 43):
        c = J(chain('adam_lora', t, s) + '/analysis/d2d_comparison_d2d_cart16.json')['generations']
        out[f'd2d_{t}_{s}'] = {n: {k: c[n][k] for k in ['hit_amp_did', 'hit_amp_did_ci95', 'amp_did', 'amp_did_ci95', 'dormant',
                                                        'bare_hit_gap', 'bare_hit_gap_ci95', 'cart_hit_gap', 'cart_hit_gap_ci95']} for n in c}
for s in (42, 43):
    m = J(chain('adam_lora', 'cat', s) + '/analysis/cartridge_readout_d2d_cart16.json')['m3b']
    out[f'vcart_cat_{s}'] = {n: m[n]['v_cart'] for n in m}
json.dump(out, open(sys.argv[1], 'w'), indent=1)
