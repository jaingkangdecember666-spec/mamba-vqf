"""Inspect split identities and timestamp supports without reading validation or test orientations."""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'C001/.deps'))
import h5py


def main():
    families=defaultdict(list)
    source=Path('E:/数据集/YT_server/YT_server_200Hz.h5')
    with h5py.File(source,'r') as f:
        for split in ('train','valid','test'):
            for key, group in f[split].items():
                t=group['ts']
                prefix, segment=key.rsplit('_',1)
                families[prefix].append({'name':key,'split':split,'segment':int(segment),'frames':len(t),
                                          'start_s':float(t[0]), 'end_s':float(t[-1])})
    records=[r for group in families.values() for r in group]
    membership=Counter('/'.join(sorted({r['split'] for r in group})) for group in families.values())
    spans=[]
    for group in families.values():
        for i,a in enumerate(group):
            for b in group[i+1:]:
                if a['split']==b['split']:
                    continue
                gap=max(b['start_s']-a['end_s'],a['start_s']-b['end_s'])
                spans.append((gap,a['name'],b['name']))
    test_families={k for k,v in families.items() if any(r['split']=='test' for r in v)}
    unrelated={k:v for k,v in families.items() if k not in test_families}
    proof={'source_size_bytes':source.stat().st_size,'trials_if_prefix_is_trial':len(families),
           'records_by_official_split':dict(Counter(r['split'] for r in records)),
           'family_split_membership':dict(membership),'families_touched_by_official_test':len(test_families),
           'records_in_families_without_official_test':dict(Counter(r['split'] for v in unrelated.values() for r in v)),
           'cross_split_same_prefix_pairs':len(spans), 'intervals_overlapping':sum(x[0]<0 for x in spans),
           'cross_split_gaps_under_one_second':sum(0<=x[0]<=1 for x in spans),
           'closest_nonoverlapping':sorted((s for s in spans if s[0]>=0))[:7],
           'largest_overlaps':sorted((s for s in spans if s[0]<0))[:7],
           'families':families}
    (ROOT/'review/yt_group_audit.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
    print(json.dumps({k:v for k,v in proof.items() if k!='families'},ensure_ascii=False))


if __name__=='__main__':
    main()
