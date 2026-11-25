"""Preselect a larger external training panel using identifiers and split membership only."""
import hashlib
import json
from collections import Counter,defaultdict

from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def tie(value):
    return hashlib.sha256(('C019-source-diversity-20261123|'+value).encode()).hexdigest()


def main():
    inventory=ROOT/'C001/inventory.json'
    rows=[r for r in json.loads(inventory.read_text(encoding='utf8'))['records'] if r['dataset']=='YT_Server']
    eligible=[r for r in rows if r['side']=='external_training_candidate' and r['official_split']=='train']
    previous=ROOT/'C004/protocol.json'
    retained=[s['record'] for s in json.loads(previous.read_text(encoding='utf8'))['source_records'] if s['domain']=='YT_Server']
    lookup={r['record']:r for r in eligible}
    pool=defaultdict(list)
    for r in eligible:
        pool[r['group']].append(r['record'])
    forbidden={r['group'] for r in rows if r['official_split']=='test' or r['side']=='development'}
    assert not (set(pool)&forbidden) and len(pool)==279 and len(eligible)==910 and len(retained)==24
    picks=[]
    used=set()
    activities=Counter()
    people=Counter()
    pairs=Counter()
    def register(record):
        group=lookup[record]['group']
        assert group not in used
        code,activity,*_=group.split('_')
        used.add(group)
        activities[activity]+=1
        people[code]+=1
        pairs[code,activity]+=1
        picks.append({'record':record,'group':group,'activity_code':activity,'leading_identifier':code,
                      'native_frames':lookup[record]['native_frames'],'in_original_24':record in retained})
    for n in retained:
        register(n)
    while len(picks)<96:
        def priority(group):
            code,activity,*_=group.split('_')
            return (activities[activity],people[code],pairs[code,activity],tie(group))
        grp=min((g for g in pool if g not in used),key=priority)
        register(min(pool[grp],key=tie))
    out={'created_from':'Frozen inventory identities and official split membership only; no data arrays, reference orientations, or scores were opened.',
         'selection_rule':'Keep the original 24. Repeatedly add one unused trial prefix by lowest acquisition-code count, '
                    'then lowest leading-identifier count, then lowest pair count. Use the stored SHA256 tie convention for '
                    'equal counts and for the one segment chosen within a prefix. Leading identifier is not a verified person ID.',
         'hmac_free_tie_prefix':'C019-source-diversity-20261123|','inventory_sha256':hashlib.sha256(inventory.read_bytes()).hexdigest(),
         'original_protocol_sha256':hashlib.sha256(previous.read_bytes()).hexdigest(),
         'eligible_records':len(eligible),'eligible_trial_groups':len(pool),'original_count':len(retained),
         'new_count':len(picks)-len(retained),'activity_counts':dict(activities),'leading_identifier_counts':dict(people),
         'intersection_with_development_and_official_test_trial_groups':sorted(used&forbidden),'records':picks,
         'status':'Prepared source panel. Optimizer participation must be established separately from actual steps and draws.'}
    p=ROOT/'review/yt_training_panel.json'
    p.write_text(json.dumps(out,indent=2)+'\n',encoding='utf8')
    print('prepared',len(picks),len(people),dict(activities),'forbidden overlap',len(used&forbidden))


if __name__=='__main__':
    main()
