"""Verify collection identities across sides using the frozen inventory, without opening any held-out sensor data."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent


def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    source=ROOT/'C001/inventory.json'
    inventory=read(source)['records']
    trained=read(ROOT/'C004/protocol.json')['source_records']
    by_domain=defaultdict(list)
    for r in inventory:by_domain[r['dataset']].append(r)
    summary={}
    for domain,rows in by_domain.items():
        dev=[r for r in rows if r['side'].startswith('development')]
        test=[r for r in rows if r['side'].startswith('sealed')]
        if domain in ('BROAD','Sassari'):test=dev.copy()
        train=[r for r in trained if r['domain']==domain]
        side={'training_used_C004B':train,'development':dev,'formal':test}
        group={k:{r['group'] for r in v} for k,v in side.items()}
        out={'records':{k:len(v) for k,v in side.items()},'independent_collection_keys':{k:len(v) for k,v in group.items()},
             'training_and_formal_same_collection_keys':sorted(group['training_used_C004B']&group['formal']),
             'development_and_formal_same_collection_keys':sorted(group['development']&group['formal'])}
        if domain=='YT_Server':
            actors={k:{r['record'].split('_')[0] for r in v} for k,v in side.items()}
            out['leading_tokens']={k:sorted(v) for k,v in actors.items()}
            out['leading_token_train_formal_overlap']=sorted(actors['training_used_C004B']&actors['formal'])
            out['leading_token_development_formal_overlap']=sorted(actors['development']&actors['formal'])
            out['scope']='The initial token is a filename grouping key, not a verified personal identity. The protocol isolates trials by prefix; it does not isolate leading tokens.'
        summary[domain]=out
    output={'scope':'Names only, before the selection lock; formal BROAD and Sassari reuse development benchmarks.',
            'inventory_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'domains':summary}
    (ROOT/'review/split_identity_audit.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf8')
    print([(d,s['records'],len(s['training_and_formal_same_collection_keys'])) for d,s in summary.items()])


if __name__=='__main__':
    main()
