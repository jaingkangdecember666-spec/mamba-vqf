"""Describe the current development winner, or explicitly seal it and the formal evaluation population."""
import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT=Path(__file__).resolve().parent.parent
REVIEW=ROOT/'review'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def digest(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def inference(key,recipes):
    heads=[]
    additions=[]
    coeffs=[]
    response_banks=[]
    routers=[]
    raw_anchor_gates=[]
    physical_caps=[]
    def expand(which,scale):
        r=recipes[which]
        if r['front_end']!='causal_sphere':
            raise ValueError(f'Unimplemented replay front end for {which}')
        if r.get('stable_force'):
            coeffs.append((scale,r['stable_force']))
        if r.get('response_bank'):
            response_banks.append((scale,r['response_bank']))
        if r.get('raw_anchor_blend'):
            assert scale==1. and len(r.get('components',[]))==1
            raw_anchor_gates.append(r['raw_anchor_blend'])
        if r.get('physical_angle_cap'):
            assert scale==1. and len(r.get('components',[]))==1
            physical_caps.append(r['physical_angle_cap'])
        for c in r.get('components',[]):
            expand((c['batch'],c['column']),scale*c['fraction'])
        if r.get('weights'):
            p=r['weights']
            name=r['batch']+'_'+r['column'].removesuffix('_deg')
            entry={'name':name,'weights':p,'sha256':digest(ROOT/p),'gain':r['gain'],'gate':r['gate']}
            if 'history_scale' in r:
                if r['history_scale']!=4 or not (r.get('additive_encoder') or r.get('backbone_weights')):
                    raise ValueError('Only additive temporal heads implement 40 ms input blocks')
                entry['history_scale']=r['history_scale']
            if r.get('response_router'):
                base=[h for h in heads if h['weights']==r['response_router']['backbone_weights']]
                if scale!=1. or len(base)!=1:
                    raise ValueError('The response router needs its original B6 backbone')
                rule=dict(entry,kind='history_softmax',backbone=base[0]['name'],prior_gain=r['response_router']['prior_gain'])
                if 'context' in r['response_router']:
                    context=r['response_router']['context']
                    if context not in ('native_motion_2s','motion_and_response_geometry','anchored_response_physics'):raise ValueError(f'Unknown router observations {context}')
                    rule['context']=context
                routers.append(rule)
            elif r.get('additive_encoder'):
                if scale!=1.:
                    raise ValueError('The complete additive encoder requires its registered gain')
                additions.append(dict(entry,kind='trained_mamba_temporal_readout'))
            elif r.get('backbone_weights'):
                if scale!=1.:
                    raise ValueError('This additive temporal branch needs a separately specified gain')
                sources=[s for s in heads if s['weights']==r['backbone_weights']]
                if len(sources)!=1:
                    raise ValueError('Temporal readout needs an unambiguous frozen Mamba backbone')
                additions.append(dict(entry,kind='frozen_mamba_temporal_readout',backbone=sources[0]['name']))
            else:
                if r['gate'] not in ('always','rest_off') or 'force_ema_gain' in r:
                    raise ValueError(f'Explicitly implement the deployed recipe before sealing {which}')
                heads.append(dict(entry,kind='mamba_v1',fraction=scale))
    expand(key,1.)
    if not heads or abs(sum(s['fraction'] for s in heads)-1.)>1e-8:
        raise ValueError('Mamba component fractions differ from a complete primary predictor')
    spec={'front_end':'causal_sphere','heads':heads,'additions':additions,
          'window_blocks':64,'block_seconds':.01,'release_seconds':.1,'vertical_in_estimator_frame':[0,0,1]}
    if coeffs:
        if len(coeffs)!=1 or coeffs[0][0]!=1.:
            raise ValueError('This combination of physical components requires an explicit replay')
        spec['stable_force']=coeffs[0][1]
    if response_banks:
        if len(response_banks)!=1 or response_banks[0][0]!=1.:
            raise ValueError('Expected a single full-scale gravity response bank')
        spec['response_bank']=response_banks[0][1]
    if routers:
        if len(routers)!=1 or response_banks:
            raise ValueError('The response router replaces the fixed response, with one global prior')
        spec['response_router']=routers[0]
    if raw_anchor_gates:
        assert len(raw_anchor_gates)==1 and len(raw_anchor_gates[0]['beta'])==4
        gate=raw_anchor_gates[0]
        assert digest(ROOT/gate['receipt'])==gate['receipt_sha256']
        spec['raw_anchor_blend']=gate
    if physical_caps:
        assert len(physical_caps)==1 and not raw_anchor_gates
        cap=physical_caps[0]
        assert 0.<=cap['degrees']<=180. and digest(ROOT/cap['receipt'])==cap['receipt_sha256']
        spec['physical_angle_cap']=cap
    return spec


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seal',action='store_true',help='Write the sole final selection, before any sealed numerical evaluation.')
    args=ap.parse_args()
    path=ROOT/'selection-lock.json'
    if path.exists():
        old=read(path)
        print('Already sealed',old['selected']['batch'],old['selected']['column'],old['locked_at'],digest(path))
        return
    counts=read(REVIEW/'campaign-counts.json')
    frontier=read(REVIEW/'development_frontier.json')
    recipes={(a['batch'],a['column']):a for a in read(REVIEW/'candidate_recipes.json')}
    primary=next(r for r in frontier['full_coverage'] if r['weights'] or r['components'])
    key=primary['batch'],primary['column']
    spec=inference(key,recipes)
    inv=ROOT/'C001/inventory.json'
    rows=read(inv)['records']
    formal=[r for r in rows if r['side'].startswith('sealed') or r['side']=='development_previously_used_for_selection']
    expected=read(REVIEW/'final-anchor-spec.json')['records_in_main']
    assert dict(Counter(r['dataset'] for r in formal))==expected and len(formal)==365
    coverage=[r for r in rows if (r['dataset']=='SHICE' and r['side']=='development') or
              (r['dataset'] in ('BROAD','Sassari') and r['side']=='historical_source_train')]
    assert len(coverage)==76
    def identity(r):
        return {k:r[k] for k in ('dataset','record','group','side')}
    sources=['protocol.json','review/candidate_recipes.json','review/development_frontier.json','review/campaign-counts.json',
             'review/FINAL_PROTOCOL.md','review/final-anchor-spec.json','review/vehicle-reference-protocol.json',
             'review/native_catalog.py','review/native_blocks.py','review/optical_reference.py','review/vehicle_reference.py',
             'review/fixture_level.py','review/locked_replay.py','review/mixture_replay.py','review/temporal_readout.py',
             'review/stable_force_source.py','review/multirate_blocks.py','review/filter_response_bank.py','review/lock_from_frontier.py','external-sources/sealed-transfer-tests/protocol.json',
             'external-sources/sealed-transfer-tests/source-files.json','review/supplemental_data.py','review/score_supplemental.py','review/oxford_sensitivity.py',
             'review/tum_reference_quality.py','review/final_geometry.py','review/mount_prediction_audit.py','review/vqf_response_controls.py','review/response_router.py','review/check_locked_causality.py',
             'C003/online_bias.py','review/optical_score.py','../transfer-mamba-20261123/models.py',
             '../transfer-mamba-20261123/review/replay_native.py']
    router_context=spec.get('response_router',{}).get('context')
    if router_context:
        sources+=['review/router_motion_context.py','review/source_motion_router_diagnostic.json']
    if router_context in ('motion_and_response_geometry','anchored_response_physics'):
        sources+=['review/causal_response_geometry.py','review/geometry_source_preflight.json']
    if router_context=='anchored_response_physics':
        sources+=['review/anchored_response_features.py','review/anchored_response_gradient.json']
    if 'raw_anchor_blend' in spec:
        sources+=['review/anchor_blend_features.py','review/raw_vqf_blend.py','review/anchor_blend_risk.py',
                  'review/raw-anchor-training-141.json',spec['raw_anchor_blend']['receipt']]
    if 'physical_angle_cap' in spec:
        sources+=['review/physical_angle_cap.py','review/external-physical-anchor-141.json',spec['physical_angle_cap']['receipt']]
    if (ROOT/'external-sources/davis-handheld/preprocessing-lock.json').is_file():
        sources+=['external-sources/davis-handheld/protocol.json','external-sources/davis-handheld/source-files.json',
                  'external-sources/davis-handheld/preprocessing-lock.json','review/davis_handheld.py','review/score_davis_transfer.py']
    if (ROOT/'external-sources/penn-handheld/protocol.json').is_file():
        sources+=['external-sources/penn-handheld/protocol.json']
        if (ROOT/'external-sources/penn-handheld/preprocessing-lock.json').is_file():
            sources+=['external-sources/penn-handheld/source-files.json','external-sources/penn-handheld/preprocessing-lock.json',
                      'review/penn_handheld.py','review/score_penn_transfer.py']
    independent=REVIEW/f'development_replay_{key[0]}_{key[1]}.json'
    if key[0]=='C038':
        audit=read(REVIEW/'C038_audit.json')
        assert audit['paired_development_records']==213 and len(audit['native_development_rows'])==4
        assert audit['parameter_max_abs_gap']==0 and audit['checkpoint_hashes'][-1]==digest(ROOT/'C038/router.pt')
        sources+=['review/C038_audit.json','review/check_midpoint.py','review/C023_C035_development_shift.json','review/C037_certificate_check.json','review/decisions.jsonl']
    if independent.is_file():
        report=read(independent)
        assert report['n_records']==213 and report['source_csv_sha256']==digest(ROOT/key[0]/'sequences.csv')
        assert report['max_abs_gap_deg']<2e-4
        sources.append(independent.relative_to(ROOT).as_posix())
    for b in counts['batches']:
        for fn in ('protocol.json','results.json','sequences.csv','training.json','coefficient-lock.json','pair-lock.json',
                   'router-lock.json','sampling-lock.json','branch-lock.json','geometry-lock.json','solver-receipt.json',
                   'finite-difference.json','synthesis-lock.json','synthesis-verification.json'):
            rel=b['batch']+'/'+fn
            if (ROOT/rel).is_file():
                sources.append(rel)
    locked={'locked_at':datetime.now().astimezone().isoformat(),'campaign_window':read(ROOT/'protocol.json'),
            'selection_basis':frontier['rule'],'inventory_sha256':digest(inv),'selected':primary,
            'development_through':counts['through_batch'],'candidate_ranking_sha256':digest(REVIEW/'development_frontier.json'),
            'review_inference':spec,'evidence_sha256':{q:digest(ROOT/q) for q in sources},
            'formal_cases':[identity(r) for r in formal],'full_coverage_additional_cases':[identity(r) for r in coverage],
            'secondary':'Same frozen front end and physical terms, with all learned tangents zero. No final result used to choose a model.',
            'metric':'Native frames shared with FullVQF; record tilt-angle RMSE in degrees, arithmetic mean across records within domain.'}
    if args.seal:
        with NamedTemporaryFile(mode='w',dir=path.parent,suffix='.lock-stage',delete=False,encoding='utf8') as f:
            staged=Path(f.name)
            json.dump(locked,f,ensure_ascii=False,indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(staged,path)
        finally:
            staged.unlink()
        print('Sealed',path,digest(path))
    print('Development winner',key,'wins',primary['winning_domains'],'worst relative ratio',primary['worst_domain_ratio'],
          'base heads',len(spec['heads']),'additions',len(spec['additions']),'main',len(formal),'extra',len(coverage))


if __name__=='__main__':
    main()
