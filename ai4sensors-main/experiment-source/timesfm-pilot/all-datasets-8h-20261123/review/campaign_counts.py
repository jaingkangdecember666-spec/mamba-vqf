"""Count actual small batches, unique physical/neural policies, fits and repeated control score cells."""
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
REVIEW=ROOT/'review'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def seconds_union(intervals):
    total=0.;prev_end=None
    for start,end in sorted(intervals):
        effective=start if prev_end is None else max(start,prev_end)
        total+=max(0.,(end-effective).total_seconds())
        prev_end=end if prev_end is None else max(prev_end,end)
    return total


def fit_durations(path):
    if not path.exists():return []
    def discover(node,where):
        if isinstance(node,list):
            return [row for i,v in enumerate(node) for row in discover(v,f'{where}[{i}]')]
        if not isinstance(node,dict):return []
        clocks=('fit_seconds','optimizer_seconds','seconds','fit_and_archive_seconds','training_and_archive_seconds')
        timing=next((key for key in clocks if isinstance(node.get(key),(float,int))),None)
        duration=node.get(timing)
        has_timer=isinstance(duration,(float,int))
        has_bounds='started_at' in node and 'completed_at' in node
        if (has_timer or has_bounds) and any(k in node for k in ('steps','optimizer_steps','accepted_steps')):
            if isinstance(duration,(float,int)):
                lo=hi=float(duration);basis=f'measured timer ({timing})'
            else:
                delta=(datetime.fromisoformat(node['completed_at'])-datetime.fromisoformat(node['started_at'])).total_seconds()
                lo=max(0.,delta-1.);hi=delta+1.;basis='wall timestamps with one-second resolution'
            return [{'field':where,'seconds_lower_bound':lo,'seconds_upper_bound':hi,'basis':basis}]
        return [row for key,v in node.items() for row in discover(v,f'{where}.{key}')]
    return discover(read(path),'training')


def job_intervals(batch_dir, start, end):
    """Identify interrupted intervals from evidence while preserving the named batch start and finish."""
    periods = [(start, end)]
    interruptions = []
    finalized=batch_dir/'recovery.json'
    if finalized.is_file():
        repair=read(finalized);failure=read(batch_dir/'failure.json')
        assert repair['original_pid']==failure['pid'] and repair['original_failure']==failure
        assert not repair['fit_repeated'] and repair['replay_rows']==0 and repair['verified_existing_development_rows']==213
        stopped=datetime.fromisoformat(failure['failed_at']);resumed=datetime.fromisoformat(repair['recovery_started_at'])
        assert start<=stopped<=resumed<=end and abs((datetime.fromisoformat(repair['recovery_completed_at'])-end).total_seconds())<1.
        periods=[(start,stopped),(resumed,end)]
        interruptions.append({'stage':'post_score_finalization','reason':failure['error'],'interruption_seconds':(resumed-stopped).total_seconds(),
                              'new_selection_on_retry':0,'replayed_records':0,'verified_existing_records':213,'evidence':str(finalized.relative_to(ROOT))})
    recovery = batch_dir/'replay-recovery.json'
    if recovery.exists():
        log = read(recovery)
        if 'original_process' in log:
            original=log['original_process']
            assert original['status']=='failed'
            first_start=datetime.fromisoformat(original['started_at'])
            first_end=datetime.fromisoformat(original['completed_at'])
            resumed=datetime.fromisoformat(log['recovery_started_at'])
            error=original['failure']
            repeated=log['selection_executions_in_recovery']
        else:
            failure_file=batch_dir/'failure.json'
            assert hashlib.sha256(failure_file.read_bytes()).hexdigest()==log.get('failure_sha256',log.get('original_failure_sha256'))
            original=read(failure_file)
            assert original['pid']==log.get('original_pid',log.get('original_fitting_pid'))
            first_start=datetime.fromisoformat(original['started_at'])
            first_end=datetime.fromisoformat(original['failed_at'])
            resumed=datetime.fromisoformat(log['started_at'])
            error=original['error']
            repeated=int(log['fit_repeated']) if 'fit_repeated' in log else int(log['optimizer_steps_repeated']!=0)
        assert abs((first_start-start).total_seconds())<1. and start <= first_end <= resumed <= end
        periods = [(first_start, first_end), (resumed, end)]
        interruptions.append({'stage':'post_lock_replay', 'reason':error,
                              'interruption_seconds':(resumed-first_end).total_seconds(),
                              'new_selection_on_retry':repeated,
                              'evidence':str(recovery.relative_to(ROOT))})
    for preflight in (batch_dir/'preparation-attempt'/'run.log',batch_dir/'preflight-failure.log'):
        if not preflight.exists():continue
        error = preflight.read_text(encoding='utf-8-sig').strip().splitlines()[-1]
        interruptions.append({'stage':'pre_optimizer_preparation','reason':error,
                              'evidence':str(preflight.relative_to(ROOT))})
    retry_file=batch_dir/'retry-launch.json'
    before=batch_dir/'preflight-attempt'/'failure.json'
    if before.is_file():
        failed=read(before)
        first=datetime.fromisoformat(failed['started_at']);stopped=datetime.fromisoformat(failed['failed_at'])
        assert first<=stopped<=start and not (before.parent/'router.pt').exists()
        assert failed['protocol_sha256']==hashlib.sha256((before.parent/'protocol.json').read_bytes()).hexdigest()
        periods.append((first,stopped))
        interruptions.append({'stage':'pre_optimizer_preparation','reason':failed['error'],'interruption_seconds':(start-stopped).total_seconds(),
                              'new_selection_on_retry':0,'evidence':str(before.relative_to(ROOT))})
    if retry_file.is_file():
        retry=read(retry_file)
        fail_file=batch_dir/'failure.json'
        failed=read(fail_file)
        assert hashlib.sha256(fail_file.read_bytes()).hexdigest()==retry['original_failure_sha256']
        assert failed['pid']==retry['original_preflight_pid'] and retry['previous_fit_count']==0
        first_start=datetime.fromisoformat(failed['started_at']);first_end=datetime.fromisoformat(failed['failed_at'])
        resumed=datetime.fromisoformat(retry['started_at'])
        assert first_start<first_end<=resumed<=end
        if not recovery.exists():
            periods=[(resumed,end)]
        periods.append((first_start,first_end))
        interruptions.append({'stage':'pre_optimizer_preparation','reason':failed['error'],
            'interruption_seconds':(resumed-first_end).total_seconds(),'new_selection_on_retry':0,
            'evidence':str(retry_file.relative_to(ROOT))})
    return periods, interruptions


def development_frontier(recipes, scores):
    """Use controls extended in later batches only when record, mask and policy identity match."""
    with (ROOT/'C006/sequences.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==213
    complete=[];partial=[]
    for entry in recipes:
        field=entry['signature']
        by_domain={}
        for row in rows:
            key=(row['dataset'],row['record'],row['scoring_indices_sha256'])
            if (*key,field) not in scores:
                continue
            ref=scores[*key,'default_full_vqf']
            assert abs(ref-float(row['full_vqf_deg']))<1e-5
            by_domain.setdefault(key[0],[]).append((ref,scores[*key,field]))
        n=sum(len(v) for v in by_domain.values())
        item={k:entry[k] for k in ['batch','column','label','signature','weights']}
        item['components']=entry.get('components',[])
        item['paired_record_count']=n
        if n<len(rows):
            partial.append(item)
            continue
        summary={}
        for d,vals in by_domain.items():
            arr=np.array(vals)
            initial,candidate=arr.mean(axis=0)
            summary[d]={'baseline_deg':float(initial),'candidate_deg':float(candidate),
                        'ratio':float(candidate/initial),'records':len(vals),
                        'improving_records':int((arr[:,1]<arr[:,0]).sum())}
        worst=max(summary,key=lambda d:summary[d]['ratio'])
        item.update(domains=summary,winning_domains=sum(v['ratio']<1. for v in summary.values()),
                    worst_domain=worst,worst_domain_ratio=summary[worst]['ratio'],
                    mean_domain_ratio=float(np.mean([v['ratio'] for v in summary.values()])))
        complete.append(item)
    complete.sort(key=lambda r:(-r['winning_domains'],r['worst_domain_ratio'],r['mean_domain_ratio'],r['batch'],r['column']))
    return {'rule':'Only the 213-record common development panel. More strictly improving domains first, then lower '
                   'worst candidate/default domain-mean ratio, then lower mean ratio. Record counts for missing '
                   'policies shown separately. A physical ablation has neither weights nor components.',
            'full_coverage':complete,'incomplete_coverage':partial}


@lru_cache(maxsize=None)
def model_fingerprint(path):
    cp=torch.load(path,map_location='cpu',weights_only=True)
    digest=hashlib.sha256()
    for k,v in sorted(cp['state_dict'].items()):
        a=v.detach().cpu().contiguous().numpy()
        digest.update(json.dumps([k,a.dtype.str,a.shape],separators=(',',':')).encode())
        digest.update(a.tobytes())
    if 'context_mean' in cp and 'p0' in cp:
        fields=('old_feature_mean','old_feature_std','context_mean','context_std','p0','log_p0')
        if 'geometry_mean' in cp:fields+=('geometry_mean','geometry_std')
    elif 'p0' in cp and 'log_p0' in cp:
        fields=('feature_mean','feature_std','p0','log_p0')
    elif 'B6_input_mean' in cp:
        from temporal_readout import trained_layout
        *_,mean_key,std_key=trained_layout(cp)
        fields=('B6_input_mean','B6_input_std',mean_key,std_key)
    else:
        fields=('feature_mean','feature_std') if 'feature_mean' in cp else ('mean','std')
    for key in fields:
        a=cp[key].cpu().contiguous().numpy()
        digest.update(json.dumps([key,a.dtype.str,a.shape],separators=(',',':')).encode())
        digest.update(a.tobytes())
    if 'context_mean' in cp and 'p0' in cp:
        digest.update(json.dumps(cp['feature_names'],separators=(',',':')).encode())
        if 'geometry_names' in cp:digest.update(json.dumps(cp['geometry_names'],separators=(',',':')).encode())
    elif 'p0' in cp and 'log_p0' in cp:
        digest.update(b'normalized_B6_history_fast_slow_original_softmax')
    elif 'B6_input_mean' in cp:
        arch=cp.get('architecture',cp.get('configuration'))
        digest.update(json.dumps(arch,sort_keys=True,separators=(',',':')).encode())
    elif 'feature_mean' in cp:
        digest.update(cp['feature'].encode())
    else:
        digest.update(str(float(cp['cap_deg'])).encode())
    return digest.hexdigest()


def main():
    recipes=read(REVIEW/'candidate_recipes.json')
    all_recipes={}
    for r in recipes:
        weights=model_fingerprint(ROOT/r['weights']) if r.get('weights') else None
        policy={k:v for k,v in r.items() if k not in ('batch','column','weights','label','new_fit','fit_kind','components','repeat_of')}
        if 'components' in r:
            policy['components']=[(all_recipes[c['batch'],c['column']]['signature'],c['fraction']) for c in r['components']]
        policy['network_and_normalization_sha256']=weights
        fingerprint=hashlib.sha256(json.dumps(policy,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        parent=r.get('repeat_of')
        if parent:
            original=all_recipes[parent['batch'],parent['column']]
            assert original['signature']==fingerprint and not r['new_fit'], (r['batch'],parent)
        elif any(previous['signature']==fingerprint for previous in all_recipes.values()):
            raise ValueError(f"Identical operating policy needs a verified repeat reference in {r['batch']}")
        all_recipes[r['batch'],r['column']]={**r,'signature':fingerprint,'model_signature':weights}
    aliases={'b6_full_deg':('C001','b6_intrinsic_deg'),'b6_gain_025_deg':('C002','b6_gain_025_deg'),
             'debiased_full_vqf_deg':('C003','debiased_full_vqf_deg'),'front_end_vqf_deg':('C003','debiased_full_vqf_deg'),
             'c003_deg':('C003','debiased_b6_gain_025_deg'),'B_fixed_deg':('C004','B_deg'),
             'B_gated_deg':('C005','B_gated_deg'),'B_rest_gated_deg':('C006','B_rest_gated_deg'),
             'C006_deg':('C006','B_rest_gated_deg'),'seed_C003_deg':('C013','seed_C003_deg'),
             'seed_C006_deg':('C013','seed_C006_deg'),'C014_B_deg':('C014','B_deg'),'C018_B_deg':('C018','B_deg')}
    aliases.update({r['column']:(r['batch'],r['column']) for r in recipes if r['column'].startswith('C0')})
    batches=[];intervals=[];seen={};fitted=set();attempt_errors=[]
    for batch in sorted({r['batch'] for r in recipes}):
        path=ROOT/batch;result=read(path/'results.json')
        if 'started_at' in result:
            start=datetime.fromisoformat(result['started_at'])
        else:
            receipt=read(path/'process.json')
            assert receipt['status']=='complete' and receipt['pid']==result['pid']
            assert receipt['results_sha256']==hashlib.sha256((path/'results.json').read_bytes()).hexdigest()
            start=datetime.fromisoformat(receipt['started_at'])
        end=datetime.fromisoformat(result['completed_at'])
        periods, errors = job_intervals(path, start, end)
        intervals.extend(periods)
        attempt_errors.extend({'batch':batch, **err} for err in errors)
        successes=failures=again=extra=duplicate_aliases=0
        method_cells={}
        active_recipe=all_recipes.get((batch,batch+'_deg'),{})
        uses_c022_physical=any(active_recipe.get(k) for k in ('response_router','raw_anchor_blend','physical_angle_cap')) or active_recipe.get('repeat_of')=={'batch':'C022','column':'physical_only_deg'}
        with (path/'sequences.csv').open(encoding='utf-8-sig',newline='') as file:
            for row in csv.DictReader(file):
                if row.get('status') in ('failed','error') or not row.get('full_vqf_deg'):
                    failures+=1
                    continue
                successes+=1
                in_row={}
                for key,value in row.items():
                    if key in ('full_vqf_deg','fixed_tau3_deg'):sign='default_full_vqf'
                    elif key in ('fixed_tau1p5_deg','fixed_tau6_deg'):sign='stock_vqf_'+key
                    elif (batch,key) in all_recipes:sign=all_recipes[batch,key]['signature']
                    elif key in aliases:sign=all_recipes[aliases[key]]['signature']
                    elif key=='physical_only_deg' and uses_c022_physical:
                        sign=all_recipes['C022','physical_only_deg']['signature']
                    else:continue
                    token=(row['dataset'],row['record'],row['scoring_indices_sha256'],sign)
                    x=float(value)
                    if not np.isfinite(x):raise ValueError((batch,row['record'],key))
                    if sign in in_row:
                        assert abs(x-in_row[sign])<1e-5,(batch,row['record'],key)
                        duplicate_aliases+=1
                        continue
                    in_row[sign]=x
                    control=method_cells.setdefault(sign,{'column':key,'previous_cells':0,'first_cells':0})
                    if token in seen:
                        again+=1
                        control['previous_cells']+=1
                        if abs(x-seen[token])>1.e-4:raise ValueError(('control changed',batch,row['record'],key))
                    else:
                        extra+=1
                        control['first_cells']+=1
                    seen[token]=x
        log_file=path/'training-steps.csv'
        steps=Counter()
        if log_file.exists():
            with log_file.open(encoding='utf-8-sig',newline='') as file:
                for x in csv.DictReader(file):steps[x.get('arm',batch)]+=1
        elif (path/'iterations.jsonl').exists():
            attempts=[json.loads(x) for x in (path/'iterations.jsonl').read_text(encoding='utf8').splitlines() if x]
            accepted=[r for r in attempts if r.get('parameter_sha256_after') and r.get('accepted_step')]
            fit=read(path/'training.json')['fit']
            assert len(accepted)==fit['accepted_steps'] and len(attempts)==fit['attempts']
            steps[batch]=len(accepted)
        else:
            for step_file in sorted(path.glob('*-steps.jsonl')):
                if step_file.name=='sampling-steps.jsonl':
                    continue
                history=[json.loads(s) for s in step_file.read_text(encoding='utf8').splitlines() if s]
                arm=history[0].get('arm',batch)
                assert all(s.get('arm',batch)==arm for s in history) and [s['step'] for s in history]==list(range(1,len(history)+1))
                assert arm not in steps
                steps[arm]=len(history)
        for (b,_),r in all_recipes.items():
            if b==batch and r['new_fit'] and r['model_signature']:fitted.add(r['model_signature'])
        batches.append({'batch':batch,'started_at':start.isoformat(),'completed_at':end.isoformat(),
                        'job_wall_seconds':(end-start).total_seconds(),'documented_process_seconds':seconds_union(periods),
                        'record_rows':successes+failures,
                        'scored_records':successes,'record_failures':failures,'result_failures':len(result.get('failures',[])),
                        'score_cells_seen_in_earlier_batches':again,'new_score_cells':extra,'same_row_alias_cells':duplicate_aliases,'optimizer_steps_by_arm':dict(steps),
                        'per_method_score_cells':method_cells,
                        'logged_development_scoring_seconds':result.get('development_replay_seconds',result.get('scoring_seconds',result.get('replay_seconds'))),
                        'verified_fit_timing':fit_durations(path/'training.json')})
    cfg=read(ROOT/'protocol.json');last=batches[-1]['completed_at']
    events=[json.loads(x) for x in (REVIEW/'decisions.jsonl').read_text(encoding='utf-8').splitlines() if x]
    source_solvers={}
    neural_solvers={}
    scalar_solvers={}
    for b in batches:
        path=ROOT/b['batch']/'solver-receipt.json'
        if path.exists():
            receipt=read(path)
            summary={k:receipt[k] for k in ('method','solve_count','nit','nfev','njev','result_success','solve_and_archive_seconds') if k in receipt}
            assert all(k in summary for k in ('method','solve_count','nit'))
            summary['locked_choice']=receipt.get('locked_choice',receipt.get('locked_beta',receipt.get('locked_a')))
            if receipt.get('neural_fits',0):
                assert receipt['neural_fits']==len(b['optimizer_steps_by_arm']) and receipt['solver_kind']=='neural_readout'
                neural_solvers[b['batch']]=summary
            elif receipt.get('parameters')==1:
                assert (path.parent/'coefficient-lock.json').is_file()
                scalar_solvers[b['batch']]=summary
            else:
                source_solvers[b['batch']]=summary
    report={'through_batch':batches[-1]['batch'],'evidence_cutoff':last,'authorized_start':cfg['start'],'authorized_end':cfg['end'],
            'completed_named_batches':len(batches),'new_or_inherited_candidate_policies_tested':len({r['signature'] for r in all_recipes.values()}),
            'whole_candidate_replay_batches':sorted({r['batch'] for r in all_recipes.values() if r.get('repeat_of')}),
            'batches_scoring_complete_213_development_panel':sum(r['scored_records']==213 and not r['record_failures'] for r in batches),
            'default_reference_policies':1,'physical_ablation_policies':len({r['signature']for r in all_recipes.values() if r['weights'] is None and not r.get('components')}),
            'additional_fixed_vqf_benchmarks':2 if (REVIEW/'vqf_response_controls_development.json').is_file() else 0,
            'different_neural_parameter_sets_including_inherited':len({r['model_signature'] for r in all_recipes.values()
                if r.get('fit_kind')!='external_physical_coefficients'}-{None}),
            'newly_fitted_distinct_neural_parameter_sets':len(fitted),
            'external_scalar_fits':sum((ROOT/r['batch']/'coefficient-lock.json').exists() or
                                       (ROOT/r['batch']/'calibration.json').is_file() for r in batches),
            'external_multicoefficient_solves':sum(s['solve_count'] for s in source_solvers.values()),
            'external_multicoefficient_iterations':sum(s['nit'] for s in source_solvers.values()),
            'external_multicoefficient_solver_receipts':source_solvers,
            'neural_constrained_solver_receipts':neural_solvers,
            'single_physical_solver_receipts':scalar_solvers,
            'averaged_checkpoints_without_new_training':sum(r.get('fit_kind')=='weight_average' for r in all_recipes.values()),
            'optimizer_run_count':sum(len(r['optimizer_steps_by_arm']) for r in batches),
            'optimizer_steps':sum(sum(r['optimizer_steps_by_arm'].values()) for r in batches),
            'fits_with_logged_timer_or_boundary_timestamps':sum(len(r['verified_fit_timing']) for r in batches),
            'logged_optimizer_wall_seconds_range':[sum(v[k] for r in batches for v in r['verified_fit_timing'])
                for k in ('seconds_lower_bound','seconds_upper_bound')],
            'scored_record_by_batch_entries':sum(r['scored_records'] for r in batches),
            'record_failures':sum(r['record_failures'] for r in batches),
            'documented_solver_search_misses':(['C036'] if batches[-1]['batch']>='C036' and (REVIEW/'C036_solver_gap.json').is_file() and read(REVIEW/'C036_solver_gap.json')['strictly_better_nearby_on_external_objective'] else []),
            'documented_execution_errors':attempt_errors,
            'successful_post_lock_evaluation_restarts':sum(r['stage']=='post_lock_replay' for r in attempt_errors),
            'recoveries_existing_scores_only':sum(r['stage']=='post_score_finalization' for r in attempt_errors),
            'fully_previously_scored_method_columns':sum(m['previous_cells']>0 and m['first_cells']==0
                     for r in batches for m in r['per_method_score_cells'].values()),
            'partly_previously_scored_method_columns':sum(m['previous_cells']>0 and m['first_cells']>0
                     for r in batches for m in r['per_method_score_cells'].values()),
            'relisted_score_cells':sum(r['score_cells_seen_in_earlier_batches'] for r in batches),
            'same_row_alias_cells_excluded_from_repeats':sum(r['same_row_alias_cells'] for r in batches),
            'batches_with_development_scoring_timer':sum(r['logged_development_scoring_seconds'] is not None for r in batches),
            'logged_development_scoring_seconds':sum(r['logged_development_scoring_seconds'] or 0 for r in batches),
            'quantities':'Repeated cells have identical method state, raw sequence and native scoring indices. They include copied '
                       'and replayed controls; they are not additional fits. Job interval union excludes preparation before the '
                       'declared start, independent reviewer work, human/agent analysis and failed attempts without a known end.',
            'archived_score_pairs_reused_at_start':sum(r['origin']=='reused_b6_raw_replay_and_mask' for r in read(ROOT/'C001/results.json')['sequences']),
            'completed_job_interval_union_s':seconds_union(intervals),
            'elapsed_from_authorization_to_last_result_s':(datetime.fromisoformat(last)-datetime.fromisoformat(cfg['start'])).total_seconds(),
            'supplemental_decision_ids':[v['id'] for v in events if v['id'] not in {r['batch'] for r in batches}],
            'batches':batches,'candidates':list(all_recipes.values())}
    (REVIEW/'campaign-counts.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    panel=development_frontier([r for r in all_recipes.values() if not r.get('repeat_of')],seen)
    (REVIEW/'development_frontier.json').write_text(json.dumps(panel,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print({k:v for k,v in report.items() if isinstance(v,(int,float))})


if __name__=='__main__':
    main()
