"""Read the final CSVs independently and confirm the delivered desktop/mobile document contains the same figures."""
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import fmean
from urllib.parse import unquote

from lock_from_frontier import ROOT, REVIEW, digest, read


def entries(path):
    with path.open(encoding='utf-8-sig',newline='') as source:
        return list(csv.DictReader(source))


def paired(rows):
    assert len(rows)==len({r['record']for r in rows})
    scored=[];failed=[]
    for r in rows:
        if not r.get('candidate_deg'):
            assert r.get('status') and r['status']!='scored'
            failed.append({'record':r['record'],'status':r['status'],'detail':r.get('detail') or r.get('error')})
            continue
        assert len(r['scoring_indices_sha256'])==64 and int(r['scored_frames'])>0
        assert all(math.isfinite(float(r[k])) and float(r[k])>=0. for k in ('candidate_deg','full_vqf_deg','physical_deg'))
        scored.append(r)
    out={'requested':len(rows),'scored':len(scored),'unscored':failed}
    if scored:
        raw,model=[fmean(float(r[k]) for r in scored)for k in ('full_vqf_deg','candidate_deg')]
        assert raw>0.
        out.update(raw_deg=raw,model_deg=model,delta_deg=model-raw,gain_percent=100*(1.-model/raw),
                   records_improved=sum(float(r['candidate_deg'])<float(r['full_vqf_deg']) for r in scored),
                   scored_frames=sum(int(r['scored_frames']) for r in scored))
    return out


def main():
    lock=ROOT/'selection-lock.json';plan=read(lock);sha=digest(lock)
    evidence=read(REVIEW/'final-verification.json')
    assert evidence['lock_sha256']==sha
    alternate=read(REVIEW/'final-alternate-check.json')
    assert alternate['selection_lock_sha256']==sha and alternate['checked_records']==14 and alternate['supplemental_checked']==31
    assert alternate['max_main_gap_deg']<2e-4
    all_native=read(REVIEW/'all-native-verification.json')
    assert all_native['lock_sha256']==sha and all_native['requested']==474 and all_native['verified']==all_native['scored']==472
    assert all_native['max_absolute_difference_deg']<2e-4 and len(all_native['per_record_hashes'])==474
    for name,checksum in all_native['per_record_hashes'].items():assert digest(REVIEW/'full-native-verification'/name)==checksum
    for path,h in plan['evidence_sha256'].items():assert digest(ROOT/path)==h,path
    main_path=ROOT/evidence['main_table']; additional_path=ROOT/evidence['additional_table']
    main_rows=entries(main_path);additional=entries(additional_path)
    assert len(main_rows)==365 and len(additional)==76
    def ids(rows):return {(r['dataset'],r['record'],r['group'])for r in rows}
    assert ids(main_rows)==ids(plan['formal_cases']) and ids(additional)==ids(plan['full_coverage_additional_cases'])
    assert not (ids(main_rows)&ids(additional))
    domain={}
    for key in sorted({r['dataset']for r in main_rows}):
        item=paired([r for r in main_rows if r['dataset']==key])
        stored=evidence['main'][key]
        assert item['requested']==stored['requested'] and item['scored']==stored['scored']
        if item['scored']:assert abs(item['gain_percent']-stored['improvement_percent'])<1e-9
        domain[key]=dict(scope='primary',**item)
    other_files=[REVIEW/'final-supplemental/sequences.csv',REVIEW/'final-davis/sequences.csv',REVIEW/'final-penn/sequences.csv']
    others=[r for path in other_files for r in entries(path)]
    requested={'EuRoC':3,'RIDI':25,'DAVIS':3,'PennCOSYVIO':2}
    assert set(requested)=={r['dataset']for r in others}
    for key,n in requested.items():
        item=paired([r for r in others if r['dataset']==key]);assert item['requested']==n
        domain[key]=dict(scope='supplemental',**item)
    assert len(domain)==12
    marked=read(REVIEW/'report-addendum.json')['queries']
    summary_table=marked['all8-final-cross-source']['rows']
    assert len(summary_table)==len(domain)
    for row in summary_table:
        key=row['c0'];item=domain[key]
        assert row['c2']==f"{item['scored']}/{item['requested']}"
        if item['scored']:
            assert row['c3']==f"{item['raw_deg']:.5f}°" and row['c4']==f"{item['model_deg']:.5f}°"
            assert row['c6']==f"{item['gain_percent']:+.2f}%"
            assert abs(row['gain_fraction']-item['gain_percent']/100)<1e-12
    timing=read(REVIEW/'delivery-timing.json')
    assert not timing['pending_named_outputs'],timing['pending_named_outputs']
    old=ROOT.parent/'overnight-20261122/review'
    html=old/'report-src/dist/index.html'; qa=read(old/'render_check.json')
    assert qa['html_sha256']==digest(html) and qa['report_status']=='complete'
    release=read(REVIEW/'published-report-qa.json')
    assert release['html_sha256']==digest(REVIEW/'REPORT.html')==qa['html_sha256']
    assert release['source_render_sha256']==digest(old/'render_check.json')
    assert all(v['offline'] and not v['network_requests'] and not v['page_errors'] and v['gain_chart_domains']==12
               for v in release['viewports'].values()) and set(release['viewports'])=={'390','1280'}
    qtime=datetime.fromisoformat(qa['reviewed_utc'])
    assert qtime>=datetime.fromisoformat(timing['evidence_checked_through_at'])>=datetime.fromisoformat(timing['science_completed_at'])
    for view in ('390','1280'):
        v=qa['viewports_css_px'][view]
        assert not v['document_overflow'] and not v['page_errors'] and v['verified_section_links']>=15
        assert v['gain_chart_marks']==sum(bool(item['scored'])for item in domain.values())
    desk=read(old/'report-src/src/data.json')['queries']
    assert summary_table==desk['all8-final-cross-source']['rows']
    guidance=read(REVIEW/'guidance-verification.json')
    assert guidance['lock_sha256']==sha and guidance['separate_batch_directives_received']==38
    assert guidance['receipt_csv_sha256']==digest(REVIEW/'guidance-receipts.csv')
    assert guidance['after_previous_summary']+guidance['after_previous_full_csv_before_summary']==37
    assert marked['all8-supervision']['rows']==desk['all8-supervision']['rows']
    link_counts={}
    for filename in ('REPORT.md','SPLITS.md','README.md'):
        links=re.findall(r'\]\(([^)]+)\)',(REVIEW/filename).read_text(encoding='utf8'))
        local=[unquote(x.split('#')[0].strip('<>')) for x in links if not x.startswith(('http:','https:','#'))]
        assert all((REVIEW/name).exists() for name in local),filename
        link_counts[filename]=len(local)
    for key,query in marked.items():
        assert all((old/path).exists() for path in query['source']['files']),key
    legacy=read(REVIEW/'historical-models-verified.json')
    assert legacy['selection_sha256']==sha and legacy['requested']==398 and legacy['scored_native_rows']==396
    assert legacy['native_anchor_count']==18 and len(legacy['independent_anchors'])==18 and legacy['original_archive_records']==20
    assert set(domain)==set(legacy['domains']) and sum(legacy['worker_status_counts'].values())==398
    comparison=marked['all8-frozen-external-models']['rows']
    assert comparison==desk['all8-frozen-external-models']['rows'] and len(comparison)==12
    for row in comparison:
        dataset=row['c0'];stat=legacy['domains'][dataset];means=stat['means_deg']
        assert stat['scored']==domain[dataset]['scored'] and stat['requested']==domain[dataset]['requested']
        assert abs(means['candidate_deg']-domain[dataset]['model_deg'])<1e-9
        assert abs(means['full_vqf_deg']-domain[dataset]['raw_deg'])<1e-9
        assert row['c1']==f"{stat['scored']}/{stat['requested']}"
        for i,key in enumerate(('full_vqf_deg','transfer_mamba_deg','strict_tcn_deg','candidate_deg'),2):
            assert row[f'c{i}']==f"{means[key]:.5f}°"
    for path,checksum in legacy['source_hashes'].items():assert digest(ROOT/path)==checksum
    assert legacy['tcn_backend']['original_backend_matching']==legacy['tcn_backend']['first_run_flags']==19
    clock=read(REVIEW/'shice-clock-verified.json')
    assert clock['selection_sha256']==sha and clock['worker_summary_sha256']==digest(ROOT/'final/shice-clock-sensitivity/summary.json')
    assert clock['reconstructed_masks']==20 and clock['raw_anchor_comparisons']==30 and clock['raw_anchor_max_gap_deg']<2e-4
    clock_table=marked['all8-self-encoder-clock']['rows']
    assert clock_table==desk['all8-self-encoder-clock']['rows'] and len(clock_table)==15
    for row in clock_table:
        group=clock['groups'][row['c0']];score=group['offsets'][str(int(row['c1']))]
        assert row['c2']==str(group['records']) and row['c6']==f"{score['gain_default_pct']:+.3f}%"
        for i,field in enumerate(('vqf3_deg','vqf1p5_deg','c038_deg'),3):
            assert row[f'c{i}']==f"{score['means'][field]:.5f}°"
    catalog=read(ROOT/'C001/inventory.json')['records']
    self_names={r['record'] for r in main_rows if r['dataset']=='SHICE'}
    raw_paths={r['record']:Path(r['path']) for r in catalog if r['dataset']=='SHICE' and r['record'] in self_names}
    assert len(raw_paths)==len(self_names)==20
    raw_sources={}
    for name,path in raw_paths.items():
        proof=read(ROOT/'final/shice-clock-sensitivity/receipts'/f'{name}.json')
        masks_path=path.parents[1]/'Label_quality_audit_20261116'/name/'screening_masks.npz'
        for file,field in ((path,'raw_imu_encoder_csv_sha256'),(masks_path,'screening_mask_sha256')):
            assert digest(file)==proof[field],file
            raw_sources[str(file)]={'bytes':file.stat().st_size,'sha256':proof[field]}
    scores=read(REVIEW/'campaign-counts.json')
    controls=read(REVIEW/'vqf_response_controls_final.json')
    assert controls['selection_lock_sha256']==sha
    strict=[]
    for d,x in controls['domains'].items():
        m=x.get('means',{})
        if m and all(m['candidate_deg']<m[k] for k in ('default_deg','fast_deg','slow_deg')):strict.append(d)
    for d in requested:
        part=[r for r in others if r['dataset']==d and r.get('candidate_deg')]
        if part and all(fmean(float(row['candidate_deg'])for row in part)<fmean(float(row[k])for row in part)
                        for k in ('full_vqf_deg','fixed_vqf_1p5_deg','fixed_vqf_6_deg')):strict.append(d)
    raw_controls=marked['all8-final-vqf-controls']['rows']
    assert len(raw_controls)==12 and set(strict)=={r['c0']for r in raw_controls if r['c6']=='3/3'}
    train=read(REVIEW/'selected-training-sources.json')
    assert train['selection_lock_sha256']==sha and train['own_fitting_records']==0 and train['distinct_external_source_records']==141
    source_ids={(r['source'],r['record']) for r in train['records']}
    exposure=read(ROOT/'final/exposure-audit.json')
    assert exposure['selection_sha256']==sha and len(source_ids)==141
    for name,group in [('main',main_rows),('additional',additional),('transfer',others)]:
        matches=source_ids&{(r['dataset'],r['record'])for r in group}
        assert len(matches)==exposure['direct_fitting_overlap']['counts'][name]
    dev=read(REVIEW/'development_replay_C038_C038_deg.json')['rows']
    leakage={(r['dataset'],r['record'])for r in dev}&{(r['dataset'],r['record'])for r in main_rows}
    assert leakage=={tuple(k)for k in exposure['development_main_reuse']['records']}
    prev=read(REVIEW/'historical_final_join.json')['previous_transfer']['groups']['R5+R6']
    assert prev['records']==20 and abs(prev['current']-domain['SHICE']['model_deg'])<1e-10
    checked_at=datetime.now().astimezone()
    began=datetime.fromisoformat(timing['window_start'])
    closes=datetime.fromisoformat(timing['window_end'])
    budget_ids={r['batch']for r in scores['batches']}
    assert {p.name for p in ROOT.glob('C[0-9][0-9][0-9]') if p.is_dir()}==budget_ids
    result={'verified_at':checked_at.isoformat(),'selected':plan['selected']['batch'],
            'authorized_window':{'begin':began.isoformat(),'end':closes.isoformat(),'seconds':int((closes-began).total_seconds())},
            'elapsed_since_authorization_seconds':(checked_at-began).total_seconds(),'window_elapsed_at_verification':checked_at>=closes,
            'selection_lock_sha256':sha,'historical_or_formal_primary_wins':evidence['main_improved_domains'],
            'total_domains_better_than_default':sum(x.get('delta_deg',0)<0 for x in domain.values()),
            'domains_better_than_all_three_fixed_VQF':sorted(strict),'domains':domain,'requested_across_twelve':len(main_rows)+len(others),
            'scored_across_twelve':sum(x['scored'] for x in domain.values()),'other_full_coverage_rows':len(additional),
            'development_named_batches':scores['completed_named_batches'],'distinct_configs':scores['new_or_inherited_candidate_policies_tested'],
            'training_sources':train['unique_by_source'],'own_records_in_fitting':train['own_fitting_records'],
            'matched_final_raw_anchors':alternate['checked_records'],'matched_supplemental_native_records':alternate['supplemental_checked'],
            'main_independent_implementation_max_gap_deg':alternate['max_main_gap_deg'],'selection_time':plan['locked_at'],
            'all_independent_replays':all_native['verified'],'all_native_rows':all_native['native_frames'],
            'verified_coordinator_batch_messages':guidance['separate_batch_directives_received'],
            'working_local_markdown_links':link_counts,'verified_report_evidence_queries':len(marked),
            'frozen_historical_models':{k:legacy[k] for k in ('wins_by_domain','scored_native_rows','native_anchor_count','worker_status_counts','original_archive_max_gap_deg','original_archive_rows_passing')},
            'shice_clock_study':{k:clock[k] for k in ('reconstructed_masks','records_per_offset','raw_anchor_comparisons','raw_anchor_max_gap_deg')},
            'live_self_raw_and_mask_files':{'files':len(raw_sources),'records':len(raw_paths),
                'bytes':sum(v['bytes'] for v in raw_sources.values()),'checksums':raw_sources},
            'max_two_implementation_difference_deg':all_native['max_absolute_difference_deg'],
            'last_science_at':timing['science_completed_at'],'guidance_audited_at':timing['guidance_verified_at'],'html_checked_utc':qa['reviewed_utc'],
            'offline_delivery':release,
            'qa':qa['viewports_css_px'],'hashes':{p.relative_to(ROOT.parent).as_posix():digest(p) for p in
                [main_path,additional_path,ROOT/'final/transfer.csv',*other_files,
                 ROOT/'final/historical-models/main.csv',ROOT/'final/historical-models/transfer.csv',
                 ROOT/'final/historical-models/tcn-backend-attribution/summary.json',
                 REVIEW/'historical-models-verified.json',REVIEW/'previous-model-native-anchors.json',
                 ROOT/'final/shice-clock-sensitivity/sequences.csv',REVIEW/'shice-clock-verified.json',
                 REVIEW/'REPORT.md',REVIEW/'SPLITS.md',REVIEW/'guidance-receipts.csv',REVIEW/'guidance-verification.json',
                 REVIEW/'final-domain-gains.png',REVIEW/'REPORT.html',REVIEW/'published-report-qa.json',html,old/'render_check.json']}}
    (REVIEW/'final-delivery.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(result['total_domains_better_than_default'],result['scored_across_twelve'],result['domains_better_than_all_three_fixed_VQF'])


if __name__=='__main__':main()
