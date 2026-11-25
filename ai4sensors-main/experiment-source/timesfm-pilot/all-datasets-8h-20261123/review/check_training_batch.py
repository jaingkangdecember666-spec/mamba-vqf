"""Audit an external Mamba fit with native-frame replays and grouped development results."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict

import numpy as np
import torch
from scipy.io import loadmat
from scipy.spatial.transform import Rotation

from check_gates import raw_gates
from check_sources import DATA, ROOT, TRANSFER, tum, weights, yt
from group_uncertainty import paired_by_group
from native_blocks import block_inputs
from optical_reference import extract
from optical_score import canonical_frame


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csv_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce_hold_labels(batch):
    """Construct the per-release mean directly over individual optical labels for the first source."""
    entry = read(batch/'target-stats.json')[0]
    mat = loadmat(DATA/entry['relative_data_path'], variable_names=('imu_acc','imu_gyr','sampling_rate','opt_quat','movement'))
    signal = np.column_stack((mat['imu_acc'], mat['imu_gyr']))
    hz = float(mat['sampling_rate'].item())
    assert np.isfinite(signal).all()
    q, windows, _, releases = block_inputs(signal, hz)
    quat = mat['opt_quat']
    valid = np.isfinite(quat).all(axis=1) & (np.linalg.norm(quat,axis=1) > 0)
    truth = np.zeros((len(quat), 3))
    truth[valid] = Rotation.from_quat(quat[valid],scalar_first=True).inv().apply([0.,0.,1.])
    score = valid & mat['movement'].ravel().astype(bool)
    use = score[releases] & canonical_frame(windows)[2]
    assert int(use.sum()) == entry['used_windows']
    stops = np.r_[releases[1:], len(score)]
    holds = []
    angle = []
    frame_counts = []
    for t, end in zip(releases[use], stops[use]):
        indices = np.flatnonzero(score[t:end]) + t
        mean = q[indices].apply(truth[indices]).mean(axis=0)
        point = q[t].apply(truth[t])
        holds.append(np.linalg.norm(mean))
        frame_counts.append(len(indices))
        angle.append(np.rad2deg(np.arctan2(np.linalg.norm(np.cross(mean,point)),mean@point)))
    rms = float(np.sqrt(np.mean(np.square(angle))))
    np.testing.assert_allclose(rms, entry['point_to_hold_rms_deg'], rtol=0, atol=1e-7)
    np.testing.assert_allclose(np.mean(holds), entry['mean_target_norm'], atol=1e-12,rtol=0)
    assert sum(frame_counts) == entry['scoring_frames_across_used_hold_targets']
    return {'record':entry['record'],'intervals':len(holds),'native_labels':sum(frame_counts),
            'point_to_interval_mean_angle_rms_deg':rms,'mean_length':float(np.mean(holds))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('batch')
    p.add_argument('checkpoint')
    p.add_argument('--candidate')
    args = p.parse_args()
    torch.set_num_threads(2)
    col = args.candidate or args.batch+'_deg'
    batch = ROOT/args.batch
    observations = csv_rows(batch/'sequences.csv')
    inventory = read(ROOT/'C001/inventory.json')
    allowed = {(r['dataset'],r['record']):r for r in inventory['records']
               if r['side'] in ('development','development_previously_used_for_selection')}
    prior = {(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C006/sequences.csv')}
    indexed = {(r['dataset'],r['record']):r for r in observations}
    assert set(indexed) == set(allowed) == set(prior) and len(indexed) == len(observations) == 213
    domains = defaultdict(list)
    for key,r in indexed.items():
        before = prior[key]
        assert r['status'] == 'scored' and not r.get('error') and np.isfinite(float(r[col]))
        for field in ('group','mask_key','scoring_indices_sha256','scored_frames'):
            assert r[field] == before[field], (key,field)
        assert abs(float(r['full_vqf_deg'])-float(before['full_vqf_deg'])) < 1e-6
        if 'c003_deg' in r:
            assert abs(float(r['c003_deg'])-float(before['c003_deg'])) < 1e-6
        if 'C006_deg' in r:
            assert abs(float(r['C006_deg'])-float(before['B_rest_gated_deg'])) < 1e-6
        domains[r['dataset']].append(r)
    original = torch.load(TRANSFER/'batch6/mamba-intrinsic-selected.pt', map_location='cpu',weights_only=True)
    path = batch/args.checkpoint
    checkpoint = torch.load(path,map_location='cpu',weights_only=True)
    assert all(torch.equal(checkpoint[k],original[k]) for k in ('mean','std'))
    changed = {k:int(t.numel()) for k,t in checkpoint['state_dict'].items()
               if not torch.equal(t,original['state_dict'][k])}
    assert all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values())
    assert 'output.weight' in changed and 'output.bias' in changed
    model = weights(path,'mamba_v1')
    anchors = [('TUM-VI','room1'),('OxIOD','running/data1/syn/imu1.csv'),('YT_Server',inventory['yt']['selected_valid'][0])]
    raw_replays = []
    for key in anchors:
        d, name = key
        raw = tum(1) if d == 'TUM-VI' else yt(name) if d == 'YT_Server' else extract(allowed[key])[:4]
        ids, _, scores = raw_gates(raw,model)
        row = indexed[key]
        assert hashlib.sha256(np.asarray(ids,dtype='<i8').tobytes()).hexdigest() == row['scoring_indices_sha256']
        error = abs(scores['rest'] - float(row[col]))
        assert error < 1e-5, (key,error)
        raw_replays.append({'dataset':d,'record':name,'scored_frames':len(ids),'tilt_deg':scores['rest'],'abs_difference_deg':error})
    by_domain = {d:{'default':float(np.mean([float(x['full_vqf_deg']) for x in rr])),
                    'candidate':float(np.mean([float(x[col]) for x in rr])),
                    **paired_by_group(rr,'full_vqf_deg',col)} for d,rr in domains.items()}
    summary = {'batch':args.batch,'results_sha256':digest(batch/'results.json'),'sequences_sha256':digest(batch/'sequences.csv'),
               'checkpoint_sha256':digest(path),'changed_tensors_vs_B6':changed,'parameters_in_changed_tensors':sum(changed.values()),
               'unchanged_normalization':True,'paired_development_records':len(indexed),'datasets':by_domain,
               'native_anchors':raw_replays}
    if args.batch == 'C007':
        assert set(changed) == {'output.weight','output.bias'}
        assert read(batch/'protocol.json')['source_records'] == read(ROOT/'C004/protocol.json')['source_records']
        steps = csv_rows(batch/'training-steps.csv')
        assert [int(r['step']) for r in steps] == list(range(1,161))
        assert np.isfinite([[float(r['loss']),float(r['gradient_norm'])] for r in steps]).all()
        summary['external_hold_labels'] = reproduce_hold_labels(batch)
    if args.batch == 'C008':
        protocol = read(batch/'protocol.json')
        prep = read(batch/'training-preparation.json')
        assert protocol['source_records'] == read(ROOT/'C004/protocol.json')['source_records']
        with np.load(ROOT/'C004/schedules.npz') as archive:
            draw = archive['B']
            offsets = archive['record_offsets']
            assert list(draw.shape) == protocol['B_schedule_shape']
            assert hashlib.sha256(draw.astype('<i8').tobytes()).hexdigest() == protocol['B_schedule_sha256']
            ids = np.searchsorted(offsets[1:],draw,side='right')
            counts = np.bincount(ids.ravel(),minlength=len(prep['records']))
            np.testing.assert_array_equal(counts,[r['B_draws'] for r in prep['records']])
        training = read(batch/'training.json')
        for fn,h in training['artifacts_sha256'].items():
            assert digest(batch/fn)==h
        steps=csv_rows(batch/'training-steps.csv')
        assert [int(r['step']) for r in steps] == list(range(1,161))
        for k in ('backbone_gradient_l2','output_gradient_l2'):
            assert all(np.isfinite(float(x[k])) and float(x[k])>0. for x in steps)
        changed_elements = sum(int((t!=original['state_dict'][k]).sum()) for k,t in checkpoint['state_dict'].items())
        assert sum(x['changed_parameters'] for x in read(batch/'layer-norms.json'))==changed_elements
        assert {'sequence.A_log','sequence.conv1d.weight','input.weight','sequence.dt_proj.weight'}.issubset(changed)
        assert read(batch/'equivariance.json')['passed_at_1e_4_deg']
        summary['matched_external_fit']={'external_records':len(prep['records']),'optimization_steps':len(steps),
                 'actual_changed_elements':changed_elements,'positive_gradient_steps_both_groups':len(steps),
                 'same_first_loss_as_readout_fit':training['first_actual_step_loss']==training['archived_C004_B_first_loss'],
                 'fit_seconds':training['fit']['seconds'],'preparation_failures_before_fit':int((batch/'preparation-attempt/run.log').exists())}
    if args.batch == 'C009':
        protocol=read(batch/'protocol.json')
        assert protocol['old_source_records'] == read(ROOT/'C004/protocol.json')['source_records']
        prep=read(batch/'training-preparation.json')
        old={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C008/sequences.csv')}
        assert all(abs(float(r['C008_deg'])-float(old[k]['C008_deg']))<1e-6 for k,r in indexed.items())
        with np.load(batch/'sampling.npz') as used, np.load(ROOT/'C004/schedules.npz') as orig:
            replace=used['replacement_mask']
            np.testing.assert_array_equal(replace,np.arange(96)[None,:]%6==np.arange(160)[:,None]%6)
            np.testing.assert_array_equal(used['C008_global_window'],orig['B'])
            for suffix in ('global_window','record_ordinal','release_native'):
                np.testing.assert_array_equal(used['C009_'+suffix][~replace],used['C008_'+suffix][~replace])
            ordinals=used['C009_record_ordinal'][replace]-63
            np.testing.assert_array_equal(ordinals,np.arange(2560)%3)
            offsets=used['euroc_eligible_offsets']
            local=used['euroc_drawn_eligible_index']
            assert np.all(local>=0) and np.all(local<np.diff(offsets)[ordinals])
            chosen=offsets[ordinals]+local
            np.testing.assert_array_equal(used['C009_release_native'][replace],used['euroc_eligible_release_native'][chosen])
            assert np.all(used['euroc_eligible_history_first_native']>=2000)
            assert np.all(used['euroc_eligible_input_last_native']<=used['euroc_eligible_release_native'])
        training=read(batch/'training.json')
        for fn,h in training['artifacts_sha256'].items():
            assert digest(batch/fn)==h
        checksums=read(ROOT/'external-sources/euroc-vicon-room1/source-manifest.json')['records']
        for raw,info in zip(checksums,prep['new_rows'],strict=True):
            assert raw['record']==info['record'] and raw['sha256']==info['file_sha256']
        summary['source_sampling']={'unchanged_original_slots':int((~replace).sum()),'new_source_slots':int(replace.sum()),
                 'eligible_new_windows':int(offsets[-1]),
                 'draws_per_new_trajectory':np.bincount(ordinals).tolist(),'changed_elements':
                 sum(int((t!=original['state_dict'][k]).sum()) for k,t in checkpoint['state_dict'].items())}
    if args.batch == 'C010':
        arm=col.removesuffix('_deg')
        assert arm in ('A','B')
        training=read(batch/'training.json')
        assert training['actual_fit_count']==2 and training['fits'][arm]['weight_sha256']==digest(path)
        prep=read(batch/'training-preparation.json')
        assert prep['shared_input_sha256_by_arm']['A']==prep['shared_input_sha256_by_arm']['B']
        step_audit=csv_rows(batch/'batch-audit.csv')
        assert len(step_audit)==160
        with np.load(ROOT/'C009/sampling.npz') as schedule:
            for step,r in enumerate(step_audit):
                assert r['A_input_sha256']==r['B_input_sha256'] and r['A_target_sha256']!=r['B_target_sha256']
                for suffix in ('record_ordinal','release_native'):
                    column='C009_'+('native_release' if suffix=='release_native' else suffix)+'_sha256'
                    data=schedule['C009_'+suffix][step].astype('<i8')
                    assert hashlib.sha256(data.tobytes()).hexdigest()==r[column]
                assert int(r['changed_easy_target_slots'])==int((schedule['C009_record_ordinal'][step]==63).sum())
        for src in prep['euroc_records']:
            assert src['C009_release_native_sha256']==src['C010_release_native_sha256']
            a,b=src['target_sha256_by_arm'].values()
            assert (a!=b)==(src['record']=='V1_01_easy')
        old={(r['dataset'],r['record']):r for r in csv_rows(ROOT/'C009/sequences.csv')}
        assert all(abs(float(r['C009_deg'])-float(old[k]['C009_deg']))<1e-6 for k,r in indexed.items())
        summary['source_quality_pair']={'arm':arm,'fixed_input_digests_by_batch_verified':len(step_audit),
                 'changed_target_draws_easy_only':sum(int(x['changed_easy_target_slots']) for x in step_audit),
                 'other_target_draws':160*96-prep['B_changed_easy_only_draw_labels']}
    if args.batch == 'C012':
        assert set(changed)=={'output.weight','output.bias'}
        first=read(batch/'first-step-control.json')
        assert read(batch/'protocol.json')['source_records']==read(ROOT/'C004/protocol.json')['source_records']
        steps=csv_rows(batch/'training-steps.csv')
        external=read(ROOT/'review/C011_source_audit.json')
        with np.load(batch/'training-windows.npz') as windows, np.load(ROOT/'C004/schedules.npz') as prior_draws, np.load(ROOT/'C011/training-windows.npz') as witness:
            draw=windows['original_B_schedule']
            flags=windows['vqf_rest']
            ids=windows['record_ordinal']
            np.testing.assert_array_equal(draw,prior_draws['B'])
            np.testing.assert_array_equal(windows['native_release'],prior_draws['release_indices'][draw])
            np.testing.assert_array_equal(ids,np.searchsorted(prior_draws['record_offsets'][1:],draw,side='right'))
            np.testing.assert_array_equal(flags,witness['rest_gate'].reshape(draw.shape))
            per_record=np.bincount(ids[flags],minlength=63)
            np.testing.assert_array_equal(per_record,[r['rest_draws'] for r in external['records']])
            assert len(steps)==160 and all(int(r['step'])==i+1 and int(r['denominator'])==96 and int(r['rest_total'])==int(flags[i].sum()) for i,r in enumerate(steps))
            points=windows['C004_points'][0].astype(float)
            feat,up,target=points[:,:-6],points[:,-6:-3],points[:,-3:]
            w=original['state_dict']['output.weight'].numpy().astype(float)
            b=original['state_dict']['output.bias'].numpy().astype(float)
            delta=.1*(feat@w.T+b)
            delta-=(delta*up).sum(1)[:,None]*up
            delta*=np.minimum(1.,np.tan(np.deg2rad(float(original['cap_deg'])))/np.maximum(1e-8,np.linalg.norm(delta,axis=1)))[:,None]
            pred=up+.25*delta
            pred[flags[0]]=up[flags[0]]
            pred/=np.linalg.norm(pred,axis=1,keepdims=True)
            loss=float(np.mean(((pred-target)**2).sum(1)))
        assert abs(loss-first['gated_loss'])<1e-9
        assert first['rest_only_parameter_grad_max_abs']==first['gated_vs_nonrest_grad_max_abs']==0.
        assert first['C004_B_archived_loss']==first['ungated_C004_B_replay_loss']
        summary['training_gate_verification']={'original_schedule_slots':int(draw.size),'rest_by_domain':external['rest_draws_by_domain'],
            'denominator_all_steps':96,'external_record_count':len(per_record),'steps':len(steps),
            'double_precision_first_loss':loss,'first_loss_abs_discrepancy':abs(loss-first['gated_loss']),
            'verified_zero_rest_parameter_gradient_first_batch':True}
    if args.batch=='C015':
        assert set(changed)=={'output.weight','output.bias'}
        old_sources=read(ROOT/'C004/protocol.json')['source_records']
        root=ROOT/'external-sources/ridi-phone-train'
        manifest=read(root/'source-audit.json')
        train=read(batch/'training.json')
        prep=read(batch/'training-preparation.json')
        assert len(prep['original_sources'])==len(old_sources)==63 and train['fit_count']==1
        check=[]
        with np.load(batch/'sampling.npz') as got,np.load(ROOT/'C004/schedules.npz') as base:
            replaced=got['replacement_mask']
            np.testing.assert_array_equal(replaced,np.arange(96)[None,:]%12==np.arange(160)[:,None]%12)
            np.testing.assert_array_equal(got['C004_B_global_window'],base['B'])
            for k in ('global_window','record_ordinal','native_release'):
                np.testing.assert_array_equal(got['C004_B_'+k][~replaced],got['C015_'+k][~replaced])
            phone=got['RIDI_track_ordinal'];indices=got['RIDI_drawn_index_within_track']
            np.testing.assert_array_equal(np.bincount(phone),np.full(4,320))
            np.testing.assert_array_equal(got['C015_record_ordinal'][replaced],len(old_sources)+phone)
            for i,row in enumerate(manifest['rows']):
                path=root/row['file'];assert digest(path)==row['file_sha256']
                with np.load(path) as original:
                    name=row['record'];signal=np.c_[original['acc_m_s2'],original['gyr_rad_s']]
                    _,window,at,release=block_inputs(signal,200.)
                    base_allowed=original['reference_valid'][release] & original['imu_observed'][release]
                    base_allowed&=(original['time_ns'][release]-original['time_ns'][0])>=10_000_000_000
                    np.testing.assert_array_equal(got[name+'_observed_imu'],original['imu_observed'])
                    np.testing.assert_array_equal(got[name+'_reference_valid'],original['reference_valid'])
                    positions=np.flatnonzero(base_allowed)
                    pos=positions[canonical_frame(window[positions])[2]]
                    expected=release[pos];np.testing.assert_array_equal(expected,got[name+'_eligible_release'])
                    np.testing.assert_array_equal(at[pos],got[name+'_input_last'])
                    takes=expected[indices[phone==i]]
                    np.testing.assert_array_equal(takes,got['C015_native_release'][replaced][phone==i])
                    check.append({'record':name,'eligible_releases':len(expected),'executed_draws':len(takes),
                                  'unique_releases_drawn':len(np.unique(takes))})
        for sk,h in train['files_sha256'].items():assert digest(batch/sk)==h
        summary['RIDI_substitution']={'unchanged_C004_B_slots':int((~replaced).sum()),
            'RIDI_slots':int(replaced.sum()),'source_inputs':check,'training_subject':'hao',
            'author_test_subjects_in_training':0}
    if args.batch=='C016':
        prior_fit=torch.load(ROOT/'C015/phone-readout.pt',map_location='cpu',weights_only=True)
        assert set(changed)=={'output.weight','output.bias'}
        steps=[json.loads(line) for line in (batch/'iterations.jsonl').read_text(encoding='utf8').splitlines() if line]
        training=read(batch/'training.json');fit=training['fit'];groups=training['domain_order']
        with np.load(batch/'training-windows.npz') as data,np.load(ROOT/'C015/sampling.npz') as matched,np.load(ROOT/'C012/training-windows.npz') as earlier:
            record=data['source_record_ordinal'];belong=data['domain_ordinal']
            np.testing.assert_array_equal(record,matched['C015_record_ordinal'].ravel())
            np.testing.assert_array_equal(data['native_release'],matched['C015_native_release'].ravel())
            keep=(~matched['replacement_mask']).ravel()
            np.testing.assert_array_equal(data['rest_gate'][keep],earlier['vqf_rest'].ravel()[keep])
            np.testing.assert_allclose(data['values'][keep],earlier['C004_points'].reshape(-1,30)[keep],rtol=0.,atol=1e-7)
            weights_per_draw=data['per_draw_weight']
            for domain_id in range(len(groups)):
                domain_rows=belong==domain_id
                ids,num=np.unique(record[domain_rows],return_counts=True)
                for idx,n in zip(ids,num):
                    np.testing.assert_allclose(weights_per_draw[record==idx],1./(n*len(ids)),atol=1e-15)
            def domain_loss(cp):
                points=data['values'].astype(float);x,u,y=points[:,:-6],points[:,-6:-3],points[:,-3:]
                layer=cp['state_dict']
                pred=.1*(x@layer['output.weight'].numpy().astype(float).T+layer['output.bias'].numpy().astype(float))
                pred-=(pred*u).sum(1)[:,None]*u
                pred*=np.minimum(1.,np.tan(np.radians(float(cp['cap_deg'])))/np.maximum(1e-8,np.linalg.norm(pred,axis=1)))[:,None]
                pred[data['rest_gate']]=0.;pred=u+.25*pred;pred/=np.linalg.norm(pred,axis=1,keepdims=True)
                return np.bincount(belong,weights=weights_per_draw*((pred-y)**2).sum(1))
            start_loss=domain_loss(prior_fit);end_loss=domain_loss(checkpoint)
        np.testing.assert_allclose(start_loss,fit['initial_loss'],atol=1e-11,rtol=1e-7)
        np.testing.assert_allclose(end_loss,fit['final_loss'],atol=1e-10,rtol=1e-6)
        previous=np.ones(4);nsteps=0;max_kkt=0.
        for s in steps:
            grad=np.array(s['relative_domain_gradients']);weights_qp=np.array(s['qp']['coefficients'])
            direction=weights_qp@grad
            assert weights_qp.min()>=-1e-10 and abs(weights_qp.sum()-1.)<1e-10
            residual=max(0.,float(direction@direction-(grad@direction).min()))
            max_kkt=max(max_kkt,residual)
            assert residual<1e-8
            np.testing.assert_allclose(s['relative_loss_before'],previous,rtol=1e-10,atol=1e-10)
            if s.get('accepted_step'):
                nsteps+=1
                trial=s['trials'][-1];updated=np.array(s['relative_loss_after'])
                slope=-grad@direction/max(1e-20,np.linalg.norm(direction))
                bound=previous+1e-4*trial['requested_l2']*slope
                assert (updated<=bound+2e-12).all() and (updated<previous).all()
                previous=updated
        np.testing.assert_allclose(previous,end_loss/start_loss,atol=1e-6)
        assert nsteps==fit['accepted_steps'] and len(steps)==fit['attempts']
        summary['common_descent_verification']={'accepted':nsteps,'attempts':len(steps),'all_steps_decrease_each_external_loss':True,
             'worst_absolute_common_direction_KKT_residual':max_kkt,'strict_float32_policy_loss_ratios':dict(zip(groups,(end_loss/start_loss).tolist())),
             'retained_C015_slots':len(record),'source_records':len(np.unique(record))}
    suffix='_'+col.removesuffix('_deg') if args.candidate else ''
    (ROOT/'review'/f'{args.batch}{suffix}_audit.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(args.batch, len(indexed), 'records;', sum(x['candidate'] < x['default'] for x in by_domain.values()), 'lower means;',
          sum(changed.values()),'parameters in changed tensors;',max(r['abs_difference_deg'] for r in raw_replays),'deg replay error')


if __name__=='__main__':
    main()
