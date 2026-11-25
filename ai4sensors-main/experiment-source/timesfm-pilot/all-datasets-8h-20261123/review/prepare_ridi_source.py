"""Prepare four real smartphone motions from one training-only RIDI subject using causal IMU timing."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from mount_check import angle
from verify_euroc_acquisition import RangeReader

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'external-sources/ridi-phone-train'
URL='https://dl.dropboxusercontent.com/s/9zzaj3h3u4bta23/ridi_data_publish_v2.zip'
BYTES=1_342_906_183


def members(archive,name):
    entry=archive.getinfo('data_publish_v2/'+name)
    path=OUT/'original'/name
    if path.exists():
        import zlib
        data=path.read_bytes()
        assert len(data)==entry.file_size and (zlib.crc32(data)&0xffffffff)==entry.CRC
    else:
        data=archive.read(entry)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.partial');temp.write_bytes(data);temp.replace(path)
    return data,{'archive_entry':entry.filename,'bytes':len(data),'crc32':f'{entry.CRC:08x}',
                 'sha256':hashlib.sha256(data).hexdigest()}


def load_sensor(raw,cols):
    values=np.loadtxt(io.BytesIO(raw),dtype=[('t','<i8'),('v','<f8',(cols,))],comments='#')
    t,idx=np.unique(values['t'],return_index=True)
    assert np.all(np.diff(values['t'])>=0) and np.all(np.diff(t)>0)
    return t,values['v'][idx],len(values)-len(t)


def align_sensors(ta,acc,tg,gyro,time):
    ai=np.searchsorted(ta,time,side='right')-1
    gi=np.searchsorted(tg,time,side='right')-1
    assert ai.min()>=0 and gi.min()>=0
    a_age=time-ta[ai];g_age=time-tg[gi]
    assert a_age.min()>=0 and g_age.min()>=0
    signal=np.c_[acc[ai],gyro[gi]]
    use=np.isfinite(signal).all(1)&(a_age<=20_000_000)&(g_age<=20_000_000)
    signal[~use]=np.nan
    return signal,use,a_age,g_age


def prepare(archive,name,placement):
    source={}
    data={}
    for k,dim in [('acce',3),('gyro',3),('pose',7)]:
        raw,source[k]=members(archive,name+'/'+k+'.txt')
        data[k]=load_sensor(raw,dim)
    ta,acc,da=data['acce'];tg,gyro,dg=data['gyro'];tp,pose,dp=data['pose']
    start=max(ta[0],tg[0]);stop=min(ta[-1],tg[-1])
    time=np.arange(start,stop+1,5_000_000,dtype='int64')
    signal,finite,a_age,g_age=align_sensors(ta,acc,tg,gyro,time)
    mid=len(time)//2
    cut=time[mid];short=align_sensors(ta[ta<=cut],acc[ta<=cut],tg[tg<=cut],gyro[tg<=cut],time[:mid+1])
    np.testing.assert_array_equal(short[0],signal[:mid+1])
    quat=pose[:,3:7]
    good=np.isfinite(quat).all(1)&(np.linalg.norm(quat,axis=1)>.8)&(np.linalg.norm(quat,axis=1)<1.2)
    tp=tp[good];quat=quat[good]
    bracket=np.searchsorted(tp,time,side='right')-1
    j=bracket.clip(0,len(tp)-2)
    valid=(bracket>=0)&(bracket+1<len(tp))&(tp[j+1]-tp[j]<=25_000_000)
    # The published preprocessing discards the first/last 800 reference observations. Keep them as IMU warmup only.
    valid&=(time>=tp[800])&(time<=tp[-801])
    q=np.zeros((len(time),4));q[:,0]=1.
    rot=Slerp((tp-tp[0])*1.e-9,Rotation.from_quat(quat))
    q[valid]=rot((time[valid]-tp[0])*1.e-9).as_quat(scalar_first=True)
    ori=Rotation.from_quat(q,scalar_first=True)
    check=valid&finite
    quiet=check&(np.linalg.norm(signal[:,3:],axis=1)<.10)&(np.abs(np.linalg.norm(signal[:,:3],axis=1)-9.80665)<.4)
    worldforce=ori[check].apply(signal[check,:3])
    means=worldforce.mean(axis=0)
    normalized=means/np.linalg.norm(means)
    still=angle(ori[quiet].apply(signal[quiet,:3]),[0.,0.,1.]) if quiet.any() else np.array([])
    hop=10
    pair_ok=np.convolve(check.astype(int),np.ones(hop+1,dtype=int),mode='valid')==hop+1
    safe=np.nan_to_num(signal[:,3:])
    csum=np.r_[np.zeros((1,3)),np.cumsum(safe,axis=0)]
    angular=(ori[:-hop].inv()*ori[hop:]).as_rotvec()/(hop*.005)
    measured=(csum[1+hop:]-csum[1:-hop])/hop
    pair_ok&=(np.linalg.norm(angular,axis=1)>.3)&(np.linalg.norm(measured,axis=1)>.3)
    errors=angle(angular[pair_ok],measured[pair_ok])
    output=OUT/(name+'.npz')
    with output.with_suffix('.partial').open('wb') as f:
        np.savez_compressed(f,time_ns=time,acc_m_s2=signal[:,:3],gyr_rad_s=signal[:,3:],imu_observed=finite,
                            tango_wxyz=q,reference_valid=valid)
    output.with_suffix('.partial').replace(output)
    return {'record':name,'subject':name.split('_')[0],'placement':placement,'phase':'external_train_only',
            'source_files':source,'native_200hz_rows':len(time),'observed_imu_rows':int(finite.sum()),
            'paired_pose_imu_rows':int(check.sum()),'duplicate_acc_gyro_pose_timestamps':[da,dg,dp],
            'max_used_acc_and_gyro_age_ms':[float(a_age[finite].max()*1.e-6),float(g_age[finite].max()*1.e-6)],
            'mean_world_specific_force_m_s2':means.tolist(),'mean_vertical_alignment_angle_deg':float(angle(normalized[None,:],[0.,0.,1.])[0]),
            'quiet_rows':int(quiet.sum()),'quiet_up_median_deg':float(np.median(still)) if len(still) else None,
            'dynamic_gyro_reference_pairs':int(pair_ok.sum()),'gyro_reference_median_deg':float(np.median(errors)) if len(errors) else None,
            'causal_input_prefix_parity':True,'file':output.name,'file_sha256':hashlib.sha256(output.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reader=RangeReader(URL,BYTES)
    with zipfile.ZipFile(reader) as archive:
        docs={};lists={}
        for kind,name in [('train','list_train_publish_v2.txt'),('test','list_test_publish_v2.txt'),('cross','list_crosssubject.txt')]:
            raw,docs[kind]=members(archive,name)
            lists[kind]=[line.strip().split(',') for line in raw.decode('utf-8-sig').splitlines() if line.strip()]
        excluded={group[0].split('_')[0] for k in ('test','cross') for group in lists[k]}
        allowed=[(s,p) for s,p in lists['train'] if s.split('_')[0] not in excluded]
        subjects=sorted({s.split('_')[0] for s,_ in allowed})
        qualifying=[s for s in subjects if {p for n,p in allowed if n.split('_')[0]==s}=={'handheld','bag','body','leg'}]
        if not qualifying:raise ValueError('No source-only participant has all four placements')
        selected=[min((n,p) for n,p in allowed if p==kind and n.split('_')[0]==qualifying[0]) for kind in ('bag','body','handheld','leg')]
        rows=[prepare(archive,n,p) for n,p in selected]
    result={'purpose':'Candidate external smartphone training source; outside the eight evaluation domains.',
            'project':'https://yanhangpublic.github.io/ridi/index.html',
            'dataset_link_from_author':'https://github.com/higerra/ridi_imu/blob/master/README.md',
            'published_processing':'https://github.com/higerra/ridi_imu/blob/master/python/gen_dataset.py',
            'archive':URL,'archive_bytes':BYTES,'retrieved_bytes_this_run':reader.transferred,'lists':docs,
            'reference':'Same-handset Tango visual-inertial pose, with original pose xyzw ordering converted to wxyz. '
                        'The Android rotation vector and vendor gravity channels are not used as references or features.',
            'timing':'Uniform 200 Hz from first shared IMU input, one-sided last-value hold (up to 20 ms). Pose only '
                     'as training outcome using a two-sided SLERP bracket up to 25 ms. Initial/final 800 pose rows excluded.',
            'selection':'Lexicographically first subject absent from official test and cross-subject lists with every placement, '
                       'then first training record for each placement. Selection uses names only.',
            'official_list_counts':{k:len(v) for k,v in lists.items()},'selected_subject':qualifying[0],
            'subject_intersection_with_official_test_and_cross':[],'rows':rows}
    (OUT/'source-audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    for x in rows:print(x['record'],x['paired_pose_imu_rows'],x['mean_vertical_alignment_angle_deg'],
                        x['quiet_up_median_deg'],x['gyro_reference_median_deg'],flush=True)


if __name__=='__main__':
    main()
