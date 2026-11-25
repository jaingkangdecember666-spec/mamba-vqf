"""Recover uncorrected training gyro from the bias subtracted by the published HDF5 converter."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from urllib.request import Request,urlopen

import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from check_euroc_source import ROOM1,load_source
from mount_check import angle

REV='69488123ed9362dd44b6f28e7f4680abbff1442b'
PREFIX='https://raw.githubusercontent.com/rpng/open_vins/'+REV+'/ov_data/euroc_mav/'
OUT=Path(__file__).resolve().parents[1]/'external-sources/euroc-original-inputs'


def fetch(url):
    with urlopen(Request(url,headers={'User-Agent':'IMU-training-source-audit'}),timeout=30) as response:
        return response.read(15_000_001)


def states(name,ref_files):
    filename=name+'.csv'
    item=ref_files[filename]
    file=OUT/filename
    raw=file.read_bytes() if file.exists() else fetch(PREFIX+filename)
    checksum=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest()
    assert checksum==item['sha'] and len(raw)==item['size'] and len(raw)<15_000_000
    if not file.exists():
        tmp=file.with_suffix('.partial');tmp.write_bytes(raw);tmp.replace(file)
    dtype=np.dtype([('utc_ns','<i8'),('state','<f8',(16,))])
    numeric=np.loadtxt(BytesIO(raw),delimiter=',',comments='#',dtype=dtype)
    assert np.all(np.diff(numeric['utc_ns'])>0) and np.isfinite(numeric['state']).all()
    provenance={'url':PREFIX+filename,'github_blob':checksum,'file_sha256':hashlib.sha256(raw).hexdigest(),'rows':len(numeric)}
    return numeric['utc_ns'],numeric['state'],provenance


def load_record(name,metadata):
    original=name+'_original' if name=='V1_01_easy' else name
    time,source,p1=states(original,metadata)
    acc,prepared,quat,dt,mask=load_source(name)
    assert len(time)==len(acc) and np.max(np.abs(np.diff(time)*1.e-9-.005))<1.e-6
    # The source quaternion is reproduced up to its double-cover sign and float32 conversion.
    published=source[:,3:7]
    normalized=published/np.linalg.norm(published,axis=1,keepdims=True)
    q_error=np.minimum(np.linalg.norm(quat-normalized,axis=1),np.linalg.norm(quat+normalized,axis=1))
    assert q_error.max()<1.e-6
    bw=source[:,10:13]
    original_gyro=prepared+bw
    metadata_out={'record':name,'original_state':p1,'hdf5_source_rows':len(acc),
                  'native_clock_interval_ns':[int(np.diff(time).min()),int(np.diff(time).max())],
                  'quaternion_max_sign_invariant_component_vector_l2':float(q_error.max()),
                  'original_bias_axis_median_rad_s':np.median(bw,axis=0).tolist(),
                  'original_bias_magnitude_median_rad_s':float(np.median(np.linalg.norm(bw,axis=1)))}
    good=np.ones(len(time),bool)
    if name=='V1_01_easy':
        tc,sc,p2=states(name,metadata)
        nearest=np.searchsorted(tc,time,side='right')-1
        safe=nearest.clip(0,len(tc)-2)
        good=(nearest>=0)&(nearest+1<len(tc))&(tc[safe+1]-tc[safe]<=100_000_000)
        aligned=np.zeros((len(time),4),dtype='float64')
        interpolator=Slerp((tc-tc[0])*1e-9,Rotation.from_quat(sc[:,3:7],scalar_first=True))
        aligned[good]=interpolator((time[good]-tc[0])*1e-9).as_quat(scalar_first=True)
        metadata_out.update(corrected_state=p2,corrected_pose_covered_rows=int(good.sum()),
                            corrected_max_bracket_s=.1,corrected_label_rows_excluded=int((~good).sum()))
    else:
        aligned=quat.copy()
    raw_w=original_gyro
    still=(np.linalg.norm(raw_w-bw,axis=1)<.10)&(np.abs(np.linalg.norm(acc,axis=1)-9.80665)<.30)&good
    original_vertical=Rotation.from_quat(quat[still],scalar_first=True).inv().apply([0.,0.,1.])
    alt_vertical=Rotation.from_quat(aligned[still],scalar_first=True).inv().apply([0.,0.,1.])
    metadata_out.update(quiet_source_check_rows=int(still.sum()),
                        original_quiet_vertical_median_deg=float(np.median(angle(acc[still],original_vertical))),
                        alternative_quiet_vertical_median_deg=float(np.median(angle(acc[still],alt_vertical))),
                        quiet_reference_to_reference_median_deg=float(np.median(angle(original_vertical,alt_vertical))))
    dest=OUT/(name+'.npz')
    with dest.with_suffix('.scratch').open('wb') as f:
        np.savez_compressed(f,utc_ns=time,acc_m_s2=acc,gyr_rad_s=raw_w,
                            original_wxyz=quat,alternative_wxyz=aligned,alternative_valid=good,
                            movement=mask)
    dest.with_suffix('.scratch').replace(dest)
    metadata_out.update(prepared_file=dest.name,prepared_sha256=hashlib.sha256(dest.read_bytes()).hexdigest())
    return metadata_out


def main():
    OUT.mkdir(exist_ok=True)
    entries=json.loads(fetch('https://api.github.com/repos/rpng/open_vins/contents/ov_data/euroc_mav?ref='+REV))
    files={e['name']:e for e in entries}
    with ThreadPoolExecutor(max_workers=3) as ex:
        records=list(ex.map(lambda n:load_record(n,files),ROOM1))
    notes={'purpose':'External training source only. Target scoring remains eight independent evaluation series collections.',
           'conversion':'https://github.com/daniel-om-weber/riann_dev/blob/435274141c08ab318c01be904de9754943b4feed/riann/prep/euroc.py',
           'correction_provenance':'https://docs.openvins.com/gs-datasets.html',
           'gyro':'RIANN conversion subtracts the released reference bias. Adding the original released bias back at every matched original timestamp restores acquisition gyro up to file rounding. Native source gyro validation is reported separately if obtained.',
           'alternate_easy_target':'SLERP the OpenVINS authors’ revised V1_01 attitude only within valid reference brackets; no extrapolation. Other two targets are the original states. Time and IMU input are identical across the two label choices.',
           'records':records}
    (OUT/'source-audit.json').write_text(json.dumps(notes,indent=2)+'\n',encoding='utf8')
    for r in records:
        print(r['record'],r['hdf5_source_rows'],r['original_bias_magnitude_median_rad_s'],
              r['original_quiet_vertical_median_deg'],r['alternative_quiet_vertical_median_deg'],flush=True)


if __name__=='__main__':
    main()
