"""Characterize fixed VQF response on three real training tracks with known source-side bias variation."""
import json
import numpy as np

from check_euroc_source import load_source
from check_sources import ROOT
from native_catalog import body_reference
from lock_from_frontier import read,digest
from vqf_response_controls import fixed_scores


def main():
    directory=ROOT/'external-sources/euroc-original-inputs'
    table=[]
    for entry in read(directory/'source-audit.json')['records']:
        name=entry['record'];assert name.startswith('V1_')
        path=directory/entry['prepared_file']
        assert digest(path)==entry['prepared_sha256']
        with np.load(path)as f:
            acc=f['acc_m_s2'];original=f['gyr_rad_s']
            truth,valid=body_reference(f['alternative_wxyz'],[0.,0.,1.])
            mask=valid&f['alternative_valid']&f['movement']&(np.arange(len(acc))>=2000)
        a,corrected,_,dt,_=load_source(name)
        np.testing.assert_allclose(acc,a,rtol=0.,atol=1e-6)
        bias=original-corrected
        prefix=np.median(bias[:2000],axis=0)
        cases={'raw':original,'fixed_prefix_bias_variation':original-prefix,'publisher_bias_variation':corrected}
        results={key:fixed_scores((np.column_stack((acc,samples)),200.,truth,mask)) for key,samples in cases.items()}
        assert len({r['scoring_indices_sha256'] for r in results.values()})==1
        table.append({'record':name,'native_rows':len(acc),'prefix_bias_deg_s':np.degrees(prefix).tolist(),
                      'original_source_sha256':digest(path),'conversion_sha256':digest(ROOT/'external-sources/euroc-vicon-room1'/(name+'.hdf5')),
                      'scores':results})
    means={c:{k:float(np.mean([r['scores'][c][k]for r in table])) for k in ('fast_deg','default_deg','slow_deg')} for c in table[0]['scores']}
    (ROOT/'review/euroc_bias_response.json').write_text(json.dumps({'scope':'Three real Vicon Room 1 external training recordings with fixed alternative-easy attitude reference and identical movement mask. The two offline input variants remove the source-published gyro bias, either the initial ten seconds median or point by point. They share the original capture, with no additional independent recording and no target-domain data.','records':table,'means':means},indent=2)+'\n',encoding='utf8')
    print(means)


if __name__=='__main__':main()
