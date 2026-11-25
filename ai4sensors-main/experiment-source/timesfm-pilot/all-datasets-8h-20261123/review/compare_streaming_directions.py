"""Compare the real single-packet path to native batch inference on a fixed external 200 Hz record."""
import argparse
import json

import numpy as np
import torch

from check_sources import ROOT
from lock_from_frontier import digest,inference,read
from locked_replay import Predictor
from stream_latency import Streaming


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('batch');parser.add_argument('--column')
    args=parser.parse_args();batch=args.batch;column=args.column or batch+'_deg'
    defs={(r['batch'],r['column']):r for r in read(ROOT/'review/candidate_recipes.json')}
    torch.set_num_threads(1)
    model=Predictor(inference((batch,column),defs))
    source=ROOT/'external-sources/ridi-phone-train/hao_handheld1.npz'
    with np.load(source)as f:
        assert f['imu_observed'][6000:9000].all()
        packets=np.ascontiguousarray(np.c_[f['acc_m_s2'][6000:9000],f['gyr_rad_s'][6000:9000]],dtype='f8')
    assert len(packets)==3000
    live=Streaming(model.stream_heads(),model.force.get('candidate_deg',0.),model.bank.get('candidate_deg',0.),model.router)
    stream=[];releases=calls=0
    for packet in packets:
        up,release=live.packet(packet);stream.append(up)
        releases+=int(release);calls+=int(live.called)
    stream=np.asarray(stream)
    full=model.predict(packets,200.,np.arange(len(packets)))['candidate_deg']
    error=np.degrees(np.arctan2(np.linalg.norm(np.cross(stream,full),axis=1),np.einsum('ij,ij->i',stream,full)))
    assert releases>0 and calls>0 and error.max()<1e-5
    result={'scope':'Deployment implementation check on a fixed 3,000-packet extract from an admitted walking source. No target scores or latency measurement.',
            'batch':batch,'column':column,'source':source.relative_to(ROOT).as_posix(),'source_sha256':digest(source),
            'packet_interval':[6000,9000],'packets':len(error),'releases':releases,'learned_releases':calls,
            'direction_max_angle_deg':float(error.max()),'direction_p99_deg':float(np.quantile(error,.99))}
    (ROOT/'review'/f'{batch}_packet_parity.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(batch,result['direction_max_angle_deg'],releases,calls)


if __name__=='__main__':main()
