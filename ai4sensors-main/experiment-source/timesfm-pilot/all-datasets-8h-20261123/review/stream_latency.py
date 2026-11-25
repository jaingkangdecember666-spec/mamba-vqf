"""Same-machine, batch-one, packet-to-tilt latency for an identified residual Mamba operating point."""
import argparse
import hashlib
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from vqf import VQF

from check_sources import ROOT, weights
from native_blocks import block_inputs
from optical_score import canonical_frame, estimate, infer_canonical
from filter_response_bank import response_delta
from router_motion_context import TAU as MOTION_TAU, G
from online_bias import Settings, sphere

Z=np.array([0.,0.,1.])


class StreamBias:
    def __init__(self,hz):
        self.cfg=Settings()
        self.buf=np.empty((round(hz*self.cfg.block_s),6))
        self.i=0
        self.offset=np.zeros(3)
        self.anchors=[];self.centers=[];self.counts=[]
        self.threshold=np.cos(np.deg2rad(self.cfg.direction_separation_deg))

    def ingest(self,row):
        self.buf[self.i]=row
        self.i+=1
        if self.i==len(self.buf):
            self.i=0
            self.complete_block()

    def complete_block(self):
        cfg=self.cfg
        if not np.isfinite(self.buf).all() or np.linalg.norm(self.buf[:,3:],axis=1).max()>cfg.max_gyro_rad_s:
            return
        acc=self.buf[:,:3]
        avg=acc.mean(0);size=np.linalg.norm(avg)
        if not cfg.minimum_acc_norm_m_s2<=size<=cfg.maximum_acc_norm_m_s2 or np.linalg.norm(acc-avg,axis=1).max()>cfg.max_acc_deviation_m_s2:
            return
        direction=avg/size
        c=np.asarray(self.anchors)@direction if self.anchors else np.empty(0)
        if len(c) and c.max()>=self.threshold:
            ix=int(c.argmax())
            self.counts[ix]+=1
            self.centers[ix]+=(avg-self.centers[ix])/self.counts[ix]
        else:
            self.anchors.append(direction)
            self.centers.append(avg)
            self.counts.append(1)
        qualified=np.asarray(self.counts)>=cfg.minimum_blocks_in_direction
        if qualified.sum()<cfg.minimum_directions:
            return
        center,_,_=sphere(np.asarray(self.centers)[qualified],cfg)
        if center is not None:
            step=cfg.smoothing*(center-self.offset)
            self.offset+=step*min(1.,cfg.maximum_step_m_s2/max(np.linalg.norm(step),1e-16))


class StreamMotion:
    """Four causal scalar observers. Share the live central VQF orientation and bias."""
    def __init__(self,hz):
        self.alpha=-np.expm1(-1./(MOTION_TAU*hz))
        self.state=np.zeros(6)
        self.mass=0.
        self.bias_norm=0.

    def ingest(self,clean,q,bias):
        acc=clean[:3];w=clean[3:]
        norm=np.linalg.norm(acc)
        good=norm>1e-12
        unit=q.apply(acc/norm) if good else np.zeros(3)
        observed=np.array([w@w,(norm/G-1.)**2,*unit,float(good)])
        self.state+=self.alpha*(observed-self.state)
        self.mass+=self.alpha*(1.-self.mass)
        self.bias_norm=np.linalg.norm(bias)

    def current(self):
        s=self.state
        spread=np.sqrt(2.*max(0.,1.-np.linalg.norm(s[2:5])/s[5])) if s[5]>1e-12 else 0.
        return np.log1p([np.sqrt(max(0.,s[0]/self.mass)),10.*np.sqrt(max(0.,s[1]/self.mass)),
                         10.*spread,np.degrees(self.bias_norm)])[None,:]


class Streaming:
    """Timing adapter for 200 Hz and complete input packets; head releases every 0.1 s after warmup."""
    def __init__(self,heads,force_gain=0.,bank_gain=0.,router=None,raw_anchor=None,physical_cap=None):
        self.heads=heads
        self.force_gain=force_gain
        self.world_force=np.zeros(3)
        self.alpha=-np.expm1(-.005/.25)
        self.packets_since_force=128
        self.filter=VQF(.005)
        self.bank_gain=bank_gain
        self.router=router
        self.motion=StreamMotion(200) if router and router.context else None
        self.filter_bank=[VQF(.005,tauAcc=tau)for tau in (1.5,6.)] if bank_gain or router else []
        self.cal=StreamBias(200)
        self.half=None
        self.ring_size=256 if any(getattr(h[0],'history_scale',1)==4 for h,_,_ in heads) else 64
        self.ring=np.zeros((self.ring_size,13),dtype='f4')
        self.blocks=0
        self.vertical=Z.copy()
        self.called=False
        self.raw_anchor=None
        self.physical_cap=physical_cap
        if physical_cap is not None:
            assert router is not None and raw_anchor is None
        if raw_anchor is not None:
            from raw_vqf_blend import StreamingBlend
            self.raw_anchor=StreamingBlend(200.,raw_anchor)

    def packet(self,row):
        clean=row.copy()
        clean[:3]-=self.cal.offset
        self.cal.ingest(row)
        self.filter.update(clean[3:],clean[:3])
        for aux in self.filter_bank:
            aux.update(clean[3:],clean[:3])
        q=Rotation.from_quat(self.filter.getQuat6D(),scalar_first=True)
        if self.motion:
            self.motion.ingest(clean,q,self.filter.getBiasEstimate()[0])
        if self.force_gain:
            norm=np.linalg.norm(clean[:3])
            self.packets_since_force+=1
            if norm>1e-12:
                self.world_force+=self.alpha*(q.apply(clean[:3]/norm)-self.world_force)
                self.packets_since_force=0
        released=False
        self.called=False
        if self.half is None:
            self.half=clean
        else:
            pos=self.blocks%self.ring_size
            self.ring[pos,:6]=.5*(clean+self.half)
            self.ring[pos,6:9]=q.inv().apply(Z)
            self.ring[pos,9:12]=self.filter.getBiasEstimate()[0]
            rest=self.filter.getRestDetected()
            self.ring[pos,12]=rest
            self.half=None;self.blocks+=1
            if self.blocks>=70 and self.blocks%10==0:
                released=True
                u=Z.copy()
                if self.physical_cap is not None:
                    pure=Z.copy()
                ordered=None
                coarse=None
                for head,gain,gate in self.heads:
                    if gate=='rest_off' and rest:
                        continue
                    scale=getattr(head[0],'history_scale',1)
                    if scale==4 and self.blocks-6<256:
                        continue
                    if ordered is None:
                        ordered=np.roll(self.ring,-(pos+1),axis=0)
                    if scale==4:
                        if coarse is None:
                            blocks4=ordered.reshape(64,4,13)
                            coarse=blocks4[:,-1,:].copy()
                            coarse[:,:6]=blocks4[:,:,:6].mean(1,dtype=np.float64)
                        window=coarse[None,:,:]
                    else:
                        window=ordered[None,-64:,:]
                    u+=gain*q.apply(infer_canonical(head,window)[0])
                    self.called=True
                if self.force_gain and not rest and self.packets_since_force<128:
                    magnitude=np.linalg.norm(self.world_force)
                    if magnitude>0. and np.sqrt(2.*max(0.,1.-magnitude))<np.deg2rad(3.):
                        direction=self.world_force/magnitude
                        direction[2]=0.
                        scale=min(self.force_gain,np.tan(np.deg2rad(6.))/max(1e-15,np.linalg.norm(direction)))
                        u+=scale*direction
                        if self.physical_cap is not None:
                            pure+=scale*direction
                if self.filter_bank and not rest:
                    if ordered is None:
                        ordered=np.roll(self.ring,-(pos+1),axis=0)
                    fast,slow=(Rotation.from_quat(aux.getQuat6D(),scalar_first=True)for aux in self.filter_bank)
                    if self.router:
                        from response_router import alternatives
                        f,s=alternatives(q,fast,slow)
                        probs,valid=self.router.probabilities(ordered[None,-64:,:],self.motion.current() if self.motion else None,(f,s))
                    else:
                        valid=canonical_frame(ordered[None,-64:,:])[2]
                    if valid[0]:
                        if self.router:
                            u+=probs[0,0]*f+probs[0,1]*s
                            if self.physical_cap is not None:
                                pure+=self.router.prior[0]*f+self.router.prior[1]*s
                            self.called=True
                        else:
                            u+=self.bank_gain*response_delta(q,fast,slow)
                self.vertical=u/np.linalg.norm(u)
                if self.physical_cap is not None:
                    from physical_angle_cap import cap_direction
                    self.vertical=cap_direction(pure,self.vertical,self.physical_cap)
        result=q.inv().apply(self.vertical)
        if self.raw_anchor:
            result=self.raw_anchor.packet(row,result)
        return result,released


def verify_bias():
    signal=np.c_[np.repeat(9.80665*np.r_[np.eye(3),-np.eye(3)]+np.array([.18,-.13,.09]),8*200,axis=0),
                 np.zeros((48*200,3))]
    ref,active,_,_=estimate(signal,np.ones(len(signal),bool),.005)
    live=StreamBias(200);copy=np.empty_like(ref)
    for i,s in enumerate(signal):
        copy[i]=live.offset
        live.ingest(s)
    assert active.any()
    np.testing.assert_allclose(copy,ref,atol=1e-12,rtol=1e-12)
    return {'samples':len(ref),'bias_active_samples':int(active.sum()),'maximum_bias_disagreement':float(np.max(abs(copy-ref)))}


def batch_reference(signal,head,gain,gate):
    bias,_,_,_=estimate(signal,np.ones(len(signal),bool),.005)
    clean=signal.copy();clean[:,:3]-=bias
    q,win,at,rel=block_inputs(clean,200.)
    dw=gain*q[at].apply(infer_canonical(head,win))
    if gate=='rest_off':
        dw[win[:,-1,12].astype(bool)]=0.
    u=Z+dw;u/=np.linalg.norm(u,axis=1,keepdims=True)
    index=np.searchsorted(rel,np.arange(len(signal)),side='right')
    return q.inv().apply(np.vstack((Z,u))[index])


def quantile_us(ns):
    values=np.array(ns)*1e-3
    return {'n':len(values),'median_us':float(np.median(values)),'p95_us':float(np.quantile(values,.95)),
            'p99_us':float(np.quantile(values,.99)),'max_us':float(values.max())}


def timing(signal,heads,force_gain,reference,bank_gain=0.,router=None,raw_anchor=None,physical_cap=None):
    ready=Streaming(heads,force_gain,bank_gain,router,raw_anchor,physical_cap)
    baseline=VQF(.005)
    native=np.empty_like(signal[:,:3])
    time_base=[];time_model=[];time_calls=[];time_release=[];cold=[];examples=0
    for i,row in enumerate(signal):
        begin=time.perf_counter_ns()
        baseline.update(row[3:],row[:3])
        Rotation.from_quat(baseline.getQuat6D(),scalar_first=True).inv().apply(Z)
        time_base.append(time.perf_counter_ns()-begin)
        begin=time.perf_counter_ns()
        native[i],is_release=ready.packet(row)
        used=time.perf_counter_ns()-begin
        time_model.append(used)
        if is_release:
            examples+=1
            if examples<=16:
                cold.append(used)
            else:
                time_release.append(used)
                if ready.called:time_calls.append(used)
    target=reference(signal)
    delta=np.degrees(np.arctan2(np.linalg.norm(np.cross(target,native),axis=1),(target*native).sum(1)))
    assert float(delta.max())<1e-4,delta.max()
    return {'baseline_packet':quantile_us(time_base),'all_candidate_packets':quantile_us(time_model),
            'warm_release_packet':quantile_us(time_release),'warm_release_packet_with_head_executed':quantile_us(time_calls),
            'first_release_us':cold[0]/1000,'warmup_releases':len(cold),'native_release_interval_ms':100.,
            'total_wall_per_native_packet_us':float(np.mean(time_model)/1000),
            'batch_one_vs_offline_max_angle_deg':float(delta.max())}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('checkpoint',type=Path,nargs='?')
    parser.add_argument('--gate',choices=['always','rest_off'])
    parser.add_argument('--gain',type=float,default=.25)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection-lock',type=Path)
    args=parser.parse_args()
    torch.set_num_threads(1)
    assert args.device!='cuda' or torch.cuda.is_available()
    if not args.selection_lock and not (args.checkpoint and args.gate):
        parser.error('Supply --selection-lock, or a standalone checkpoint and --gate')
    lock_sha=None
    begin=time.perf_counter_ns()
    if args.selection_lock:
        from locked_replay import SelectedPredictor
        selected=SelectedPredictor(args.selection_lock,args.device)
        if args.checkpoint or args.gate:
            parser.error('The locked recipe supplies every checkpoint and gate')
        lock_sha=hashlib.sha256(args.selection_lock.read_bytes()).hexdigest()
        checkpoints=[h['sha256'] for h in selected.spec['heads']+selected.spec.get('additions',[])]
        if selected.spec.get('response_router'):
            checkpoints.append(selected.spec['response_router']['sha256'])
        heads=selected.stream_heads()
        force_gain=selected.force.get('candidate_deg',0.)
        bank_gain=selected.bank.get('candidate_deg',0.)
        router=selected.router
        reference=lambda s:selected.predict(s,200.,np.arange(len(s)))['candidate_deg']
        recipe=selected.spec
    else:
        checkpoint_hash=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        checkpoints=[checkpoint_hash]
        head=weights(args.checkpoint,'mamba_v1',args.device)
        heads=[(head,args.gain,args.gate)]
        force_gain=0.
        bank_gain=0.
        router=None
        reference=lambda s:batch_reference(s,head,args.gain,args.gate)
        recipe={'kind':'mamba_v1','gate':args.gate,'gain':args.gain}
    if args.device=='cuda':torch.cuda.synchronize()
    load_ms=(time.perf_counter_ns()-begin)/1e6
    src=ROOT/'external-sources/ridi-phone-train/hao_handheld1.npz'
    with np.load(src) as f:
        signal=np.ascontiguousarray(np.c_[f['acc_m_s2'],f['gyr_rad_s']][:6000],dtype='f8')
        assert len(signal)==6000 and f['imu_observed'][:6000].all()
    files=[Path(__file__),Path(__file__).with_name('mixture_replay.py'),Path(__file__).with_name('response_router.py'),
           Path(__file__).with_name('router_motion_context.py'),Path(__file__).with_name('locked_replay.py'),
           Path(__file__).with_name('causal_response_geometry.py'),Path(__file__).with_name('anchored_response_features.py')]
    report={'device':args.device,'computer':platform.node(),'platform':platform.platform(),'cpu':platform.processor(),
            'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'torch_threads':torch.get_num_threads(),
            'torch_version':torch.__version__,'vqf_version':version('vqf'),'checkpoint_hashes':checkpoints,
            'source_code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            'selection_lock_sha256':lock_sha,'checkpoint_load_ms':load_ms,'recipe':recipe,
            'source':str(src.relative_to(ROOT)),'native_packets_per_repeat':len(signal),'repeat_count':3,
            'scope':'200 Hz complete walking packets already in RAM. The same Python process invokes VQF once per '
                    'native packet. The candidate includes the live sphere estimator, block assembly, canonical '
                    'orientation, batch-one head on release, and reconstructed body-frame tilt. Device I/O is excluded.',
            'bias_parity':verify_bias(),'trials':[]}
    raw_anchor=recipe.get('raw_anchor_blend',{}).get('beta')
    physical_cap=recipe.get('physical_angle_cap',{}).get('degrees')
    if physical_cap is not None:
        p=Path(__file__).with_name('physical_angle_cap.py')
        report['source_code_sha256'][p.name]=hashlib.sha256(p.read_bytes()).hexdigest()
    if raw_anchor is not None:
        for filename in ('raw_vqf_blend.py','anchor_blend_features.py'):
            report['source_code_sha256'][filename]=hashlib.sha256(Path(__file__).with_name(filename).read_bytes()).hexdigest()
    for i in range(3):
        report['trials'].append(timing(signal,heads,force_gain,reference,bank_gain,router,raw_anchor,physical_cap))
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(args.device,'p95 release us',[int(r['warm_release_packet_with_head_executed']['p95_us']) for r in report['trials']],
          'max angle',[r['batch_one_vs_offline_max_angle_deg'] for r in report['trials']])


if __name__=='__main__':
    main()
