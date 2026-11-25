"""Packet-by-packet C038 timing on the original near-1 kHz SHICE clock, including missing input rows."""
import argparse
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from vqf import VQF

from lock_from_frontier import ROOT, REVIEW, digest, read
from locked_replay import SelectedPredictor
from native_catalog import Catalog
from optical_score import infer_canonical
from response_router import alternatives
from stream_latency import StreamBias, Z, quantile_us


class NativeC038:
    def __init__(self, selected, hz):
        assert hz > 100 and selected.router and not selected.router.context
        assert set(selected.spec).isdisjoint({"raw_anchor_blend","physical_angle_cap"})
        self.heads = selected.stream_heads()
        assert all(getattr(w[0], "history_scale", 1)==1 for w, _, _ in self.heads)
        self.router = selected.router
        self.gain = selected.force["candidate_deg"]
        self.hz = hz
        self.central, self.fast, self.slow = VQF(1/hz), VQF(1/hz,tauAcc=1.5), VQF(1/hz,tauAcc=6.)
        self.bias = StreamBias(hz)
        self.last_input = np.zeros(6)
        self.alpha = -np.expm1(-1./(.25*hz))
        self.force = np.zeros(3)
        self.last_force = -1
        self.index = -1
        self.blocks = 0
        self.start = 0
        self.stop = int(np.floor(hz/100.+1e-9))
        self.block_sum = np.zeros(6)
        self.ring = np.zeros((64,13),dtype="f4")
        self.pending = None
        self.vertical = Z.copy()

    def publish(self, packet):
        _, window, central, fast, slow, rest, stable_force = packet
        world = Z.copy()
        for model, gain, gate in self.heads:
            if gate == "always" or not rest:
                world += gain * central.apply(infer_canonical(model,window)[0])
        world += stable_force
        if not rest:
            f,s = alternatives(central, fast, slow)
            probabilities, valid = self.router.probabilities(window, None, (f,s))
            if valid[0]:
                world += probabilities[0,0]*f+probabilities[0,1]*s
        return world/np.linalg.norm(world)

    def ingest(self, row):
        self.index += 1
        seen = np.isfinite(row).all()
        if seen:
            self.last_input = row
        clean = self.last_input.copy()
        clean[:3] -= self.bias.offset
        self.bias.ingest(row)
        for filt in (self.central,self.fast,self.slow):
            filt.update(clean[3:],clean[:3])
        central = Rotation.from_quat(self.central.getQuat6D(),scalar_first=True)
        size = np.linalg.norm(clean[:3])
        if seen and size>1e-12:
            self.force += self.alpha*(central.apply(clean[:3]/size)-self.force)
            self.last_force = self.index
        self.block_sum += clean
        if self.index+1 == self.stop:
            pos = self.blocks % 64
            self.ring[pos,:6] = self.block_sum/(self.stop-self.start)
            self.ring[pos,6:9] = central.inv().apply(Z)
            self.ring[pos,9:12] = self.central.getBiasEstimate()[0]
            rest = self.central.getRestDetected()
            self.ring[pos,12] = rest
            self.blocks += 1
            if self.blocks>=70 and self.blocks%10==0:
                release = int(np.ceil(self.blocks*(self.hz/100.)-1e-9))-1
                first_input = int(np.floor((self.blocks-64)*(self.hz/100.)+1e-9))
                stable_force = np.zeros(3)
                length = np.linalg.norm(self.force)
                if not rest and self.last_force>=first_input and length>0 and np.sqrt(2.*max(0.,1.-length))<np.deg2rad(3.):
                    unit = self.force/length
                    unit[2] = 0.
                    stable_force = unit*min(self.gain,np.tan(np.deg2rad(6.))/max(np.linalg.norm(unit),1e-15))
                assert self.pending is None and release in (self.index,self.index+1)
                self.pending=(release,np.roll(self.ring,-(pos+1),axis=0)[None,:,:],central,
                              Rotation.from_quat(self.fast.getQuat6D(),scalar_first=True),
                              Rotation.from_quat(self.slow.getQuat6D(),scalar_first=True),rest,stable_force)
            self.block_sum.fill(0.)
            self.start = self.stop
            self.stop = int(np.floor((self.blocks+1)*(self.hz/100.)+1e-9))
        released = self.pending is not None and self.pending[0]==self.index
        if released:
            self.vertical=self.publish(self.pending)
            self.pending=None
        return central.inv().apply(self.vertical),released


def directions_gap(a,b):
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b),axis=1),np.sum(a*b,axis=1)))


def run_once(x,hz,selected,expected):
    native = NativeC038(selected,hz)
    reference = VQF(1/hz)
    last = np.zeros(6)
    durations = np.empty((len(x),2),dtype="i8")
    vectors = {key:np.empty((len(x),3)) for key in ("full_vqf_deg","candidate_deg")}
    releases = []
    for j,row in enumerate(x):
        begin=perf_counter_ns()
        if np.isfinite(row).all():last=row
        reference.update(last[3:],last[:3])
        vectors["full_vqf_deg"][j]=Rotation.from_quat(reference.getQuat6D(),scalar_first=True).inv().apply(Z)
        durations[j,0]=perf_counter_ns()-begin
        begin=perf_counter_ns()
        vectors["candidate_deg"][j],release=native.ingest(row)
        durations[j,1]=perf_counter_ns()-begin
        if release:releases.append(j)
    exact_indices=np.ceil(np.arange(70, int(np.floor(len(x)*100/hz+1e-9))+1,10)*(hz/100.)-1e-9).astype("i8")-1
    np.testing.assert_array_equal(releases,exact_indices)
    gaps={k:float(directions_gap(v,expected[k]).max()) for k,v in vectors.items()}
    if max(gaps.values())>=1e-4:
        raise ValueError(f"Online and offline predictions differ {gaps}")
    warm=exact_indices[16:]
    return {"packets":len(x),"release_count":len(warm)+16,"observed_missing_input_rows":int((~np.isfinite(x).all(1)).sum()),
            "default_packet":quantile_us(durations[:,0]),"candidate_packet":quantile_us(durations[:,1]),
            "first_release_us":durations[releases[0],1]/1000.,"warm_release":quantile_us(durations[warm,1]),
            "packets_over_native_interval":int(np.sum(durations[:,1]>1e9/hz)),
            "native_interval_us":1e6/hz,"max_online_offline_gap_deg":gaps}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--device",choices=("cpu","cuda"),required=True)
    ap.add_argument("--check",action="store_true")
    args=ap.parse_args()
    torch.set_num_threads(1)
    lock=ROOT/"selection-lock.json"
    assert read(lock)["selected"]["batch"]=="C038"
    selected=SelectedPredictor(lock,args.device)
    catalog=Catalog(lock)
    names=read(REVIEW/"final-anchor-spec.json")["anchors"]["SHICE"]
    assert names==["R5-1","R6-1"]
    seconds,repeats=(5,1) if args.check else (60,3)
    trials=[]
    for name in names:
        (full,hz,_,_),_=catalog.native("SHICE",name)
        x=full[:int(np.floor(seconds*hz))]
        indices=np.arange(len(x))
        expected=selected.predict(x,hz,indices)
        for number in range(1,repeats+1):
            result=run_once(x,hz,selected,expected)
            result.update(record=name,hz=hz,repeat=number)
            trials.append(result)
            print(name,number,round(result["warm_release"]["p95_us"]/1000,2),result["max_online_offline_gap_deg"],flush=True)
    if not args.check:
        out={"selection_sha256":digest(lock),"script_sha256":digest(Path(__file__)),"device":args.device,
             "clock_scope":"First 60 s of R5-1 and R6-1, each from native sample 0; three independent initializations per device backend. Real input missingness, previous observed input fill, continuous native filter and bias state, exact non-integer 100 Hz block boundaries, 10 Hz frozen head. Python software timings, all input in memory. Baseline uses one VQF; candidate uses the complete C038 recipe including all physical filters.",
             "prefix_seconds":seconds,"repeat_count":repeats,"rows":trials}
        (REVIEW/("shice-native-latency-"+args.device+".json")).write_text(json.dumps(out,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
