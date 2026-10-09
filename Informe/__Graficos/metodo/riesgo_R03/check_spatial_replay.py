#!/usr/bin/env python3
"""Audit recorded map memory, contextual risk and observable haptic fusion."""
import argparse
import json
from pathlib import Path
import sqlite3
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag',type=Path)
    p.add_argument('--z-min',type=float,default=.25)
    p.add_argument('--z-max',type=float,default=3.)
    args=p.parse_args()
    if not (args.bag/'metadata.yaml').exists():
        raise RuntimeError('Read only a finalized bag; do not lock an active recorder')
    out=args.bag/'chequeo';out.mkdir(exist_ok=True)
    files=list(args.bag.glob('*.db3'))
    assert len(files)==1, 'This audit expects an unsplit bag'
    con=sqlite3.connect(f'file:{files[0]}?mode=ro',uri=True)
    topics={name:(tid,get_message(typ)) for tid,name,typ in con.execute('select id,name,type from topics')}
    def messages(name):
        tid,cls=topics[name]
        for rt,raw in con.execute('select timestamp,data from messages where topic_id=? order by timestamp',(tid,)):
            yield rt,deserialize_message(raw,cls)
    odom=list(messages('/nav_odom'));t0=stamp(odom[0][1])
    risks=list(messages('/spatial_awareness/collision_risk'))
    clocks=list(messages('/clock'))
    clock_rt=np.array([rt for rt,m in clocks],dtype=np.int64)
    clock_s=np.array([m.clock.sec+m.clock.nanosec*1e-9 for rt,m in clocks])
    def age(rt,ts):
        i=np.searchsorted(clock_rt,rt,side='right')-1
        return None if i<0 else float(clock_s[i]-ts)
    active=[]
    for rt,m in risks:
        for region in ['left','right','rear']:
            r=getattr(m,region)
            if m.valid and r.active:
                active.append(dict(t=stamp(m)-t0,region=region,intensity=float(r.intensity),
                    distance=float(r.distance_m),closing=float(r.closing_speed_mps),ttc=float(r.time_to_collision_s)))
    memory=[]
    for rt,g in messages('/local_mapper/occupancy_grid'):
        a=np.array(g.data).reshape(g.info.height,g.info.width);iz,ix=np.indices(a.shape)
        x=g.info.origin.position.x+(ix+.5)*g.info.resolution
        z=g.info.origin.position.z+(iz+.5)*g.info.resolution
        roi=(x>=-.9)&(x<=-.2)&(z>=1.7)&(z<=2.5)
        memory.append(dict(t=stamp(g)-t0,occupied=int(((a>=65)&roi).sum()),
            mean=float(a[roi].mean()),low=int(((a<=35)&(a>=0)&roi).sum())))
    depth={stamp(m):(rt,m) for rt,m in messages('/perception/depth_grid')}
    risk_rt=np.array([rt for rt,m in risks],dtype=np.int64)
    checks=[];best=None;best_delta=-1;rear_samples=[]
    for rt,m in messages('/perception/haptic_grid'):
        ts=stamp(m)
        if ts not in depth:continue
        _,d=depth[ts];shape=(m.rows,m.cols)
        assert shape==(d.rows,d.cols)
        dist=np.array([c.min_m for c in d.cells],dtype=np.float32)
        valid=np.array([c.count>0 and np.isfinite(c.min_m) and c.min_m>0 for c in d.cells])
        front=np.zeros(len(dist),dtype=np.float32)
        lo,hi=np.float32(args.z_min),np.float32(args.z_max)
        front[valid]=(hi-np.clip(dist[valid],lo,hi))/(hi-lo)*np.float32(100)
        actual=np.array(m.intensities,dtype=np.float32);actual_active=np.array(m.active)
        delta=actual-front
        # Recorder ordering does not expose subscriber callback ordering. Search
        # recent published risks, allowing 10 ms recorder scheduling skew.
        indices=np.where((risk_rt<=rt+10_000_000)&(risk_rt>=rt-750_000_000))[0][::-1]
        match=None
        for j in indices:
            rrt,r=risks[j];expected=front.copy().reshape(shape);enabled=valid.copy().reshape(shape)
            if r.valid:
                for key,cols in [('left',[0]),('right',[m.cols-1]),('rear',list(range((m.cols-2)//2,(m.cols-2)//2+2)))]:
                    v=getattr(r,key)
                    if v.active and v.intensity>0:
                        for col in cols:
                            expected[:,col]=np.maximum(expected[:,col],np.clip(v.intensity,0,100));enabled[:,col]=True
            if np.allclose(actual,expected.ravel(),atol=2e-4,rtol=0) and np.array_equal(actual_active,enabled.ravel()):
                match=dict(risk_t=stamp(r)-t0,regions=[key for key in ['left','right','rear'] if r.valid and getattr(r,key).active],
                    recorded_gap_s=(rt-rrt)*1e-9);break
        frontal_match=np.allclose(actual,front,atol=2e-4,rtol=0) and np.array_equal(actual_active,valid)
        extra=int((delta>2e-4).sum())
        check=dict(t=ts-t0,extra_cells=extra,max_increase=float(delta.max()),
            matched_risk=match,frontal_only=bool(frontal_match),source_age_s=age(rt,ts))
        # Inspect rear-active messages published immediately before this output,
        # even when max(front, risk) makes the contextual overlay invisible.
        past=np.where((risk_rt<=rt)&(risk_rt>=rt-150_000_000))[0]
        if len(past):
            _,rr=risks[past[-1]]
            if rr.valid and rr.rear.active:
                central=list(range((m.cols-2)//2,(m.cols-2)//2+2))
                rear_samples.append(dict(t=ts-t0,risk_t=stamp(rr)-t0,
                    risk_intensity=float(rr.rear.intensity),source_age_s=age(rt,ts),
                    frontal_min_central=float(front.reshape(shape)[:,central].min()),
                    extra_cells=extra,frontal_only=bool(frontal_match),
                    front=front.reshape(shape).tolist(),haptic=actual.reshape(shape).tolist()))
        checks.append(check)
        if match and extra and delta.max()>best_delta:
            best_delta=float(delta.max())
            best=dict(**check,front=front.reshape(shape).tolist(),haptic=actual.reshape(shape).tolist(),
                increment=delta.reshape(shape).tolist())
    info=list(messages('/odom_info'))
    invalid=sum(bool(m.lost) for rt,m in info)
    summary=dict(bag=str(args.bag),t0=t0,map_count=len(memory),odom_count=len(odom),
        odom_lost=invalid,risk_count=len(risks),valid_risks=sum(m.valid for rt,m in risks),
        active=active,memory=memory,haptic_checks=checks,best_fusion=best,rear_samples=rear_samples,
        haptic_matched=sum(r['matched_risk'] is not None or r['frontal_only'] for r in checks),
        haptic_extra_frames=sum(r['extra_cells']>0 for r in checks),
        note='Fusion matching searches recorded messages; it does not expose exact callback causality.')
    (out/'resultado.json').write_text(json.dumps(summary,indent=2))
    print('COUNTS', {k:v for k,v in summary.items() if k in ['map_count','odom_count','odom_lost','risk_count','valid_risks','haptic_matched','haptic_extra_frames']})
    for region in ['left','right','rear']:
        rows=[r for r in active if r['region']==region]
        print(region,len(rows),'range',None if not rows else [rows[0]['t'],rows[-1]['t']],
            'peak',None if not rows else max(rows,key=lambda r:r['intensity']))
    for target in [30,37,40,65,85,88,90,92,98]:
        print('MEMORY',min(memory,key=lambda r:abs(r['t']-target)))
    if best:
        print('FUSION', {k:v for k,v in best.items() if k not in ['front','haptic','increment']})
        fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
        for ax,key,title in zip(axes,['front','haptic','increment'],['Percepción frontal','Salida háptica publicada','Incremento contextual']):
            a=np.array(best[key]);im=ax.imshow(a,vmin=0,vmax=100,cmap='viridis',aspect='auto')
            ax.set_title(title);ax.set_xlabel('Columna')
            for row,col in np.ndindex(a.shape):ax.text(col,row,f'{a[row,col]:.0f}',ha='center',va='center',fontsize=8,color='white' if a[row,col]<45 else 'black')
        fig.colorbar(im,ax=axes,label='Intensidad [0–100]')
        fig.suptitle(f"R03: evidencia publicada a t={best['t']:.2f} s")
        fig.savefig(out/'fusion_verificada.png',dpi=180)
    print('REAR FUSION', [{k:v for k,v in r.items() if k not in ['front','haptic']} for r in rear_samples])
    for topic in ['/local_mapper/occupancy_grid','/nav_odom','/perception/depth_grid','/perception/haptic_grid']:
        ages=[age(rt,stamp(m)) for rt,m in messages(topic)]
        ages=[a for a in ages if a is not None]
        print('AGE',topic,'median/p95/max',np.percentile(ages,[50,95,100]).tolist())
    con.close()

if __name__=='__main__':main()
