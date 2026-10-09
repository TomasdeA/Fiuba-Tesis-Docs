#!/usr/bin/env python3
"""Plot recorded occupancy maps and evaluator markers, never synthetic alarms."""
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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, Polygon, Wedge, Patch
from matplotlib.lines import Line2D

def stamp(m):
    return m.header.stamp.sec+m.header.stamp.nanosec/1e9

def load(bag):
    if not (bag/'metadata.yaml').exists():
        raise RuntimeError('Read only a finalized bag: concurrent SQLite reads can interrupt recording')
    con = sqlite3.connect(f'file:{next(bag.glob("*.db3"))}?mode=ro', uri=True)
    topics = {n:(i,get_message(t)) for i,n,t in con.execute('select id,name,type from topics')}
    result = {}
    for name in ['/local_mapper/occupancy_grid','/nav_odom','/spatial_awareness/collision_risk',
                 '/spatial_awareness/debug/markers','/perception/haptic_grid','/perception/depth_grid']:
        tid, cls = topics[name]
        messages = []
        for raw, in con.execute('select data from messages where topic_id=? order by timestamp',(tid,)):
            m = deserialize_message(raw,cls)
            ts = stamp(m) if hasattr(m,'header') else next((stamp(x) for x in m.markers if x.ns=='velocity'),0)
            if ts > 0:
                messages.append((ts,m))
        result[name] = sorted(messages,key=lambda x:x[0])
    con.close()
    return result

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('bag',type=Path)
    p.add_argument('--inspect',action='store_true')
    p.add_argument('--times',nargs=3,type=float)
    args = p.parse_args()
    out = Path(__file__).with_name('salida'); out.mkdir(exist_ok=True)
    data = load(args.bag)
    odom = data['/nav_odom']; t0 = odom[0][0]
    risks = data['/spatial_awareness/collision_risk']
    summary = {'source':str(args.bag),'origin_stamp':t0,'counts':{k:len(v) for k,v in data.items()},'active':[]}
    for ts,m in risks:
        for region in ['left','right','rear']:
            r = getattr(m,region)
            if m.valid and r.active:
                summary['active'].append(dict(t=ts-t0,stamp=ts,region=region,intensity=r.intensity,
                    distance=r.distance_m,closing=r.closing_speed_mps,ttc=r.time_to_collision_s))
    (out/'riesgos_medidos.json').write_text(json.dumps(summary,indent=2))
    print('COUNTS',summary['counts'], 'ACTIVE',len(summary['active']))
    for region in ['left','right','rear']:
        rows = [r for r in summary['active'] if r['region']==region]
        print(region, 'n=',len(rows), 'range=',(rows[0]['t'],rows[-1]['t']) if rows else None,
              'peak=',max(rows,key=lambda r:r['intensity']) if rows else None)
    if args.inspect:
        return
    if not summary['active'] and not args.times:
        raise RuntimeError('No recorded active risk: do not generate an illustrative false alarm')
    markers = data['/spatial_awareness/debug/markers']
    maps = data['/local_mapper/occupancy_grid']
    peak = max(summary['active'],key=lambda r:r['intensity']) if summary['active'] else None
    times = args.times or [max(.5,peak['t']-1.5),peak['t'],min(risks[-1][0]-t0,peak['t']+1.5)]
    diagnostics = json.loads((out/'diagnostico.json').read_text()) if (out/'diagnostico.json').exists() else []
    fig, axes = plt.subplots(1,3,figsize=(14.5,5.6),constrained_layout=True)
    chosen = []
    for ax, target in zip(axes,times):
        ts, marker = min(markers,key=lambda x:abs(x[0]-t0-target))
        risk_ts, risk = min(risks,key=lambda x:abs(x[0]-ts))
        assert risk.valid and abs(risk_ts-ts)<1e-3, 'Markers must match a valid recorded evaluation'
        previous_maps = [(t,m) for t,m in maps if t <= ts]
        mt, grid = previous_maps[-1]
        velocity = next(m for m in marker.markers if m.ns=='velocity')
        fov = next(m for m in marker.markers if m.ns=='fov')
        pos = np.array([velocity.points[0].x,velocity.points[0].z])
        vel = np.array([velocity.points[1].x,velocity.points[1].z])-pos
        edges = [np.array([fov.points[i].x,fov.points[i].z])-pos for i in [1,3]]
        forward = edges[0]+edges[1]; forward /= np.linalg.norm(forward)
        aperture = np.degrees(np.arccos(np.clip(np.dot(edges[0],forward)/np.linalg.norm(edges[0]),-1,1)))+3
        res = grid.info.resolution
        values = np.array(grid.data).reshape(grid.info.height,grid.info.width)
        codes = np.zeros_like(values)
        codes[(values>=0)&(values<=35)] = 1
        codes[(values>35)&(values<65)] = 2
        codes[values>=65] = 3
        origin = grid.info.origin.position
        # Mapper encodes its horizontal X,Z plane using a +90 degree rotation about X.
        q = grid.info.origin.orientation
        assert abs(q.x-2**-.5)<1e-4 and abs(q.w-2**-.5)<1e-4
        ax.imshow(codes,origin='lower',extent=[origin.x,origin.x+grid.info.width*res,
            origin.z,origin.z+grid.info.height*res],cmap=ListedColormap(['#f1f1f1','#d6ebf3','#bec8cf','#b84743']),vmin=0,vmax=3,
            interpolation='nearest',zorder=0)
        theta = np.degrees(np.arctan2(forward[1],forward[0]))
        ax.add_patch(Wedge(pos,1.65,theta-aperture,theta+aperture,facecolor='#57b4dc',alpha=.16,zorder=1))
        for angle in [theta-aperture,theta+aperture]:
            direction = np.array([np.cos(np.radians(angle)),np.sin(np.radians(angle))])
            ax.plot(*np.stack([pos,pos+1.65*direction]).T,color='#208bb0',lw=1)
        for radius in [.4,1.2]:
            ax.add_patch(Circle(pos,radius,fill=False,ls='--',lw=.9,color='#697586'))
        speed = np.linalg.norm(vel)
        if speed >= .1:
            direction = vel/speed; lateral = np.array([-direction[1],direction[0]])*(.2+res/np.sqrt(2))
            poly = np.array([pos-lateral,pos+lateral,pos+1.5*direction+lateral,pos+1.5*direction-lateral])
            ax.add_patch(Polygon(poly,facecolor='#e9ad36',alpha=.2,edgecolor='#c78c17'))
            ax.annotate('',xy=pos+vel,xytext=pos,arrowprops=dict(arrowstyle='->',color='#bb7900',lw=2.4))
        ax.annotate('',xy=pos+.4*forward,xytext=pos,arrowprops=dict(arrowstyle='->',color='#182230',lw=2))
        ax.scatter(*pos,s=45,color='#101828',zorder=6)
        active = []
        names = {'left':'izquierda','right':'derecha','rear':'posterior'}
        for mark in marker.markers:
            if mark.ns=='selected_risk':
                key = ['left','right','rear'][mark.id]
                rr = getattr(risk,key)
                endpoint = np.array([mark.pose.position.x,mark.pose.position.z])
                ix, iz = np.floor((endpoint-np.array([origin.x,origin.z]))/res).astype(int)
                assert rr.active and values[iz,ix]>=65, 'Selected cell is not occupied in the associated map'
                ax.scatter(*endpoint,s=150,facecolors='none',edgecolors='#8a1c78',lw=2.2,zorder=7)
                ax.plot(*np.stack([pos,endpoint]).T,color='#8a1c78',lw=1.4,zorder=4)
                active.append(f"{names[key]}: d={rr.distance_m:.2f} m; cierre={rr.closing_speed_mps:.2f} m/s\nTTC={rr.time_to_collision_s:.2f} s; intensidad={rr.intensity:.0f}/100")
        title = 'Riesgo activo' if active else 'Sin riesgo activo'
        annotation = '\n'.join(active) if active else f'Velocidad: {speed:.2f} m/s'
        diagnostic = min(diagnostics,key=lambda r:abs(r['t']-(ts-t0))) if diagnostics else None
        if not active and diagnostic:
            assert abs(diagnostic['t']-(ts-t0))<1e-3
            if speed<.1:
                title = 'Velocidad bajo el umbral'
                annotation = f'v = {speed:.3f} m/s < 0.10 m/s\nSalida contextual: sin riesgo'
            elif diagnostic['min_ttc'] is not None:
                title = 'TTC fuera del horizonte'
                candidate = diagnostic['candidate']; endpoint=np.array(candidate['position'])
                ax.scatter(*endpoint,s=130,marker='s',facecolors='none',edgecolors='#8a1c78',lw=2,zorder=8)
                ax.plot(*np.stack([pos,endpoint]).T,color='#8a1c78',ls=':',lw=1.3)
                annotation = (f"d = {candidate['distance']:.2f} m; cierre = {candidate['closing']:.2f} m/s\n"
                    f"TTC = {candidate['ttc']:.2f} s ≥ 2 s\nSalida contextual: sin riesgo")
            elif diagnostic['band']>0 and diagnostic['outside']==0:
                title = 'Obstáculo dentro del frente'
                annotation = f"{diagnostic['band']} celdas en banda, todas frontales\nSalida contextual: sin riesgo"
        ax.set_title(f"t = {ts-t0:.2f} s\n"+title,fontsize=12)
        ax.text(.02,.02,annotation,transform=ax.transAxes,fontsize=9,
                bbox=dict(facecolor='white',alpha=.92,edgecolor='none'),va='bottom',zorder=9)
        ax.set(xlim=(pos[0]-1.75,pos[0]+1.75),ylim=(pos[1]-1.75,pos[1]+1.75),xlabel='X en odometría [m]',ylabel='Z en odometría [m]')
        ax.set_aspect('equal'); ax.grid(alpha=.12)
        chosen.append(dict(time_s=ts-t0,stamp=ts,map_stamp=mt,risk_stamp=risk_ts,
                           pose=pos.tolist(),velocity=vel.tolist(),aperture_excluded_deg=aperture,
                           active=bool(active),annotations=annotation,diagnostic=diagnostic))
    handles = [Patch(color='#b84743',label='ocupado (≥65 %)'),Patch(color='#d6ebf3',label='baja ocupación (≤35 %)'),
        Patch(color='#57b4dc',alpha=.3,label='frente excluido'),Patch(color='#e9ad36',alpha=.4,label='corredor de movimiento'),
        Line2D([],[],color='#697586',ls='--',label='banda de distancia'),
        Line2D([],[],marker='o' if summary['active'] else 's',mfc='none',color='#8a1c78',ls='',
               label='celda seleccionada' if summary['active'] else 'candidato descartado por TTC')]
    handles += [Line2D([],[],color='#182230',lw=2,label='orientación frontal'),
                Line2D([],[],color='#bb7900',lw=2,label='velocidad (escala: 1 s)'),
                Patch(color='#f1f1f1',label='desconocido'),
                Patch(color='#bec8cf',label='ocupación intermedia')]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=9,bbox_to_anchor=(.5,-.14))
    fig.savefig(out/'riesgo_contextual_R03.png',dpi=220,bbox_inches='tight')
    (out/'seleccion.json').write_text(json.dumps(chosen,indent=2))
    print('SELECTED',chosen)

if __name__=='__main__':
    main()
