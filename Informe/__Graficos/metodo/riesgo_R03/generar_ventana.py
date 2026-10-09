#!/usr/bin/env python3
"""Rolling local map with the recorded path, in fixed odometry axes."""
import argparse,json,sqlite3
from pathlib import Path
import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle,Circle,Patch
from matplotlib.lines import Line2D
p=argparse.ArgumentParser();p.add_argument('bag',type=Path);p.add_argument('--output',type=Path);p.add_argument('--accumulated-bag',type=Path);args=p.parse_args()
out=args.output or Path(__file__).with_name('salida');out.mkdir(exist_ok=True,parents=True)
c=sqlite3.connect(f'file:{next(args.bag.glob("*.db3"))}?mode=ro',uri=True)
topics={n:(i,get_message(t)) for i,n,t in c.execute('select id,name,type from topics')}
def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec/1e9
def messages(name):
 tid,cls=topics[name]
 return [deserialize_message(raw,cls) for raw, in c.execute('select data from messages where topic_id=? order by timestamp',(tid,))]
odom=sorted(messages('/nav_odom'),key=stamp);t0=stamp(odom[0]);ot=np.array([stamp(m) for m in odom])
poses=np.array([[-m.pose.pose.position.y,m.pose.pose.position.x] for m in odom])
maps=messages('/local_mapper/occupancy_grid');markers={}
for m in messages('/spatial_awareness/debug/markers'):
 v=next((v for v in m.markers if v.ns=='velocity'),None)
 if v:markers[stamp(v)]=m
cmap=ListedColormap(['#dedede','#f7f7f7','#aab7bc','#262626'])
# The second snapshot follows the deliberate right-alarm episode.
risks=messages('/spatial_awareness/collision_risk')
right_end=max(stamp(r)-t0 for r in risks if r.valid and r.right.active and 61<=stamp(r)-t0<=78)
second_t=right_end+1.
# Select by the actual window centre, not by the sensor position or array index.
intermediate=min((g for g in maps if stamp(g)-t0>second_t),
 key=lambda g:abs(g.info.origin.position.x+g.info.width*g.info.resolution/2+7))
times=[33.,second_t,stamp(intermediate)-t0,108.]
labels=['Reconstrucción inicial','Después de la alarma derecha',
        'Durante el alejamiento','Final del recorrido']
plot_xlim=(-21,7);plot_ylim=(-6,14)
global_summary=None
if args.accumulated_bag:
 gc=sqlite3.connect(f'file:{next(args.accumulated_bag.glob("*.db3"))}?mode=ro',uri=True)
 tid,typ=gc.execute("select id,type from topics where name='/local_mapper/occupancy_grid'").fetchone()
 cls=get_message(typ)
 # All possible observations within the unchanged 5 m sensing radius must fit
 # every published enlarged window. This is stronger than checking only its end.
 bound_lo=poses.min(axis=0)-5.;bound_hi=poses.max(axis=0)+5.
 global_map=None;global_stamps=[];global_count=0;minimum_margin=float('inf')
 for raw, in gc.execute('select data from messages where topic_id=? order by timestamp',(tid,)):
  g=deserialize_message(raw,cls);o=g.info.origin.position
  lo=np.array([o.x,o.z]);hi=lo+g.info.width*g.info.resolution
  margin=float(min((bound_lo-lo).min(),(hi-bound_hi).min()))
  assert margin>g.info.resolution, 'Accumulated grid too small: observations could be clipped'
  minimum_margin=min(minimum_margin,margin);global_map=g;global_count+=1;global_stamps.append(stamp(g))
 assert global_map is not None
 assert sorted(global_stamps)==sorted(stamp(g) for g in maps), 'Different observations integrated in accumulated replay'
 # Verify that the replay did not substitute another odometry estimate.
 tid,typ=gc.execute("select id,type from topics where name='/nav_odom'").fetchone();cls=get_message(typ)
 replay_odom=[deserialize_message(raw,cls) for raw, in gc.execute('select data from messages where topic_id=? order by timestamp',(tid,))]
 original={stamp(m):m for m in odom}
 assert len(replay_odom)==len(odom), 'Odometry delivery incomplete'
 for m in replay_odom:
  assert stamp(m) in original and m.pose==original[stamp(m)].pose
 global_summary=dict(bag=str(args.accumulated_bag),t=stamp(global_map)-t0,
  grid_size=global_map.info.width,resolution=global_map.info.resolution,
  map_count=global_count,minimum_boundary_margin_m=minimum_margin,
  same_odometry_verified=True,same_observation_stamps_verified=True,sensor_range_m=5.,forget_radius_m=0.)
 fig=plt.figure(figsize=(12,12));gs=fig.add_gridspec(3,2,height_ratios=[1.2,1,1],hspace=.45,wspace=.16)
 ax=fig.add_subplot(gs[0,:]);g=global_map;o=g.info.origin.position
 data=np.array(g.data).reshape(g.info.height,g.info.width);codes=np.zeros_like(data)
 codes[(data>=0)&(data<=35)]=1;codes[(data>35)&(data<65)]=2;codes[data>=65]=3
 size=g.info.width*g.info.resolution
 ax.imshow(codes,origin='lower',extent=[o.x,o.x+size,o.z,o.z+size],
  cmap=cmap,vmin=0,vmax=3,interpolation='nearest')
 mask=ot<=stamp(g);path=poses[mask];ax.plot(path[:,0],path[:,1],color='#30952d',lw=1.7)
 ax.scatter(*path[0],marker='s',s=22,color='#30952d');ax.scatter(*path[-1],s=30,color='#2357ad')
 ax.set(xlim=plot_xlim,ylim=plot_ylim,xlabel='X en odometría [m]',ylabel='Z en odometría [m]')
 # Check that the plotted bounds include all observed cells, not just the path.
 iz,ix=np.where(data>=0);wx=o.x+(ix+.5)*g.info.resolution;wz=o.z+(iz+.5)*g.info.resolution
 plot_xlim=(min(-21.,float(np.floor(wx.min()-1))),max(7.,float(np.ceil(wx.max()+1))))
 plot_ylim=(min(-6.,float(np.floor(wz.min()-1))),max(14.,float(np.ceil(wz.max()+1))))
 ax.set(xlim=plot_xlim,ylim=plot_ylim)
 global_summary['observed_bounds']=[float(wx.min()),float(wx.max()),float(wz.min()),float(wz.max())]
 global_summary['plot_xlim']=plot_xlim;global_summary['plot_ylim']=plot_ylim
 ax.set_title(f'Mapa acumulado del recorrido · t = {stamp(g)-t0:.2f} s\nSin descarte por distancia',fontsize=11)
 ax.grid(alpha=.15)
else:
 fig=plt.figure(figsize=(12,9));gs=fig.add_gridspec(2,2,hspace=.38,wspace=.16)
selection=[]
for col,t in enumerate(times):
 g=min(maps,key=lambda g:abs(stamp(g)-t0-t));ts=stamp(g);o=g.info.origin.position;size=g.info.width*g.info.resolution
 ax=fig.add_subplot(gs[col//2+(1 if args.accumulated_bag else 0),col%2]);a=np.array(g.data).reshape(g.info.height,g.info.width);codes=np.zeros_like(a)
 codes[(a>=0)&(a<=35)]=1;codes[(a>35)&(a<65)]=2;codes[a>=65]=3
 ax.imshow(codes,origin='lower',extent=[o.x,o.x+size,o.z,o.z+size],cmap=cmap,vmin=0,vmax=3,interpolation='nearest')
 ax.add_patch(Rectangle((o.x,o.z),size,size,fill=False,edgecolor='#627a84',lw=1.5))
 mask=ot<=ts;path=poses[mask];pos=path[-1];ax.plot(path[:,0],path[:,1],color='#30952d',lw=1.7)
 m=markers[min(markers,key=lambda s:abs(s-ts))];f=next(m for m in m.markers if m.ns=='fov');v=next(m for m in m.markers if m.ns=='velocity')
 mp=np.array([v.points[0].x,v.points[0].z]);forward=sum(np.array([f.points[i].x,f.points[i].z])-mp for i in [1,3]);forward/=np.linalg.norm(forward)
 ax.scatter(*pos,color='#2357ad',s=30,zorder=5);ax.annotate('',xy=pos+1.2*forward,xytext=pos,arrowprops=dict(arrowstyle='->',color='#2357ad',lw=2))
 ax.scatter(*poses[0],marker='s',s=22,color='#30952d')
 ax.set(xlim=plot_xlim,ylim=plot_ylim,xlabel='X en odometría [m]',ylabel='Z en odometría [m]')
 ax.set_title(f'{labels[col]} · t = {ts-t0:.2f} s\nCentro de la ventana: ({o.x+size/2:.1f}, {o.z+size/2:.1f}) m',fontsize=11)
 ax.grid(alpha=.15);selection.append(dict(t=ts-t0,origin=[o.x,o.z],window_center=[o.x+size/2,o.z+size/2],pose=pos.tolist()))
fig.suptitle('Mapa acumulado y ventanas locales' if args.accumulated_bag else 'Ventana local y camino recorrido en ejes fijos',fontsize=16,fontweight='bold')
fig.legend(handles=[Patch(facecolor='#262626',label='Ocupado'),Patch(facecolor='#f7f7f7',edgecolor='#bbb',label='Libre'),
 Patch(facecolor='#dedede',label='Desconocido'),Patch(facecolor='#aab7bc',label='Intermedio'),Line2D([],[],color='#30952d',label='Trayectoria'),
 Line2D([],[],marker='o',color='#2357ad',label='Sensor y frente')],loc='lower center',ncol=6,fontsize=10)
fig.subplots_adjust(left=.065,right=.98,top=.92,bottom=.07)
fig.savefig(out/'ventana_trayectoria.png',dpi=220)
(out/'ventana_trayectoria.json').write_text(json.dumps(dict(bag=str(args.bag),t0=t0,samples=selection,
 right_episode_last_active_s=right_end,intermediate_target_center_x_m=-7.,accumulated=global_summary),indent=2))
print(out/'ventana_trayectoria.png')
