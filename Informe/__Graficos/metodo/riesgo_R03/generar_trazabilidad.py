#!/usr/bin/env python3
"""Trace a classified depth point into angular and spatial cells, then rear risk."""
import argparse,json,sqlite3,struct,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
sys.path.insert(0,'/home/tomasdea/Tesis/tools/experimentos')
from generar_contacto_fases import decode_depth
p=argparse.ArgumentParser();p.add_argument('bag',type=Path);p.add_argument('source',type=Path);args=p.parse_args()
out=Path(__file__).with_name('salida');out.mkdir(exist_ok=True)
def stamp(m):return m.header.stamp.sec+m.header.stamp.nanosec/1e9
class Bag:
 def __init__(self,path):
  assert (path/'metadata.yaml').exists()
  self.c=sqlite3.connect(f'file:{next(path.glob("*.db3"))}?mode=ro',uri=True)
  self.topics={n:(i,get_message(t)) for i,n,t in self.c.execute('select id,name,type from topics')};self.indices={}
 def index(self,name):
  if name not in self.indices:
   tid,_=self.topics[name];rows=[]
   for mid,raw,receipt in self.c.execute('select id,substr(data,1,12),timestamp from messages where topic_id=? order by timestamp',(tid,)):
    sec,ns=struct.unpack_from(('<' if raw[1]==1 else '>')+'iI',raw,4);rows.append((sec+ns/1e9,mid,receipt))
   self.indices[name]=rows
  return self.indices[name]
 def get(self,name,row):
  raw=self.c.execute('select data from messages where id=?',(row[1],)).fetchone()[0]
  return deserialize_message(raw,self.topics[name][1])
 def near(self,name,t):return self.get(name,min(self.index(name),key=lambda row:abs(row[0]-t)))
 def all(self,name):
  tid,cls=self.topics[name]
  return [(rt,deserialize_message(raw,cls)) for rt,raw in self.c.execute('select timestamp,data from messages where topic_id=? order by timestamp',(tid,))]
b=Bag(args.bag);src=Bag(args.source);t0=min(x[0] for x in b.index('/nav_odom'))
risks=[r for _,r in b.all('/spatial_awareness/collision_risk') if r.valid and r.rear.active and not r.left.active and not r.right.active and 45<stamp(r)-t0<49]
assert risks,'No posterior risk in the user-confirmed interval'
risk=max(risks,key=lambda r:r.rear.intensity);rt=stamp(risk)
marker=min((m for _,m in b.all('/spatial_awareness/debug/markers') if any(x.ns=='velocity' for x in m.markers)),
           key=lambda m:abs(stamp(next(x for x in m.markers if x.ns=='velocity'))-rt))
target=next(x for x in marker.markers if x.ns=='selected_risk' and x.id==2 and x.action==0)
target_world=np.array([target.pose.position.x,target.pose.position.z])
tf={}
for _,m in b.all('/tf'):
 for tr in m.transforms:
  if tr.child_frame_id=='camera_depth_optical_frame' and tr.header.frame_id=='gravity_aligned_frame':tf[stamp(tr)]=tr.transform.rotation
def cloud(m):
 assert m.point_step==12 and not m.is_bigendian
 return np.frombuffer(bytes(m.data),dtype='<f4').reshape(-1,3)
R_ros_internal=np.array([[0,-1,0],[0,0,-1],[1,0,0]])
odom_rows=sorted(b.index('/nav_odom'))
odom_times=np.array([row[0] for row in odom_rows])
def pose_sample(row):
 m=b.get('/nav_odom',row);q=m.pose.pose.orientation
 f=R_ros_internal@Rotation.from_quat([q.x,q.y,q.z,q.w]).apply([0,0,1])
 return np.array([-m.pose.pose.position.y,m.pose.pose.position.x]),np.arctan2(f[0],f[2])
def pose_at(ts):
 i=int(np.searchsorted(odom_times,ts))
 if i<len(odom_rows) and abs(odom_times[i]-ts)<1e-6:
  pos,yaw=pose_sample(odom_rows[i]);return pos,yaw,ts
 assert 0<i<len(odom_rows),'No pose bracket'
 gap=odom_times[i]-odom_times[i-1]
 assert 0<gap<=.200001,'Pose bracket exceeds mapper limit'
 a=(ts-odom_times[i-1])/gap;p0,y0=pose_sample(odom_rows[i-1]);p1,y1=pose_sample(odom_rows[i])
 dy=np.arctan2(np.sin(y1-y0),np.cos(y1-y0))
 return (1-a)*p0+a*p1,y0+a*dy,ts
def world_points(points,ts):
 pos,yaw,_=pose_at(ts);x,z=points[:,0],points[:,2]
 return np.column_stack((np.cos(yaw)*x+np.sin(yaw)*z,-np.sin(yaw)*x+np.cos(yaw)*z))+pos
# Select a point observed early that falls into the SAME world cell as later risk.
found=None
for desired in [3,4,5,6,7,8]:
 row=min(b.index('/depth_obstacle_filter/obstacle_cloud'),key=lambda row:abs(row[0]-t0-desired));ts=row[0]
 g=b.near('/local_mapper/occupancy_grid',ts);d=b.near('/perception/depth_grid',ts)
 if abs(stamp(g)-ts)>.001 or abs(stamp(d)-ts)>.001 or ts not in tf:continue
 pts=cloud(b.get('/depth_obstacle_filter/obstacle_cloud',row));world=world_points(pts,ts)
 o=np.array([g.info.origin.position.x,g.info.origin.position.z]);res=g.info.resolution
 cell=np.floor((target_world-o)/res).astype(int);cells=np.floor((world-o)/res).astype(int)
 matching=np.where(np.all(cells==cell,axis=1))[0]
 if not len(matching):continue
 # Height is unstamped: use recorder receipt relative to the cloud receipt.
 heights=[(receipt,m.data) for receipt,m in b.all('/depth_obstacle_filter/camera_height') if receipt<=row[2]]
 if not heights:continue
 ymax=heights[-1][1]+.1
 candidates=[i for i in matching if -.2<=pts[i,1]<ymax and .25<=pts[i,2]<=5 and abs(np.degrees(np.arctan2(pts[i,0],pts[i,2])))<44]
 if not candidates:continue
 index=candidates[len(candidates)//2];point=pts[index].astype(float)
 q=tf[ts];rot=Rotation.from_quat([q.x,q.y,q.z,q.w]);optical=rot.inv().apply(point)
 info=src.near('/camera/camera/depth/camera_info',ts);depthmsg=src.near('/camera/camera/depth/image_rect_raw',ts)
 if abs(stamp(depthmsg)-ts)>.001:continue
 depth=decode_depth(depthmsg);u=int(round(info.k[0]*optical[0]/optical[2]+info.k[2]));v=int(round(info.k[4]*optical[1]/optical[2]+info.k[5]))
 if not (0<=v<depth.shape[0] and 0<=u<depth.shape[1]):continue
 z=depth[v,u];reproject=np.array([(u-info.k[2])*z/info.k[0],(v-info.k[5])*z/info.k[4],z])
 err=float(np.linalg.norm(rot.apply(reproject)-point))
 if err>1e-4:continue
 # Reconstruct high-resolution horizontal bins, then reduce their centers.
 angles=np.degrees(np.arctan2(pts[:,0],pts[:,2]));hc=np.floor(angles+70).astype(int)
 cc=np.floor((hc-70+.5+45)/9).astype(int);rr=np.floor((pts[:,1]+.2)/(ymax+.2)*d.rows).astype(int)
 valid=(pts[:,2]>=.25)&(pts[:,2]<=5)&(pts[:,1]>=-.2)&(pts[:,1]<ymax)&(hc>=0)&(hc<140)&(cc>=0)&(cc<d.cols)&(rr>=0)&(rr<d.rows)
 ri,ci=int(rr[index]),int(cc[index]);members=valid&(rr==ri)&(cc==ci);dist=np.hypot(pts[members,0],pts[members,2]);dc=d.cells[ri*d.cols+ci]
 if int(members.sum())!=dc.count or not np.isclose(dist.min(),dc.min_m,atol=2e-4):continue
 found=(ts,g,d,pts,world,point,world[index],cell,ymax,depth,u,v,optical,ri,ci,err,dc)
 break
assert found,'No fully verified pixel/class/angular/map correspondence found'
ts,g,d,pts,world,point,wp,cell,ymax,depth,u,v,optical,ri,ci,err,dc=found
assert np.array(g.data).reshape(g.info.height,g.info.width)[cell[1],cell[0]]>=65
# Find a recorded haptic output compatible with this rear risk and its depth grid.
fusion=None
for row in b.index('/perception/haptic_grid'):
 if not rt-.6<row[0]<=rt+.05:continue
 h=b.get('/perception/haptic_grid',row);frontmsg=b.near('/perception/depth_grid',stamp(h))
 if abs(stamp(frontmsg)-stamp(h))>.001:continue
 front=np.array([max(0,min(100,(3-x.min_m)/2.75*100)) if x.count and np.isfinite(x.min_m) and x.min_m>0 else 0. for x in frontmsg.cells],dtype=float).reshape(h.rows,h.cols)
 expected=front.copy();cols=[(h.cols-2)//2,(h.cols-2)//2+1];expected[:,cols]=np.maximum(expected[:,cols],risk.rear.intensity)
 actual=np.array(h.intensities).reshape(h.rows,h.cols)
 if np.allclose(actual,expected,atol=2e-4):fusion=(h,front,actual);break
assert fusion,'No matching published haptic grid'
h,front,actual=fusion
late=b.get('/local_mapper/occupancy_grid',max((x for x in b.index('/local_mapper/occupancy_grid') if x[0]<=rt),key=lambda x:x[0]))
late_origin=np.array([late.info.origin.position.x,late.info.origin.position.z])
late_cell=np.floor((target_world-late_origin)/late.info.resolution).astype(int)
assert np.array(late.data).reshape(late.info.height,late.info.width)[late_cell[1],late_cell[0]]>=65
fig,axes=plt.subplots(2,3,figsize=(14,9));fig.subplots_adjust(left=.065,right=.94,top=.9,bottom=.09,wspace=.35,hspace=.48)
ax=axes[0,0];im=ax.imshow(np.ma.masked_where(depth<=0,depth),vmin=.25,vmax=5,cmap='turbo');fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Z axial [m]');ax.scatter(u,v,facecolors='none',edgecolors='magenta',s=130,lw=2)
ax.set(title=f'1. Píxel de profundidad · t={ts-t0:.2f} s',xlabel='u [píxeles]',ylabel='v [píxeles]');ax.text(.02,.98,f'({u}, {v}) · Z={depth[v,u]:.3f} m',transform=ax.transAxes,va='top',bbox=dict(facecolor='white',alpha=.85))
ax=axes[0,1];ground=cloud(b.near('/depth_obstacle_filter/debug/ground_cloud',ts))
ax.scatter(ground[::30,0],ground[::30,2],s=1,c='#aabbbb',label='Suelo');ax.scatter(pts[::5,0],pts[::5,2],s=2,c='#c54a35',label='Obstáculo');ax.scatter(point[0],point[2],c='magenta',s=50)
ax.set(title='2. Punto 3D clasificado como obstáculo',xlabel='X alineado [m]',ylabel='Z alineado [m]');ax.legend(fontsize=8,loc='lower right')
ax.text(.02,.98,f'X={point[0]:.3f}; Y={point[1]:.3f}; Z={point[2]:.3f} m',transform=ax.transAxes,va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.8))
values=np.array([x.min_m if x.count else np.nan for x in d.cells]).reshape(d.rows,d.cols)
ax=axes[0,2];im=ax.imshow(values,cmap='viridis',vmin=.25,vmax=7,aspect='auto');fig.colorbar(im,ax=ax,fraction=.035,pad=.02,label='Distancia XZ [m]');ax.add_patch(Rectangle((ci-.5,ri-.5),1,1,fill=False,edgecolor='magenta',lw=2))
for ij in np.ndindex(values.shape):
 if np.isfinite(values[ij]):ax.text(ij[1],ij[0],f'{values[ij]:.1f}',ha='center',va='center',fontsize=7,color='white')
ax.set(title=f'3. DepthGrid · celda ({ri}, {ci})',xlabel='Columna angular',ylabel='Fila por altura');ax.text(.02,-.25,f'n={dc.count}; mínimo={dc.min_m:.3f} m\nEl punto contribuye; la celda agrega otras muestras.',transform=ax.transAxes,fontsize=9)
def plotmap(ax,grid,title):
 a=np.array(grid.data).reshape(grid.info.height,grid.info.width);codes=np.zeros_like(a);codes[(a>=0)&(a<=35)]=1;codes[(a>35)&(a<65)]=2;codes[a>=65]=3
 o=grid.info.origin.position;size=grid.info.width*grid.info.resolution
 ax.imshow(codes,origin='lower',extent=[o.x,o.x+size,o.z,o.z+size],cmap=ListedColormap(['#ddd','#e0eff2','#aabbbb','#b84743']),vmin=0,vmax=3)
 pos,yaw,_=pose_at(stamp(grid));ax.scatter(*pos,c='black',s=25);ax.annotate('',xy=pos+.5*np.array([np.sin(yaw),np.cos(yaw)]),xytext=pos,arrowprops=dict(arrowstyle='->',color='black',lw=2))
 ax.scatter(*target_world,facecolors='none',edgecolors='magenta',s=100,lw=2)
 ax.set(xlim=(-1.5,2.5),ylim=(-.5,4.5),title=title,xlabel='X en odometría [m]',ylabel='Z en odometría [m]')
plotmap(axes[1,0],g,f'4. Celda del mapa ({cell[0]}, {cell[1]})\nt={ts-t0:.2f} s')
plotmap(axes[1,1],late,f'5. Misma ubicación: riesgo posterior\nt={rt-t0:.2f} s · TTC={risk.rear.time_to_collision_s:.2f} s')
ax=axes[1,2];ax.imshow(actual,vmin=0,vmax=100,cmap='viridis',aspect='auto')
for ij in np.ndindex(actual.shape):ax.text(ij[1],ij[0],f'{actual[ij]:.0f}',ha='center',va='center',fontsize=8,color='white' if actual[ij]<45 else 'black')
for row,col in np.argwhere(actual>front+2e-4):ax.add_patch(Rectangle((col-.5,row-.5),1,1,fill=False,edgecolor='#ff7900',lw=1.5))
ax.set(title=f'6. HapticGrid publicado\nsello frontal: {stamp(h)-t0:.2f} s',xlabel='Columna',ylabel='Fila')
fig.suptitle('De una medición de profundidad a la memoria y la advertencia posterior',fontsize=15,fontweight='bold')
fig.savefig(out/'trazabilidad_observacion.png',dpi=220)
summary=dict(bag=str(args.bag),source=str(args.source),t0=t0,observation_stamp=ts,risk_stamp=rt,haptic_stamp=stamp(h),
 pixel=[u,v],depth_m=float(depth[v,u]),optical_point=optical.tolist(),aligned_point=point.tolist(),world_point=wp.tolist(),
 map_cell=cell.tolist(),later_map_cell=late_cell.tolist(),target_world=target_world.tolist(),angular_cell=[ri,ci],cell_count=dc.count,cell_min=dc.min_m,
 ymax=ymax,reprojection_error_m=err,risk_intensity=risk.rear.intensity,ttc=risk.rear.time_to_collision_s,
 note='Same world cell at two timestamps, not point/object identity tracking; early point contributes but need not set the cell minimum')
(out/'trazabilidad_observacion.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
