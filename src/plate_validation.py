"""Nominal-pose plate manipulation. All outputs are external; dependencies read only."""
import argparse,hashlib,json,math,time,traceback,xml.etree.ElementTree as ET
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from openarm_vla.backend import NativeBackend
from openarm_vla.common import JOINT_NAMES,atomic_json,jsonl
from aspire_lab.openarm_adapter import OpenArmAdapter,rotation
from scoring import evaluate,CRITERIA

ROOT=Path(__file__).resolve().parents[1]
HOST=ROOT/'assets'
PACK=ROOT/'assets'
SCENE=PACK/'plate_transfer.xml'

def model_xml(layout_offset_x=0.):
    root=ET.fromstring(SCENE.read_text())
    root.find('asset/model').set('file',str(HOST/'openarm_mujoco/v2/cell/cell.xml'))
    for name in ['plate','source_seat','destination_seat']:
        body=root.find(f"worldbody/body[@name='{name}']");pos=list(map(float,body.get('pos').split()));pos[0]+=layout_offset_x;body.set('pos',' '.join(map(str,pos)))
    return ET.tostring(root,encoding='unicode')

class PlateBackend(NativeBackend):
    def __init__(self,layout_offset_x=0.):
        self.scene='v2/external/biolab/cell/plate_transfer.xml'
        self.model=m=mujoco.MjModel.from_xml_string(model_xml(layout_offset_x))
        self.data=d=mujoco.MjData(m);self.renderers={}
        ids=[m.joint(n).id for n in JOINT_NAMES]
        self.joint_ids=np.array(ids);self.qadr=m.jnt_qposadr[ids];self.dadr=m.jnt_dofadr[ids]
        acts=[]
        for j in ids:
            a=np.where(m.actuator_trnid[:,0]==j)[0]
            if len(a)!=1:raise ValueError('ambiguous actuator')
            acts.append(a[0])
        self.act_ids=np.array(acts);self.limits=m.actuator_ctrlrange[self.act_ids].copy()
        for i in range(16):
            if i not in (7,15):self.limits[i]=[max(self.limits[i,0],m.jnt_range[ids[i],0]),min(self.limits[i,1],m.jnt_range[ids[i],1])]
        self.bias_dofs=[int(self.dadr[i]) for i in range(16) if i not in (7,15)]
        lifter=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_JOINT,'openarm_lifter_joint')
        if lifter>=0:self.bias_dofs.append(int(m.jnt_dofadr[lifter]))
        key=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,'home')
        if key>=0:mujoco.mj_resetDataKeyframe(m,d,key)
        for i in range(m.nu):
            j=m.actuator_trnid[i,0]
            if j>=0:d.ctrl[i]=d.qpos[m.jnt_qposadr[j]]
        self.target=d.ctrl[self.act_ids].copy();self.target_time=float(d.time)
        m.vis.global_.offwidth=1280;m.vis.global_.offheight=960
        self.option=mujoco.MjvOption();self.option.geomgroup[3:]=0
        mujoco.mj_forward(m,d)
        self.asset_hash=hashlib.sha256(model_xml(layout_offset_x).encode()).hexdigest()

class Trial(OpenArmAdapter):
    def __init__(self,directory,layout_offset_x=0.):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        self.backend=PlateBackend(layout_offset_x);self.ik_data=mujoco.MjData(self.backend.model)
        self.frame=self.observations=0;self.start=0.;self.max_seconds=120;self.deadline=time.monotonic()+600
        self.arm_speed=1.;self.gripper_speed=2.;self.phase='settle';self.history=[]
        self.handles={n:(self.directory/f'{n}.jsonl').open('w') for n in ['trajectory','actions','evaluation','events']}
        for source in Path(__file__).parent.glob('*.py'):(self.directory/source.name).write_bytes(source.read_bytes())
        atomic_json(self.directory/'criteria.json',CRITERIA)
        atomic_json(self.directory/'scene-variant.json',{'layout_offset_x_m':layout_offset_x,'translated_bodies':['plate','source_seat','destination_seat'],'geometry_and_physics_unchanged':True})
        atomic_json(self.directory/'initial-state.json',self.backend.snapshot())
        atomic_json(self.directory/'manifest.json',{'scene':self.backend.scene,'source_scene':str(SCENE),'source_scene_sha256':hashlib.sha256(SCENE.read_bytes()).hexdigest(),'resolved_xml_sha256':self.backend.asset_hash,'mujoco_version':mujoco.__version__,'controller':'nominal model-based skill; no ACT or agent','arm_speed_rad_s':1,'finger_speed_rad_s':2,'trace_hz':30,'free_plate':True,'object_reset_during_trial':False})
        self.wait(.5)

    def measure(self):
        b=self.backend;m,d=b.model,b.data;p=m.body('plate').id
        pos=d.xpos[p].copy();r=d.xmat[p].reshape(3,3);target=d.site_xpos[m.site('destination_seat_target').id]
        contacts=[];environment=[]
        for ci,c in enumerate(d.contact):
            g1,g2=int(c.geom1),int(c.geom2)
            bodies=[m.body(int(m.geom_bodyid[g])).name for g in (g1,g2)]
            if c.dist<=0 and any(n.startswith('openarm_') for n in bodies) and any(n in ['world','source_seat','destination_seat'] for n in bodies):
                force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                if force[0]>.1:environment.append({'geoms':[m.geom(g1).name,m.geom(g2).name],'normal_force_n':float(force[0])})
            if p in (m.geom_bodyid[g1],m.geom_bodyid[g2]) and c.dist<=.001:
                other=g2 if m.geom_bodyid[g1]==p else g1
                force=np.zeros(6);mujoco.mj_contactForce(m,d,ci,force)
                contacts.append({'geom':m.geom(other).name,'body':m.body(int(m.geom_bodyid[other])).name,'distance':float(c.dist),'normal_force_n':float(force[0])})
        support=any(c['geom']=='destination_seat_support' and c['normal_force_n']>.01 for c in contacts)
        fingers=sorted(set(c['body'] for c in contacts if 'finger' in c['body'] and c['normal_force_n']>.005))
        adr=m.jnt_dofadr[m.body_jntadr[p]]
        v=d.qvel[adr:adr+6]
        corners=np.array([[x,y,z] for x in [-.06388,.06388] for y in [-.04274,.04274] for z in [0,.01435]])@r.T+pos
        hands={arm:self.pose(arm) for arm in ['left','right']}
        return {'sim_time':float(d.time),'phase':self.phase,'plate_xyz':pos.tolist(),'plate_quat':d.xquat[p].tolist(),'min_plate_z':float(corners[:,2].min()),'left_finger_angle':float(abs(b.position()[15])),'right_finger_angle':float(abs(b.position()[7])),'hand_distance_m':min(float(np.linalg.norm(np.array(s['position'])-pos)) for s in hands.values()),'xy_error_m':float(np.linalg.norm(pos[:2]-target[:2])),'z_error_m':float(pos[2]-target[2]),'tilt_deg':math.degrees(math.acos(np.clip(r[2,2],-1,1))),'yaw_deg':math.degrees(math.atan2(r[1,0],r[0,0])),'linear_speed':float(np.linalg.norm(v[:3])),'angular_speed':float(np.linalg.norm(v[3:])),'destination_supported':support,'finger_bodies':fingers,'contacts':contacts,'robot_fixture_contacts':environment,**hands}

    def step(self,target,dt=1/30):
        self.check();b=self.backend;applied=b.step(target,dt,self.arm_speed,self.gripper_speed);self.frame+=1
        jsonl(self.handles['trajectory'],b.sample(self.frame));jsonl(self.handles['actions'],{'frame':self.frame,'requested':np.asarray(target).tolist(),'applied':applied.tolist()})
        state=self.measure();self.history.append(state);jsonl(self.handles['evaluation'],state)

    def capture(self,label):
        b=self.backend
        camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.azimuth=225;camera.elevation=-38;camera.lookat[:]=[.4,0,1];camera.distance=1.45
        for name,cam in [('overview',camera),('ceiling','biolab_top'),('head','camera_head_left')]:
            if name=='overview':
                r=b.renderers.setdefault('overview',mujoco.Renderer(b.model,height=480,width=640)) if 'overview' not in b.renderers else b.renderers['overview']
                r.update_scene(b.data,camera=cam,scene_option=b.option)
                cell=b.model.geom('cell_vis').id
                for g in r.scene.geoms[:r.scene.ngeom]:
                    if g.objtype==mujoco.mjtObj.mjOBJ_GEOM and g.objid==cell:g.rgba[3]=.1
                rgb=r.render().copy()
            else:rgb=b.capture({'im':{'camera':cam,'height':480,'width':640}})['im']
            Image.fromarray(rgb).save(self.directory/f'{label}-{name}.png')
        atomic_json(self.directory/('final-measurement.json' if label=='final' else f'{label}-state.json'),self.measure())

    def aperture(self,arm,opening,seconds=.8):
        target=self.backend.target.copy();target[15 if arm=='left' else 7]=opening*(1 if arm=='left' else -1)
        self.event('aperture',arm=arm,angle_rad=opening)
        for _ in range(math.ceil(seconds*30)):self.step(target)

    def execute(self,config):
        atomic_json(self.directory/'config.json',config);self.capture('00-initial')
        for i,a in enumerate(config['actions']):
            a=dict(a);self.phase=a.pop('label',f'action-{i}');kind=a.pop('kind')
            print(json.dumps({'phase':self.phase,'time':float(self.backend.data.time)}),flush=True)
            if kind=='move':
                self.move(**a)
                error=float(np.linalg.norm(np.asarray(self.pose(a['arm'])['position'])-a['position']))
                if error>.008:raise RuntimeError(f'physical_target_not_reached:{error:.4f}m')
            elif kind=='aperture':self.aperture(**a)
            elif kind=='wait':self.wait(**a)
            else:raise ValueError(kind)
            self.capture(f'{i+1:02d}-{self.phase}')

    def finish(self,error=None):
        self.capture('final');atomic_json(self.directory/'final-state.json',self.backend.snapshot())
        result=evaluate(self.history,error)
        atomic_json(self.directory/'summary.json',result);print(json.dumps(result,ensure_ascii=False),flush=True);return result

    def close(self):
        for f in self.handles.values():f.close()
        self.backend.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['inspect','run']);p.add_argument('--run',required=True);p.add_argument('--config');a=p.parse_args()
    if not a.run.replace('-','').replace('_','').isalnum():raise ValueError('invalid run')
    config=json.loads(Path(a.config).read_text()) if a.config else None
    t=Trial(ROOT/'runs'/a.run,config.get('layout_offset_x',0.) if config else 0.);error=None
    try:
        if a.command=='inspect':
            t.capture('initial');b=t.backend;m=b.model;d=b.data
            info={'left':t.pose('left'),'right':t.pose('right'),'fingers':[],'plate':t.measure()}
            for i in range(m.ngeom):
                body=m.body(int(m.geom_bodyid[i])).name
                if 'finger' in body:info['fingers'].append({'body':body,'geom':m.geom(i).name,'type':int(m.geom_type[i]),'size':m.geom_size[i].tolist(),'pos':d.geom_xpos[i].tolist(),'friction':m.geom_friction[i].tolist()})
            atomic_json(t.directory/'geometry.json',info);print(json.dumps(info),flush=True)
        else:t.execute(config)
    except Exception as e:
        error=f'{type(e).__name__}: {e}';(t.directory/'error.txt').write_text(traceback.format_exc());print(error,flush=True)
    finally:
        result=t.finish(error);t.close()
    if a.command=='run' and result['outcome']!='success':raise SystemExit(1)

if __name__=='__main__':main()
