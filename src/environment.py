"""External native simulator. Observation and evaluation data are separate."""
import hashlib,json,math,time
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from perception_study import estimate
from plate_validation import PlateBackend,Trial
from plate_recovery import Recovery
from openarm_vla.common import atomic_json,jsonl
from quality import seated

ROOT=Path(__file__).resolve().parents[1]
CAMERAS={'top':{'camera':'biolab_top','height':128,'width':160},'head':{'camera':'camera_head_left','height':128,'width':160}}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def opening_events(actions):
    # Hysteresis: closed below .60; release after +.05 cumulative opening;
    # rearm only after fully open (> .65) and a subsequent closure.
    armed=False;can_arm=True;low=None;onset=None;releases=[]
    for a in actions:
        angle=a['applied'][15];t=(a['frame']-1)/30
        if not armed:
            if angle>.65:can_arm=True
            if can_arm and angle<.60:armed=True;low=angle;onset=None
        else:
            if angle<=low+.001:
                low=min(low,angle);onset=None
            elif onset is None:onset=t
            if angle-low>=.05:
                releases.append(onset);armed=False;can_arm=angle>.65
    return releases

def evaluate(history,release_times,error=None):
    # Method-independent final seating and whole-task quality; never policy inputs.
    end=history[-1]['sim_time'];tail=[x for x in history if x['sim_time']>=end-2-1e-8]
    stable=len(tail)>1 and tail[-1]['sim_time']-tail[0]['sim_time']>=2-1e-6 and all(seated(x) for x in tail)
    held=maxheld=0;streak=0;seat_time=None;support=[];acquired=False;missing=0;losses=[]
    for x in history:
        good=len(x['finger_bodies'])>=2 and x['min_plate_z']>1.037 and not x['destination_supported']
        held=held+1/30 if good else 0;maxheld=max(maxheld,held)
        if maxheld>=.3:acquired=True
        missing=missing+1 if acquired and not x['finger_bodies'] and not x['destination_supported'] else 0
        if missing==3:losses.append(x['sim_time'])
        streak=streak+1/30 if seated(x) else 0
        if streak>=2 and seat_time is None:seat_time=x['sim_time']-history[0]['sim_time']
    for t in release_times:
        before=[x for x in history if x['sim_time']<=t+1e-8];duration=0
        for x in reversed(before):
            if not x['destination_supported']:break
            duration+=1/30
        support.append(duration)
    collision=max((c['normal_force_n'] for x in history for c in x['robot_fixture_contacts']),default=0)
    # Retain completion even if the controller exhausted its decision budget.
    quality=bool(stable and maxheld>=.3 and support and all(v>=.2 for v in support) and not losses and collision<=.5)
    suffix_start=None
    if stable:
        for state in reversed(history):
            if not seated(state):break
            suffix_start=state['sim_time']
    terminal_completion=None if suffix_start is None else suffix_start-history[0]['sim_time']+2
    x=history[-1]
    return {'task_completed':bool(stable),'quality_success':quality,'error':error,'elapsed_sim_s':end-history[0]['sim_time'],'first_stable_completion_s':seat_time,'terminal_stable_completion_s':terminal_completion,'unsupported_loss_times':losses,'completion_without_verified_lift':bool(stable and maxheld<.3),'completion_after_unsupported_release_or_loss':bool(stable and (losses or any(v<.2 for v in support))),'two_finger_clear_hold_s':maxheld,'release_times':release_times,'supported_before_release_s':support,'all_releases_supported':bool(support and all(v>=.2 for v in support)),'max_fixture_contact_n':collision,'final_xy_error_mm':1000*x['xy_error_m'],'final_tilt_deg':x['tilt_deg'],'final_yaw_deg':x['yaw_deg'],'max_tilt_deg':max(x['tilt_deg'] for x in history),'final':x}

class Experiment(Recovery):
    def __init__(self,directory,condition=None,collect=False):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False)
        source_dir=self.directory/'source';source_dir.mkdir()
        for source in (ROOT/'src').glob('*.py'):(source_dir/source.name).write_bytes(source.read_bytes())
        self.backend=PlateBackend(-.15);b=self.backend;m,d=b.model,b.data
        self.condition=condition or {};offset=np.array(self.condition.get('plate_xy_m',[0.,0.]))
        source_offset=np.array(self.condition.get('source_xy_m',[0.,0.]))
        m.body_pos[m.body('source_seat').id,:2]+=source_offset
        if np.any(source_offset):b.asset_hash=hashlib.sha256((b.asset_hash+json.dumps({'source_xy_m':source_offset.tolist()},sort_keys=True)).encode()).hexdigest()
        q=m.jnt_qposadr[m.joint('plate_free').id];d.qpos[q:q+2]+=offset+source_offset
        yaw=self.condition.get('plate_yaw_rad',0.);d.qpos[q+3:q+7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
        mujoco.mj_forward(m,d)
        self.ik_data=mujoco.MjData(m);self.frame=self.observations=0;self.start=0.;self.max_seconds=120;self.deadline=time.monotonic()+7200
        self.arm_speed=1.;self.gripper_speed=2.;self.phase='initial';self.history=[];self.release_times=[];self.grasp_count=0;self.finished=False
        self.collect=collect;self.samples=[];self.pending=None;self.closed=False
        self.handles={n:(self.directory/f'{n}.jsonl').open('w') for n in ['trajectory','actions','evaluation','events']}
        atomic_json(self.directory/'initial-state.json',b.snapshot());atomic_json(self.directory/'condition.json',self.condition)
        atomic_json(self.directory/'manifest.json',{'model_hash':b.asset_hash,'source_scene':'assets/plate_transfer.xml','controller_inputs':'two RGB images and 16 measured joint positions','layout_offset_x':-.15,'source_fixture_offset_m':source_offset.tolist(),'initialization_only':True,'control_hz':10,'physics_dt':.001,'arm_speed':1,'gripper_speed':2,'sources':{str(p):sha(p) for p in (ROOT/'src').glob('*.py')}})
        x=self.measure();self.history.append(x);jsonl(self.handles['evaluation'],x);jsonl(self.handles['trajectory'],b.sample(0))

    def inputs(self):return {'state':self.backend.position().astype(np.float32),'images':self.backend.capture(CAMERAS)}

    def step(self,target,dt=1/30):
        if self.collect and self.frame%3==0:self.pending=self.inputs()
        # A release is the start of an opening command after closure, not the later open-angle crossing.
        old=float(self.backend.target[15]);delta=float(target[15])-old
        if old<.60:self.closed=True
        if self.closed and delta>.001:
            self.release_times.append(float(self.backend.data.time));self.closed=False
        super(Recovery,self).step(target,dt)
        if self.collect and self.frame%3==0:
            self.samples.append((self.pending,self.backend.target.astype(np.float32).copy(),self.backend.data.time-.1))

    def observe(self,camera='camera_ceiling',depth=False):
        if camera not in {'camera_ceiling','biolab_top','camera_head_left','camera_head_right','camera_wrist_left','camera_wrist_right'} or not isinstance(depth,bool) or (depth and camera not in ['camera_ceiling','biolab_top']):raise ValueError('invalid_camera_or_depth')
        name=f'obs-{self.observations:04d}';dest=self.directory/'observations'/name;dest.mkdir(parents=True)
        h,w=(720,960) if depth else (480,640);b=self.backend
        key=(h,w)
        if key not in b.renderers:b.renderers[key]=mujoco.Renderer(b.model,height=h,width=w)
        r=b.renderers[key];r.disable_depth_rendering();r.update_scene(b.data,camera=camera,scene_option=b.option);rgb=r.render().copy()
        Image.fromarray(rgb).save(dest/'rgb.png');obs={'id':name,'camera':camera,'modality':'RGB-D' if depth else 'RGB','rgb':str(dest/'rgb.png'),**self.public_state()}
        if depth:
            r.enable_depth_rendering();d=r.render().copy();r.disable_depth_rendering();np.save(dest/'depth.npy',d,allow_pickle=False)
            cam=b.model.camera(camera).id;cal={'focal_px':h/2/math.tan(math.radians(b.model.cam_fovy[cam])/2),'rotation':b.data.cam_xmat[cam].tolist(),'position':b.data.cam_xpos[cam].tolist()}
            found,mask=estimate(rgb,d,cal);obs['perception']=found
            atomic_json(dest/'calibration.json',cal);Image.fromarray((mask*255).astype(np.uint8)).save(dest/'segmentation.png')
        atomic_json(dest/'observation.json',obs);self.observations+=1;self.event('observation',id=name,camera=camera,depth=depth)
        return obs

    def teacher(self,adapt=False,timing=1.):
        config=json.loads((ROOT/'configs/calibration.json').read_text())
        self.wait(.5)
        for a in config['actions']:
            a=dict(a);self.phase=a.pop('label');kind=a.pop('kind')
            if adapt and self.phase in ['approach','descend','lift']:
                a['position'][:2]=list(np.array(a['position'][:2])+self.condition.get('plate_xy_m',[0,0]))
            if kind!='wait':a['seconds']=round(a.get('seconds',1)*timing,2)
            if kind=='move':self.move(**a)
            elif kind=='aperture':self.aperture(**a)
            else:self.wait(**a)

    def finish(self,error=None):
        self.finished=True
        # Hysteretic opening events: require a total command increase of 0.05 rad.
        # This prevents tiny learned command noise from counting as a release.
        for f in self.handles.values():f.flush()
        actions=[json.loads(s) for s in (self.directory/'actions.jsonl').read_text().splitlines()]
        releases=opening_events(actions)
        self.release_times=releases
        result=evaluate(self.history,releases,error);atomic_json(self.directory/'summary.json',result)
        atomic_json(self.directory/'final-state.json',self.backend.snapshot());self.capture('final')
        if self.collect and self.samples:
            np.savez_compressed(self.directory/'demonstration.npz',state=np.stack([s[0]['state'] for s in self.samples]),action=np.stack([s[1] for s in self.samples]),time=np.array([s[2] for s in self.samples]),**{k:np.stack([s[0]['images'][k] for s in self.samples]) for k in CAMERAS})
        return result
