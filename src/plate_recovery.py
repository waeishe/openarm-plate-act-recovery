import argparse,hashlib,json,math,time,traceback
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image
from plate_validation import PlateBackend,Trial
from openarm_vla.common import atomic_json,jsonl
from openarm_recovery.runtime import restore
from perception import estimate
from quality import score,CRITERIA

ROOT=Path(__file__).resolve().parents[1]
VALIDATION=ROOT
CAMERAS={'camera_ceiling','biolab_top','camera_head_left','camera_head_right','camera_wrist_left','camera_wrist_right'}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

class Recovery(Trial):
    def __init__(self,directory,source,budget=90):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=False);source=Path(source)
        variant=read(source/'scene-variant.json');manifest=read(source/'manifest.json');snapshot=read(source/'final-state.json')
        self.backend=PlateBackend(variant['layout_offset_x_m'])
        if self.backend.asset_hash!=manifest['resolved_xml_sha256']:raise ValueError('source_model_hash_changed')
        if sha(manifest['source_scene'])!=manifest['source_scene_sha256']:raise ValueError('source_scene_changed')
        proof=restore(self.backend,snapshot,self.backend.asset_hash)
        self.ik_data=mujoco.MjData(self.backend.model)
        self.frame=self.observations=0;self.start=float(self.backend.data.time);self.max_seconds=budget;self.deadline=time.monotonic()+1800
        self.arm_speed=1.;self.gripper_speed=2.;self.phase='handoff';self.history=[];self.release_times=[];self.grasp_count=0;self.finished=False
        self.handles={n:(self.directory/f'{n}.jsonl').open('w') for n in ['trajectory','actions','evaluation','events']}
        initial=self.measure();self.history.append(initial);jsonl(self.handles['evaluation'],initial);jsonl(self.handles['trajectory'],self.backend.sample(0))
        atomic_json(self.directory/'initial-state.json',self.backend.snapshot());atomic_json(self.directory/'initial-evaluation.json',initial)
        atomic_json(self.directory/'handoff.json',{**proof,'source':str(source),'source_snapshot_sha256':sha(source/'final-state.json'),'no_physics_step_before_first_observation':True})
        atomic_json(self.directory/'manifest.json',{'source':str(source),'scene':self.backend.scene,'layout_offset_x_m':variant['layout_offset_x_m'],'model_hash':self.backend.asset_hash,'budget_s':budget,'trace_hz':30,'agent_inputs':'RGB, selected RGB-D estimates, calibrated fixture map, arm state; no object truth or contacts','source_sha256':{p.name:sha(p) for p in Path(__file__).parent.glob('*.py')},'criteria':CRITERIA})
        for p in Path(__file__).parent.glob('*.py'):(self.directory/p.name).write_bytes(p.read_bytes())

    def public_state(self):
        return {'sim_time':float(self.backend.data.time),'elapsed_s':float(self.backend.data.time)-self.start,'coordinate_frame':'world','joints':self.backend.position().tolist(),'left':self.pose('left'),'right':self.pose('right')}

    def observe(self,camera='camera_ceiling',depth=False):
        if camera not in CAMERAS or not isinstance(depth,bool) or (depth and camera not in ['camera_ceiling','biolab_top']):raise ValueError('invalid_camera_or_depth')
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

    def perform(self,a):
        a=dict(a);kind=a.pop('kind');self.phase=kind
        if kind=='observe':return self.observe(**a)
        if kind=='move':
            self.move(**a)
            error=float(np.linalg.norm(np.asarray(self.pose(a['arm'])['position'])-a['position']))
            if error>.008:raise RuntimeError(f'target_not_reached:{error:.5f}m')
        elif kind=='aperture':
            arm=a['arm'];opening=a['opening_rad'];idx=15 if arm=='left' else 7;old=abs(self.backend.target[idx]);seconds=a.get('seconds',1.)
            if opening<.6 and old>=.6:self.grasp_count+=1
            if opening>=.7 and old<.6:self.release_times.append(float(self.backend.data.time))
            self.aperture(arm,opening,seconds)
        elif kind=='wait':self.wait(**a)
        else:raise ValueError('invalid_kind')
        return {'kind':kind,'motion_completed':True,**self.public_state()}

    def apply(self,request):
        results=[];error=None
        for a in request.get('actions',[]):
            try:results.append(self.perform(a))
            except Exception as e:
                error=f'{type(e).__name__}: {e}';self.backend.hold();self.event('action_error',error=error,action=a);break
        self.event('agent_request',request=request,error=error)
        atomic_json(self.directory/'checkpoint.json',{'snapshot':self.backend.snapshot(),'target':self.backend.target.tolist(),'target_time':self.backend.target_time})
        return {'state':'error' if error else 'done','error':error,'results':results,'actual':self.public_state()}

    def finish(self,error=None):
        self.finished=True;self.capture('final');atomic_json(self.directory/'final-state.json',self.backend.snapshot())
        result=score(self.history,self.release_times,self.grasp_count,error);atomic_json(self.directory/'summary.json',result);return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['probe','auto']);p.add_argument('--run',required=True);p.add_argument('--source',default=str(VALIDATION/'runs/nominal-07'));p.add_argument('--max-decisions',type=int,default=20);a=p.parse_args()
    if not a.run.replace('-','').replace('_','').isalnum():raise ValueError('invalid_run')
    if a.command=='auto':
        from supervisor import run
        result=run(a.run,Path(a.source),a.max_decisions)
    else:
        t=Recovery(ROOT/'runs'/a.run,Path(a.source))
        try:
            results=[t.observe('camera_ceiling'),t.observe('camera_ceiling',True),t.observe('camera_head_left'),t.observe('biolab_top',True)]
            result={'probe_only':True,'observations':results};atomic_json(t.directory/'probe.json',result)
        finally:t.finish();t.close()
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
