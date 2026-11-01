import json,time,traceback,hashlib
import numpy as np
from environment import ROOT,Experiment,sha
from monitor import Monitor
from openarm_vla.common import atomic_json,jsonl

def run(a):
    t=Experiment(ROOT/'runs'/a.run,{'plate_xy_m':[a.x_mm/1000,a.y_mm/1000],'source_xy_m':[a.source_x_mm/1000,a.source_y_mm/1000]});t.max_seconds=a.horizon;started=time.monotonic();error=None;monitor=Monitor();completed=False;handoff=False
    t.handles['timing']=(t.directory/'timing.jsonl').open('w');old_step=t.step
    def step(target,dt=1/30):
        old_step(target,dt);jsonl(t.handles['timing'],{'sim_time':float(t.backend.data.time),'wall_s':time.monotonic()-started})
    t.step=step
    atomic_json(t.directory/'run-config.json',vars(a))
    try:
        if a.method=='fixed':t.teacher();completed=True
        else:
            from learning import Policy
            policy=Policy(a.checkpoint);atomic_json(t.directory/'checkpoint.json',{'path':a.checkpoint,'sha256':sha(a.checkpoint)})
            t.phase='ACT';next_monitor=26
            with (t.directory/'policy-inputs.jsonl').open('w') as f:
                while t.backend.data.time<a.horizon-1e-6:
                    obs=t.inputs();actions=policy.predict(obs)
                    jsonl(f,{'sim_time':float(t.backend.data.time),'state':obs['state'].tolist(),'image_hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in obs['images'].items()},'predicted_actions':actions.tolist()})
                    for target in actions[:a.execute]:
                        for _ in range(3):t.step(target)
                        if t.backend.data.time>=a.horizon-1e-6:break
                    if t.backend.data.time>=next_monitor-1e-6:
                        completed=monitor.update(t);next_monitor+=2
                        atomic_json(t.directory/'monitor.json',monitor.samples)
                        if completed:break
                    if a.method=='agent' and t.backend.data.time>=a.handoff-1e-6:
                        handoff=True;break
            atomic_json(t.directory/'act-end-state.json',t.backend.snapshot())
            atomic_json(t.directory/'act-end.json',{'sim_time':float(t.backend.data.time),'completed_by_monitor':completed,'handoff':handoff,'policy_calls':policy.calls,'wall_s':time.monotonic()-started})
            if handoff:
                from agent_control import recover
                t.phase='Agent';recover(t,config=getattr(a,'agent_config',None));completed=True
    except Exception as e:error=f'{type(e).__name__}: {e}';(t.directory/'error.txt').write_text(traceback.format_exc())
    finally:
        atomic_json(t.directory/'controller-end.json',{'sim_time':float(t.backend.data.time),'wall_s':time.monotonic()-started,'completed':completed,'error':error})
        # Every method is observed to the common horizon without outcome-dependent padding.
        try:
            t.phase='hold'
            while t.frame<round(a.horizon*30):t.step(t.backend.target.copy())
        except Exception as e:error=error or str(e)
        s=t.finish(error);s.update(wall_s=time.monotonic()-started,controller_completed=completed,agent_called=handoff);atomic_json(t.directory/'summary.json',s);t.close();print(json.dumps({k:v for k,v in s.items() if k!='final'}),flush=True)
