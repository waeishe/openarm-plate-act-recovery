"""The same scripted demonstration generator used for training."""
import json,time,traceback
import numpy as np
from environment import ROOT,Experiment
from openarm_vla.common import atomic_json
def demos(a):
    rng=np.random.default_rng(20261031)
    conditions=[{'plate_xy_m':[0.,0.],'timing':1.}]+[{'plate_xy_m':rng.uniform(-.0007,.0007,2).tolist(),'timing':float(rng.uniform(.94,1.06))} for _ in range(a.count-1)]
    atomic_json(ROOT/'configs/data-plan.json',{'seed':20261031,'conditions':conditions,'split':'last 4 episodes validation; no final evaluation states','expert':'model-based calibrated trajectory with known initialization offset; not human demonstrations'})
    for i,c in enumerate(conditions):
        dest=ROOT/'datasets'/f'demo-{i:03d}'
        if dest.exists():continue
        t=Experiment(dest,c,True);error=None;start=time.monotonic()
        try:t.teacher(adapt=True,timing=c['timing'])
        except Exception as e:error=str(e);(dest/'error.txt').write_text(traceback.format_exc())
        finally:
            s=t.finish(error);t.close();print(json.dumps({'demo':i,'completed':s['task_completed'],'quality':s['quality_success'],'hold':s['two_finger_clear_hold_s'],'release':s['supported_before_release_s'],'wall':time.monotonic()-start}),flush=True)
