"""Call the authorized real model; no scripted fallback selects recovery actions."""
import json,time
from pathlib import Path
from supervisor import PROMPT as PREVIOUS_PROMPT,public,validate
from openarm_recovery import agent as engine
from openarm_vla.common import atomic_json

PROMPT=PREVIOUS_PROMPT.replace('inspecting a completed plate-placement operation','taking over a stopped learned ACT plate-transport controller')+'''
A global_grid_pose estimate is additionally available from RGB-D only when visible wells span all12 columns and8 rows of the known9mm lattice. It works at source or destination and is preferable to the local near_seat_grid_pose. Null means insufficient view, not absent plate. The source support and plate may have shifted together by a few mm from nominal. If a hand occludes the plate, move the empty hand clear and observe biolab_top with depth again. For completion use a conservative estimated XY error <=2 mm (the independent task tolerance is3 mm); the extra 1 mm margin accounts for local image-estimation uncertainty. Reobserve if an estimate is close to the boundary. This experiment starts with a plate on the SOURCE support at [.43,+.08], support top z=1.027. ACT has run and has not verified task completion by its allotted deadline. The plate may still be at the source, in the hand, near destination, or on a locator. Inspect before assuming which failure occurred. The calibrated source grasp is pad [.43,+.08,1.058], pitch=-.6, aperture .43; lift to [.43,+.08,1.12], pitch=-1.1, then transfer to [.43,-.08,1.12], lower near [.427,-.08,1.058], pitch=-.6, and release only when supported. These are nominal calibrations supplied equally to the scripted baseline, not current object state. Use new observations to adjust or choose a different sequence. A plane fit at the destination center is an extrapolation when the plate is elsewhere: use the reported observed blue-pixel bounds and the images to distinguish source vs destination. The local grid-pose estimator only works NEAR the destination and cannot locate a plate at the source. If it returns null, use views and the known nominal source calibration, with fresh observations after approach/lift. Failed ACT can leave fingers closed while empty. Open before a new grasp only after checking that a plate will not be dropped. Do not assume that a motion return means grasp success. You control all recovery actions; no automatic grasp or placement correction follows your response.
'''

def recover(t,max_decisions=30,config=None):
    directory=t.directory;engine.API=PROMPT;config=config or engine.settings()
    atomic_json(directory/'agent-settings.json',config);(directory/'agent-prompt.txt').write_text(PROMPT)
    atomic_json(directory/'handoff-state.json',t.backend.snapshot());atomic_json(directory/'handoff.json',{'sim_time':float(t.backend.data.time),'reason':'ACT failed to verify placement within deadline','same_in_memory_backend':True,'reset_or_object_mutation':False})
    started=time.monotonic();history=[];observations={};usages=[];claimed=False
    response={'state':'done','results':[t.observe('camera_ceiling')]}
    for i in range(max_decisions):
        t.check()
        for o in response.get('results',[]):
            if 'id' in o:observations[o['id']]=o
        latest=list(observations.values())[-4:]
        packet={'task':'Complete transport of the empty plate from the teal source to stable seating on the orange destination. ACT has stopped. Choose extra observations and recovery actions.','decision_index':i,'decisions_remaining':max_decisions-i,'remaining_sim_s':t.max_seconds-(float(t.backend.data.time)-t.start),'latest_response':public(response),'recent_history':history[-8:],'known_observation_ids':list(observations),'attached_images':[{'id':o['id'],'camera':o['camera'],'modality':o['modality'],'sim_time':o['sim_time']} for o in latest]}
        atomic_json(directory/'agent-status.json',{'phase':'waiting_agent','decision':i,'sim_time':float(t.backend.data.time)})
        decision,usage=engine.decide(directory/'agent'/f'decision-{i:03d}',packet,[Path(o['rgb']) for o in latest],config,directory/'STOP',timeout=180);usages.append(usage)
        try:r=validate(decision,observations)
        except (ValueError,TypeError,KeyError) as e:
            response={'state':'validation_error','error':str(e),'results':[]};history.append({'decision':decision,'response':response});continue
        if r.get('finish'):
            claimed=True;history.append({'request':r,'response':{'state':'finished'}});break
        response=t.apply(r);history.append({'request':r,'response':public(response)})
        atomic_json(directory/'agent-history.json',history);atomic_json(directory/'agent-usage.json',usages)
    atomic_json(directory/'agent-history.json',history);atomic_json(directory/'agent-usage.json',usages)
    atomic_json(directory/'agent-end.json',{'claimed_finish':claimed,'calls':len(usages),'wall_s':time.monotonic()-started,'sim_time':float(t.backend.data.time),'total_tokens':sum(u['total_tokens'] for u in usages)})
    if not claimed:raise RuntimeError('maximum_agent_decisions')
