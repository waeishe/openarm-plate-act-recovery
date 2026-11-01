import json,math,time
from pathlib import Path
import numpy as np
from openarm_recovery import agent as engine
from openarm_vla.common import atomic_json
from plate_recovery import ROOT,Recovery

PROMPT='''You control a SIMULATED OpenArm robot inspecting a completed plate-placement operation. Verify whether the empty 96-well plate is correctly seated on the orange destination support. If needed, choose extra observations and physical recovery actions; otherwise leave it alone. A stopped operation is not proof of failure. Use only supplied observations and history. No tools or file access. Return a concise reason and request_json, a JSON string containing based_on:[observation IDs] and either actions:[...] or finish:true.
Each action uses "kind". At most 4 actions per decision. Actions stop on error. API:
 observe: camera one of camera_ceiling, biolab_top, camera_head_left, camera_head_right, camera_wrist_left, camera_wrist_right; depth boolean optional. Depth is only available for camera_ceiling/biolab_top. biolab_top looks directly at the work area; ceiling can be occluded by the hand. RGB-D estimates the plane of blue well floors and, only near the seat, a locally calibrated well-grid pose. Null means insufficient visual evidence, not a flat plate. RGB alone may hide small tilt. No object-state or contact oracle is supplied.
 move: arm left/right, position [x,y,z] world meters, seconds .1..8, pitch and yaw radians. Orientation is Rz(yaw)Ry(pitch). Position is approximate pad center, not plate center. Reobserve after motion or tracking error; never repeat a blocked descent blindly.
 aperture: arm left/right, opening_rad .35.. .7854, seconds .1..3. This is a finger ANGLE in radians: .7854 is open; .43 has physically held this empty plate in previous nominal calibration. This is not a grasp-success guarantee.
 wait: seconds .1..5.
Known fixture calibration (not object truth): destination center [.43,-.08], support top z=1.027 m, level desired plate bottom z=1.027 and yaw=0. Empty plate dimensions [.12776,.08548,.01435] m; blue well floors are .00356 m above plate bottom. Positive plane slope x means +x edge is higher. Four small corner locators can support a misplaced plate at an angle even when it appears inside the target. Desired placement: centered within 3 mm, tilt <=1 degree, yaw <=3 degrees, resting on central support, grippers open and clear, stable >=2 seconds. Depth images are ideal simulated sensors, no contact sensing.
Nominal low-level calibration: left arm can approach this seat at pad [.43,-.08,1.12], pitch=-1.1,yaw=0. Near-seat side grasp uses pitch=-.6,yaw=0 and pad z roughly observed well-floor height +.027 m, with open fingers on approach. Source nominal well-floor z=1.03056. At greater lift heights the wrist must tilt more (pitch around -1.1 at pad z=1.12); keeping pitch=-.6 that high may be unreachable. This wrist rotation tilts an empty plate; compensate its observed offset and orientation when lowering. The right arm begins parked and need not move unless its view is useful. The hand and plate can shift slightly when closing and lifting. Choose current x/y from observations, do not assume pad center equals plate center. A small lift can clear locators; correct alignment while clear, then lower, release, retreat and reobserve. Do not slide/push against a locator. Do not release from height. Observe after lift, after release, and after retreat/settling. Before irreversible finish, verify the result using current observations. Do not finish just because motion completed. If already properly seated, avoid an unnecessary grasp, wait and reobserve.
Mechanical execution envelope: when left pad z is below 1.080 m in this task, use the calibrated near-seat wrist pitch=-.600, yaw=0, and pad z >=1.058 m. Other low wrist angles can cause a finger to hit the support base even when its nominal pad point is high enough. The runtime rejects commands outside this envelope rather than silently correcting them. Correct x/y while lifted; do not directly subtract a measured plate tilt from the low wrist pitch. A supported placement at the calibrated height and a fresh post-release observation determine whether reseating worked. You choose whether, where and when to regrasp, reposition, release, reobserve or finish; the guard does not choose a recovery trajectory.
You select observations, coordinates, wrist poses within that envelope, grip timing, correction and termination. There is no automatic reseat skill. The simulator is frozen while you reason. The evaluator's true state is not available until finish. Reasons should identify the observation and why the next step is appropriate.'''

def public(x):
    if isinstance(x,dict):return {k:public(v) for k,v in x.items() if k not in {'rgb','summary','evaluation','calibration','source','depth'}}
    if isinstance(x,list):return [public(v) for v in x]
    return x

def validate(value,known):
    if set(value)!={'reason','request_json'} or not isinstance(value['reason'],str) or not value['reason'].strip():raise ValueError('invalid_envelope')
    r=json.loads(value['request_json']);refs=r.get('based_on',[]);actions=r.get('actions',[]);finish=r.get('finish',False)
    if set(r)-{'based_on','actions','finish'} or not isinstance(refs,list) or not refs or any(x not in known for x in refs):raise ValueError('invalid_observation_reference')
    if not isinstance(finish,bool) or not isinstance(actions,list) or (finish and actions) or (not finish and not 1<=len(actions)<=4):raise ValueError('invalid_action_count')
    allowed={'observe':{'kind','camera','depth'},'move':{'kind','arm','position','seconds','pitch','yaw'},'aperture':{'kind','arm','opening_rad','seconds'},'wait':{'kind','seconds'}}
    def num(v,lo,hi):return isinstance(v,(float,int)) and not isinstance(v,bool) and math.isfinite(v) and lo<=v<=hi
    for a in actions:
        if not isinstance(a,dict) or a.get('kind') not in allowed or set(a)-allowed[a['kind']]:raise ValueError('unsupported_action')
        k=a['kind']
        if k=='observe':
            if a.get('camera','camera_ceiling') not in {'camera_ceiling','biolab_top','camera_head_left','camera_head_right','camera_wrist_left','camera_wrist_right'} or not isinstance(a.get('depth',False),bool):raise ValueError('invalid_camera')
        if k in {'move','aperture'} and a.get('arm') not in {'left','right'}:raise ValueError('invalid_arm')
        if k=='move':
            a.setdefault('seconds',2);a.setdefault('pitch',-.6);a.setdefault('yaw',0)
            p=a.get('position');
            if not isinstance(p,list) or len(p)!=3 or not all(num(v,lo,hi) for v,lo,hi in zip(p,[.25,-.4,1.015],[.6,.4,1.25])):raise ValueError('invalid_workspace')
            if not num(a.get('seconds',2),.1,8) or not num(a.get('pitch',-.6),-1.7,0) or not num(a.get('yaw',0),-1.7,1.7):raise ValueError('invalid_pose_or_time')
            if a['arm']=='left' and p[2]<1.08 and (p[2]<1.058 or abs(a['pitch']+.6)>.005 or abs(a['yaw'])>.03):raise ValueError('outside_calibrated_near_seat_envelope: use pitch=-.6, yaw=0, pad_z>=1.058; reobserve and choose a revised action')
        if k=='aperture' and (not num(a.get('opening_rad'),.35,.7854) or not num(a.get('seconds',1),.1,3)):raise ValueError('invalid_aperture')
        if k=='wait' and not num(a.get('seconds'),.1,5):raise ValueError('invalid_wait')
    return {**r,'reason':value['reason']}
