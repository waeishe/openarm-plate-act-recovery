"""Offline-capable scoring of a physical plate transport; never selects actions."""
import math

CRITERIA={'version':2,'xy_error_max_m':.003,'z_error_max_m':.002,'tilt_max_deg':1,'yaw_max_deg':3,'linear_speed_max_m_s':.005,'angular_speed_max_rad_s':.03,'final_stability_s':2,'fully_lifted_clearance_m':.010,'two_finger_hold_s':.5,'supported_before_open_s':.2,'unsupported_contact_loss_frames':3,'final_locator_load_max_n':.02,'robot_fixture_contact_max_n':.5,'trace_hz':30}

def evaluate(history,error=None):
    last=history[-1];dt=1/30
    def locator_load(x):return sum(c['normal_force_n'] for c in x['contacts'] if 'destination_seat_locator' in c['geom'])
    def seated(x):
        return x['xy_error_m']<=.003 and abs(x['z_error_m'])<=.002 and x['tilt_deg']<=1 and abs(x['yaw_deg'])<=3 and x['linear_speed']<.005 and x['angular_speed']<.03 and x['destination_supported'] and not x['finger_bodies'] and x['left_finger_angle']>.65 and x['right_finger_angle']>.65 and x['hand_distance_m']>.05 and locator_load(x)<=.02
    tail=[x for x in history if x['sim_time']>=last['sim_time']-2-1e-8]
    stable=tail[-1]['sim_time']-tail[0]['sim_time']>=2-1e-6 and all(seated(x) for x in tail)
    held=maxheld=0.;acquired=False;loss=0;unsupported=[]
    for x in history:
        good=len(x['finger_bodies'])>=2 and x['min_plate_z']>1.027+.010 and not x['destination_supported']
        held=held+dt if good else 0.;maxheld=max(maxheld,held)
        if held>=.5:acquired=True
        missing=acquired and not x['finger_bodies'] and not x['destination_supported'] and x['phase'] not in ['release','retreat','stable']
        loss=loss+1 if missing else 0
        if loss==3:unsupported.append(x['sim_time'])
    release=next((i for i,x in enumerate(history) if x['phase']=='release'),None)
    support_duration=0.
    if release is not None:
        for x in reversed(history[:release]):
            if not x['destination_supported']:break
            support_duration+=dt
    release_supported=support_duration>=.2
    fixture_max=max((c['normal_force_n'] for x in history for c in x['robot_fixture_contacts']),default=0.)
    success=stable and maxheld>=.5 and release_supported and not unsupported and fixture_max<=.5 and error is None
    return {'outcome':'success' if success else 'failure','error':error,'final_stable_2s':stable,'two_finger_free_hold_s':maxheld,'supported_before_release_s':support_duration,'unsupported_contact_loss_times':unsupported,'intentional_supported_release':release_supported,'xy_error_mm':last['xy_error_m']*1000,'tilt_deg':last['tilt_deg'],'yaw_deg':last['yaw_deg'],'maximum_tilt_deg':max(x['tilt_deg'] for x in history),'max_sum_finger_normal_force_n':max(sum(c['normal_force_n'] for c in x['contacts'] if 'finger' in c['body']) for x in history),'max_robot_fixture_contact_n':fixture_max,'final_locator_load_n':locator_load(last),'sim_time_s':last['sim_time'],'frames':len(history),'criteria':CRITERIA,'final':last}
