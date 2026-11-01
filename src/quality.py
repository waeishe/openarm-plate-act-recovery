"""Independent task scoring. Never sent to the acting model."""
CRITERIA={'xy_error_m':.003,'z_error_m':.002,'tilt_deg':1,'yaw_deg':3,'stable_s':2,'linear_speed_m_s':.005,'angular_speed_rad_s':.03,'locator_load_n':.02,'fixture_collision_n':.5,'supported_release_s':.2,'lift_clearance_m':.010,'lift_hold_s':.3}

def seated(x):
    return (x['xy_error_m']<=.003 and abs(x['z_error_m'])<=.002 and x['tilt_deg']<=1 and abs(x['yaw_deg'])<=3 and x['linear_speed']<.005 and x['angular_speed']<.03 and x['destination_supported'] and not x['finger_bodies'] and x['left_finger_angle']>.65 and x['right_finger_angle']>.65 and x['hand_distance_m']>.05 and sum(c['normal_force_n'] for c in x['contacts'] if 'locator' in c['geom'])<=.02)

def score(history,release_times,grasp_count,error=None):
    last=history[-1];tail=[x for x in history if x['sim_time']>=last['sim_time']-2-1e-8]
    stable=tail[-1]['sim_time']-tail[0]['sim_time']>=2-1e-6 and all(seated(x) for x in tail)
    held=maxheld=0.;acquired=False;missing=0;loss=[]
    for x in history:
        good=len(x['finger_bodies'])>=2 and x['min_plate_z']>1.037 and not x['destination_supported']
        held=held+1/30 if good else 0.;maxheld=max(maxheld,held)
        if held>=.3:acquired=True
        absent=acquired and not x['finger_bodies'] and not x['destination_supported']
        missing=missing+1 if absent else 0
        if missing==3:loss.append(x['sim_time'])
    support_times=[]
    for t in release_times:
        before=[x for x in history if x['sim_time']<=t+1e-8];duration=0.
        for x in reversed(before):
            if not x['destination_supported']:break
            duration+=1/30
        support_times.append(duration)
    collision=max((c['normal_force_n'] for x in history for c in x['robot_fixture_contacts']),default=0.)
    manipulation_ok=not grasp_count or (maxheld>=.3 and support_times and support_times[-1]>=.2 and not loss)
    success=bool(stable and manipulation_ok and collision<=.5 and error is None)
    return {'outcome':'success' if success else 'failure','error':error,'initial_correctly_seated':seated(history[0]),'final_correctly_seated':seated(last),'final_stable_2s':stable,'grasp_commands':grasp_count,'two_finger_clear_hold_s':maxheld,'supported_before_release_s':support_times,'unsupported_losses':loss,'max_fixture_contact_n':collision,'elapsed_sim_s':last['sim_time']-history[0]['sim_time'],'final_xy_error_mm':last['xy_error_m']*1000,'final_tilt_deg':last['tilt_deg'],'final_yaw_deg':last['yaw_deg'],'final_locator_load_n':sum(c['normal_force_n'] for c in last['contacts'] if 'locator' in c['geom']),'max_tilt_deg':max(x['tilt_deg'] for x in history),'criteria':CRITERIA,'final':last}
