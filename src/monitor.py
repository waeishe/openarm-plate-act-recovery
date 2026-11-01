"""Shared completion monitor. RGB-D + proprioception, never evaluation state."""
import math
import numpy as np

def check_observation(o):
    p=o.get('perception',{});plane=p.get('plate_plane');pose=p.get('global_grid_pose') or p.get('near_seat_grid_pose')
    if not plane or not pose:return False
    bounds=np.array(plane['observed_xy_bounds_m']);mid=bounds.mean(0);center=np.array(pose['center_xy_m'])
    # Global gate prevents applying destination lattice alias to source observations.
    if np.linalg.norm(mid-[.43,-.08])>.025:return False
    return bool(np.linalg.norm(center-[.43,-.08])<=.002 and abs(pose['yaw_deg'])<=3 and plane['tilt_deg']<=1 and abs(plane['floor_z_at_destination_center_m']-1.03056)<=.001 and abs(o['joints'][15])>.65 and abs(o['joints'][7])>.65 and min(np.linalg.norm(np.array(o[arm]['position'])-np.r_[center,1.027]) for arm in ['left','right'])>.05)

class Monitor:
    def __init__(self):self.first=None;self.samples=[]
    def update(self,t):
        o=t.observe('biolab_top',True);valid=check_observation(o);now=float(o['sim_time'])
        if valid and self.first is None:self.first=now
        if not valid:self.first=None
        complete=self.first is not None and now-self.first>=2-1e-6
        self.samples.append({'observation_id':o['id'],'sim_time':now,'valid':valid,'complete':complete})
        return complete
