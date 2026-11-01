"""Plate-plane estimates from RGB pixels and metric depth, without simulator IDs."""
import numpy as np
from aspire_lab.vision import point_cloud

def estimate(rgb,depth,calibration):
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        cloud=point_cloud(np.where(np.isfinite(depth)&(depth>.01)&(depth<10),depth,np.nan),calibration)
    r,g,b=np.moveaxis(rgb.astype(float),-1,0)
    # Blue well-floor markings differ from pale plate walls and teal/orange seats.
    mask=(b>1.08*g)&(g>1.15*r)&(b-r>25)&(g>35)&np.isfinite(depth)
    # Known, calibrated work volume; these are not measured object bounds.
    mask&=(cloud[:,:,0]>.32)&(cloud[:,:,0]<.55)&(cloud[:,:,1]>-.18)&(cloud[:,:,1]<.19)&(cloud[:,:,2]>1.015)&(cloud[:,:,2]<1.18)
    pts=cloud[mask]
    result={'source':'RGB_color_segmentation_and_rendered_metric_depth','pixels':len(pts),'plate_plane':None}
    if len(pts)<40:return result,mask
    span=np.ptp(pts[:,:2],axis=0)
    if span[0]<.035 or span[1]<.025:return result,mask
    # Robust least squares: reject edge pixels mixing floors and walls.
    a=np.c_[pts[:,0]-.43,pts[:,1]+.08,np.ones(len(pts))];keep=np.ones(len(pts),bool)
    for _ in range(5):
        coeff=np.linalg.lstsq(a[keep],pts[keep,2],rcond=None)[0]
        residual=pts[:,2]-np.sum(a*coeff,axis=1);scale=max(.00008,2.5*float(np.median(np.abs(residual[keep]))))
        new=np.abs(residual)<scale
        if new.sum()<40:return result,mask
        keep=new
    coeff=np.linalg.lstsq(a[keep],pts[keep,2],rcond=None)[0];error=float(np.sqrt(np.mean((pts[keep,2]-np.sum(a[keep]*coeff,axis=1))**2)))
    if error>.0006:return result,mask
    result['plate_plane']={'tilt_deg':float(np.degrees(np.arctan(np.linalg.norm(coeff[:2])))),'slope_z_per_x':float(coeff[0]),'slope_z_per_y':float(coeff[1]),'floor_z_at_destination_center_m':float(coeff[2]),'height_difference_x_ends_mm':float(coeff[0]*.12776*1000),'height_difference_y_sides_mm':float(coeff[1]*.08548*1000),'observed_xy_span_m':span.tolist(),'observed_xy_bounds_m':[pts[keep,:2].min(axis=0).tolist(),pts[keep,:2].max(axis=0).tolist()],'fit_rms_m':error,'inlier_pixels':int(keep.sum()),'notes':'Plane of visible well floors, not an object-state oracle; partial occlusion can bias bounds. No contact or success label is inferred.'}
    result['near_seat_grid_pose']=None
    # Index the known 9 mm lattice only close to the calibrated nominal seat.
    # This is explicitly a local estimator: unknown shifts >= half a pitch alias.
    if result['plate_plane']['tilt_deg']<5 and 1.025<coeff[2]<1.036:
        remaining=set(zip(*np.where(mask)));groups=[]
        while remaining:
            first=remaining.pop();todo=[first];group=[first]
            while todo:
                v,u=todo.pop()
                for nxt in [(v-1,u),(v+1,u),(v,u-1),(v,u+1)]:
                    if nxt in remaining:remaining.remove(nxt);todo.append(nxt);group.append(nxt)
            if 3<=len(group)<=100:groups.append(np.asarray(group))
        world=[];local=[]
        for group in groups:
            p=np.mean(cloud[group[:,0],group[:,1]],axis=0)
            col=int(round((p[0]-.43)/.009+5.5));row=int(round((p[1]+.08)/.009+3.5))
            if 0<=col<12 and 0<=row<8:world.append(p[:2]);local.append([(col-5.5)*.009,(row-3.5)*.009])
        if len(world)>=8:
            world=np.asarray(world);local=np.asarray(local);w0=world-world.mean(axis=0);l0=local-local.mean(axis=0)
            u,_,vt=np.linalg.svd(np.einsum('ni,nj->ij',l0,w0));rot=np.einsum('ij,jk->ik',u,vt)
            if np.linalg.det(rot)>0:
                center=world.mean(axis=0)-np.einsum('i,ij->j',local.mean(axis=0),rot);err=float(np.sqrt(np.mean(np.sum((np.einsum('ni,ij->nj',local,rot)+center-world)**2,axis=1))))
                if err<.0015:
                    result['near_seat_grid_pose']={'center_xy_m':center.tolist(),'yaw_deg':float(np.degrees(np.arctan2(rot[0,1],rot[0,0]))),'rms_m':err,'visible_wells':len(world),'validity':'Local estimate assumes plate within 4 mm and 3 degrees of nominal destination; not a global pose detector.'}
    return result,mask

# Global indexing is only accepted when visible well centers span the entire
# 12x8 lattice. Otherwise retain the explicitly local estimate or return null.
_local_estimate=estimate

def estimate(rgb,depth,calibration):
    result,mask=_local_estimate(rgb,depth,calibration)
    result['global_grid_pose']=None
    plane=result.get('plate_plane')
    if not plane or plane['tilt_deg']>=5:return result,mask
    with np.errstate(invalid='ignore',divide='ignore',over='ignore'):
        cloud=point_cloud(np.where(np.isfinite(depth)&(depth>.01)&(depth<10),depth,np.nan),calibration)
    remaining=set(zip(*np.where(mask)));world=[]
    while remaining:
        first=remaining.pop();todo=[first];group=[first]
        while todo:
            v,u=todo.pop()
            for nxt in [(v-1,u),(v+1,u),(v,u-1),(v,u+1)]:
                if nxt in remaining:remaining.remove(nxt);todo.append(nxt);group.append(nxt)
        if 3<=len(group)<=100:
            g=np.array(group);world.append(cloud[g[:,0],g[:,1]].mean(0)[:2])
    if len(world)<20:return result,mask
    world=np.array(world);span=np.ptp(world,axis=0)
    if span[0]<.096 or span[1]<.060:return result,mask
    center=(world.max(0)+world.min(0))/2
    indexes=np.round((world-center)/.009+[5.5,3.5]).astype(int)
    good=(indexes[:,0]>=0)&(indexes[:,0]<12)&(indexes[:,1]>=0)&(indexes[:,1]<8)
    world=world[good];indexes=indexes[good]
    if len(world)<20 or set(indexes[:,0])!=set(range(12)) or set(indexes[:,1])!=set(range(8)):return result,mask
    local=(indexes-[5.5,3.5])*.009;w0=world-world.mean(0);l0=local-local.mean(0)
    u,_,vt=np.linalg.svd(np.einsum('ni,nj->ij',l0,w0));rot=np.einsum('ij,jk->ik',u,vt)
    if np.linalg.det(rot)<0:return result,mask
    center=world.mean(0)-np.einsum('i,ij->j',local.mean(0),rot)
    err=float(np.sqrt(np.mean(np.sum((np.einsum('ni,ij->nj',local,rot)+center-world)**2,axis=1))))
    yaw=float(np.degrees(np.arctan2(rot[0,1],rot[0,0])))
    if not np.isfinite([*center,err,yaw]).all() or err>.001 or abs(yaw)>5:return result,mask
    result['global_grid_pose']={'center_xy_m':center.tolist(),'yaw_deg':yaw,'rms_m':err,'visible_wells':len(world),'validity':'Requires visible centers spanning all12 columns and8 rows; known9mm well lattice, plane tilt<5deg. Can locate source or destination plate. Occlusion may produce null; clear the view and reobserve.'}
    # Do not present an aliased local estimate when the global fit contradicts it.
    if np.linalg.norm(center-[.43,-.08])>.004:result['near_seat_grid_pose']=None
    return result,mask
