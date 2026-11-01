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
        residual=pts[:,2]-a@coeff;scale=max(.00008,2.5*float(np.median(np.abs(residual[keep]))))
        new=np.abs(residual)<scale
        if new.sum()<40:return result,mask
        keep=new
    coeff=np.linalg.lstsq(a[keep],pts[keep,2],rcond=None)[0];error=float(np.sqrt(np.mean((pts[keep,2]-a[keep]@coeff)**2)))
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
            u,_,vt=np.linalg.svd(l0.T@w0);rot=u@vt
            if np.linalg.det(rot)>0:
                center=world.mean(axis=0)-local.mean(axis=0)@rot;err=float(np.sqrt(np.mean(np.sum((local@rot+center-world)**2,axis=1))))
                if err<.0015:
                    result['near_seat_grid_pose']={'center_xy_m':center.tolist(),'yaw_deg':float(np.degrees(np.arctan2(rot[0,1],rot[0,0]))),'rms_m':err,'visible_wells':len(world),'validity':'Local estimate assumes plate within 4 mm and 3 degrees of nominal destination; not a global pose detector.'}
    return result,mask
