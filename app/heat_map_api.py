import numpy as np
import cv2
from scipy.stats import multivariate_normal
import json
import matplotlib.pyplot as plt
id2device_conf =json.load(open("app/device_info.json",'r'))


eps=1e-5

def scaling(factor, cov):
    val, vec = np.linalg.eig(cov)
    idx = np.argsort(val)[::-1]
    val=val[idx]
    multiple=val[0]/val[1]
    if multiple >=5:
        val[0]/=factor
        vec=vec[:,idx]
        A = np.eye(2)*val
        cov =vec@A@vec.T
    return cov

def get_gaze_heat_map(id:str, gaze:np.ndarray, gaze_cov:np.ndarray, factor:int=10):
    """
    id: DID id,
    
    gaze: gaze point(x,z) position
    gaze_cov: 2d covariance matrix of gaze point
    factor: likelihood ellipse
    """
    if gaze is None or gaze_cov is None:
        return None
    try:
        screen_info = id2device_conf[id.upper()]
    except KeyError as e:
        return None
    width = screen_info["screen_width"]
    height = screen_info["screen_height"]
    btm_x = screen_info["screen_right_bottom_x"]
    btm_z = screen_info["screen_right_bottom_z"]
    x,z = np.mgrid[0:width,0:height]
    pos = np.dstack((x, z))  
    
    # print('DiD:(685,1715),(0,500)')
    # print('gaze:',gaze)
    gaze[0]-=btm_x
    gaze[1]-=btm_z
    gaze_cov = scaling(factor, gaze_cov)
    rv = multivariate_normal(gaze, gaze_cov)
    p = rv.pdf(pos)
    npixs=(p>eps).sum()
    print(f'{id}:{pos.shape}, npixs:{npixs}')  
    pp=None
    if npixs>100:
        pp = p.copy()
        pp = pp.T
        pp/=pp.max()
        pp*=255
        pp=pp.astype(np.uint8)
        pp = cv2.cvtColor(pp, cv2.COLOR_GRAY2BGR)
        pp = cv2.applyColorMap(pp, cv2.COLORMAP_JET)
        pp = cv2.flip(pp, 1)
        return pp
    return pp

