from gazeEstimator import Gaze, Face
import cv2
import numpy as np
from glob import glob 
import os
import json
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
face = Face()
estimator = Gaze(calib_file='jdc_cctv1_watch_v2.calib', input_size=(2592, 1944))
x, y = np.mgrid[0:685,0:1325]
pos = np.dstack((x, y))
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
        

def get_gaze_heat_map(res, factor=10):
    heatmaps = []
    landmark_indices = face.get_landmark_indices()
    for info in res['recog_face']:
        obj_pts = face.get_object_points(info['sex'], int(info['age']))
        if 'face_landmark' in info.keys():
            img_pts = np.array(info['face_landmark'])[landmark_indices]
            # img_pts[:,0]*=(2592/1280)
            # img_pts[:,1]*=(1922/720)
            gaze, cov = estimator.estimate_gaze_point(obj_pts, img_pts)

            if gaze is not None:
                # print('DiD:(685,1715),(0,500)')
                # print('gaze:',gaze)
                gaze[1]-=390
                cov = scaling(factor, cov)
                rv = multivariate_normal(gaze, cov)
                p = rv.pdf(pos)
                npixs=(p>eps).sum()
                if npixs>100:
                    
                    pp = p.copy()
                    pp = pp.T
                    pp/=pp.max()
                    pp*=255
                    pp=pp.astype(np.uint8)
                    pp = cv2.cvtColor(pp, cv2.COLOR_GRAY2BGR)
                    pp = cv2.applyColorMap(pp, cv2.COLORMAP_JET)
                    pp = cv2.flip(pp, 1)
                    heatmaps.append(pp)

    return heatmaps
 




if __name__=="__main__":
    root = 'data/scenario2/s2_1_avi_json/*.json'
    files = sorted(glob(root), key = lambda x: int(os.path.basename(x).split('.')[0]))
    for file in files:
        with open(file, 'r') as f:
            contents=json.load(f)
            heatmaps = get_gaze_heat_map(contents, factor=100)
            for heatmap in heatmaps:
                cv2.imshow('pp',heatmap)
                cv2.waitKey(0)
