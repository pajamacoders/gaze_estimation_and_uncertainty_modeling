from .gazeEstimator import Gaze, Face
import numpy as np
import json
from fastapi import FastAPI
from pydantic import BaseModel


class RecogData(BaseModel):
    device_info: int
    Scene_Info: dict
    recog_human: list
    recog_face: list

app = FastAPI()

id2device_conf =json.load(open("app/device_info.json",'r'))
face = Face()
estimator = Gaze(calib_file='app/calib_json/jdc_cctv1_watch_v2.calib', input_size=(2592, 1944))

@app.put("/gaze/{device_id}/")
async def estimate_gaze_point(device_id:str, data:RecogData):

    dev_info = id2device_conf[device_id.upper()]
    calib_file = dev_info["calib_file"]
    img_height = dev_info["image_height"]
    img_width = dev_info["image_width"]
    with open(calib_file, 'r') as f:
        info = json.load(f)
    calib_img_height = info['img_height']
    calib_img_width = info['img_width']
    estimator.set_calib_param(calib_file, calib_img_width, calib_img_height)
    landmark_indices = face.get_landmark_indices()
    recog_data = data.dict()
    gaze_list = []
    for info in recog_data['recog_face']:
        obj_pts = face.get_object_points(info['sex'], int(info['age']))

        if 'face_landmark' not in info.keys():
            continue 
        
        img_pts = np.array(info['face_landmark'])[landmark_indices]
        # scale image point to the same scale with the image I used to calibrate the camera
        img_pts[:,0]*=(calib_img_width/img_width)
        img_pts[:,1]*=(calib_img_height/img_height)
        gaze, cov = estimator.estimate_gaze_point(obj_pts, img_pts)
        gaze_list.append({"face_id":info['face_id'], "gaze": gaze.tolist(), "gaze_cov":cov.tolist()})
    return {"device_id":device_id, "gaze_info": gaze_list}



# if __name__=="__main__":
#     root = 'data/scenario2/s2_1_avi_json/*.json'
#     files = sorted(glob(root), key = lambda x: int(os.path.basename(x).split('.')[0]))
#     for file in files:
#         with open(file, 'r') as f:
#             contents=json.load(f)
#             heatmaps = get_gaze_heat_map(contents, factor=100)
#             for heatmap in heatmaps:
#                 cv2.imshow('pp',heatmap)
#                 cv2.waitKey(0)
