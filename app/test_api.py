from fastapi import FastAPI
from fastapi.testclient import TestClient
from .gaze_estimation_api import app
import json
import os
import cv2
from random import shuffle
from .heat_map_api import get_gaze_heat_map
url = "/gaze/"

client = TestClient(app)

def test_read_main():
    dataset = {"P2achJNX001":"data/scenario1/s1_1_avijson/json",
     "P2ACHJNX002":"", 
     "P2ACHJNX003":"", 
     "p2achjnx004":"data/air_space_001", 
     "P2ACHJNX005":"", 
     "P2ACHJNX006":"", 
     "P2ACHJNX007":"", 
     "P2ACHJNX008":""}
    dummy_data = []
    for key, value in dataset.items():
        
        if value == "":
            continue
        [dummy_data.append((key, os.path.join(value, fname))) for fname in os.listdir(value)]

    shuffle(dummy_data)
    for id, fpath in dummy_data:
        payload = json.load(open(fpath, 'r'))
        response = client.put(
            f"/gaze/{id}/",
            # headers={"X-Token": "coneofsilence"},
            json=payload,
        )
        assert response.status_code == 200
        gaze_est_res = response.json()
        id = gaze_est_res['device_id']
        for gaze_info in gaze_est_res['gaze_info']:
            heat_map = get_gaze_heat_map(id, gaze_info['gaze'], gaze_info['gaze_cov'])
            if heat_map is not None:
                face_id = gaze_info['face_id']
                # cv2.imshow(f'{id}_{face_id}', heat_map)
                # cv2.waitKey(0)

            

