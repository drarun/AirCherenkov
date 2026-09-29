import sys
import os
import glob
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import uvicorn

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src'))
from sim.camera import Camera
from recon.hillas import compute_hillas

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'viewer_test', 'raw_fixed')
cam = Camera(n_rings=12)
pixel_x = cam.pixel_x.tolist()
pixel_y = cam.pixel_y.tolist()

events = []
file_list = glob.glob(os.path.join(DATA_DIR, '*.pt'))
if file_list:
    file_list.sort()
    print(f"Loading data from {file_list[0]}...")
    events = torch.load(file_list[0], weights_only=False)
    print(f"Loaded {len(events)} events.")

@app.get("/api/config")
def get_config():
    return {
        "num_events": len(events),
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
    }

adj_matrix = cam.get_neighbor_matrix()

def tailcut_clean(c_raw, pic_th=4.0, bnd_th=2.0, ped=2.0):
    c_sub = np.maximum(0.0, c_raw - ped)
    pic_mask = c_sub >= pic_th
    has_pic_neighbor = (adj_matrix @ pic_mask) > 0
    clean_mask = pic_mask | (has_pic_neighbor & (c_sub >= bnd_th))
    c_cleaned = np.where(clean_mask, c_sub, 0.0)
    return c_cleaned, clean_mask

@app.get("/api/events/{event_id}")
def get_event(event_id: int, clean: bool = True):
    if event_id < 0 or event_id >= len(events):
        raise HTTPException(status_code=404, detail="Event not found")
    
    event = events[event_id]
    fadc_traces = event['fadc_traces'] # shape [4, 469, 16]
    raw_charge = np.sum(fadc_traces, axis=2) # [4, 469]
    
    charges_out = []
    hillas_params = []
    for tel_idx in range(4):
        c_raw = raw_charge[tel_idx]
        if clean:
            c_disp, mask = tailcut_clean(c_raw, pic_th=4.0, bnd_th=2.0, ped=2.0)
        else:
            c_disp = c_raw
            mask = c_raw > 10.0
            
        charges_out.append(c_disp.tolist())
        
        if np.sum(mask) >= 3:
            h = compute_hillas(cam, c_disp, mask)
            if h is not None:
                hillas_params.append({
                    'centroid_x': float(h.centroid_x),
                    'centroid_y': float(h.centroid_y),
                    'length': float(h.length),
                    'width': float(h.width),
                    'psi': float(h.psi)
                })
            else:
                hillas_params.append(None)
        else:
            hillas_params.append(None)
    
    return {
        "energy": float(event['energy']),
        "label": int(event['label']),
        "impact_x": float(event['impact_x']),
        "impact_y": float(event['impact_y']),
        "charge": charges_out,
        "fadc_traces": fadc_traces.tolist(),
        "hillas": hillas_params,
        "cleaned": clean
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
