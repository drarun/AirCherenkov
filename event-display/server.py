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

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'test', 'raw')
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

@app.get("/api/events/{event_id}")
def get_event(event_id: int):
    if event_id < 0 or event_id >= len(events):
        raise HTTPException(status_code=404, detail="Event not found")
    
    event = events[event_id]
    fadc_traces = event['fadc_traces'] # shape [4, 469, 16]
    
    # Calculate integrated charge per pixel: [4, 469]
    charge = np.sum(fadc_traces, axis=2).tolist()
    
    return {
        "energy": float(event['energy']),
        "label": int(event['label']),
        "impact_x": float(event['impact_x']),
        "impact_y": float(event['impact_y']),
        "charge": charge,
        "fadc_traces": fadc_traces.tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
