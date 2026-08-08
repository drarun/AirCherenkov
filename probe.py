import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
import torch

try:
    from sim.camera import Camera
    cam = Camera(n_rings=12)
    print("Camera instantiated.")
    print("pixel_x shape:", cam.pixel_x.shape)
except Exception as e:
    print("Camera import error:", e)

data = torch.load('data/test/raw/sim_batch_00000.pt', weights_only=False)
print("Data type:", type(data))
print("List length:", len(data))
if len(data) > 0:
    print("Type of first element:", type(data[0]))
    if isinstance(data[0], dict):
        print("Keys:", data[0].keys())
        for k, v in data[0].items():
            if isinstance(v, torch.Tensor):
                print(f"{k} shape: {v.shape}")
            else:
                print(f"{k}: {type(v)}")
