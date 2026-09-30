import sys; sys.path.insert(0, 'src')
import torch
import numpy as np

batch = torch.load('data/test_20_gammas_fixed/sim_batch_00000.pt', weights_only=False)
print('Type:', type(batch))
if isinstance(batch, list):
    print('Num events in batch:', len(batch))
    ev = batch[0]
    print('Event keys:', list(ev.keys()))
    print('FADC traces shape:', ev['fadc_traces'].shape)
    print('Label:', ev['label'])
    print('Energy:', ev['energy'])
    
    # Check FADC for first 5 events
    for i, ev in enumerate(batch[:5]):
        traces = ev['fadc_traces']  # (4, 469, 16) or similar
        max_pe = traces.sum(axis=-1).max()
        print(f"Event {i}: max_PE={max_pe:.1f}, label={ev['label']}")
