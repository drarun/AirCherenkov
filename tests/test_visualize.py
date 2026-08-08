import pandas as pd
import numpy as np
from sim.visualize import plot_shower, plot_cherenkov_pool

def test_plot_shower_combines_segments_by_particle_type(tmp_path):
    # Construct a mock tracks DataFrame instead of running a full simulation
    # State fields: particle_id, pid, energy, generation, x, y, z, event_id
    mock_tracks = pd.DataFrame([
        {
            'particle_id': 0,
            'pid': 'gamma',
            'energy': 10.0,
            'generation': 0,
            'x': 0.0,
            'y': 0.0,
            'z': 10000.0,
            'event_id': 0
        },
        {
            'particle_id': 0,
            'pid': 'gamma',
            'energy': 10.0,
            'generation': 0,
            'x': 10.0,
            'y': 10.0,
            'z': 9000.0,
            'event_id': 0
        },
        {
            'particle_id': 1,
            'pid': 'proton',
            'energy': 20.0,
            'generation': 0,
            'x': 0.0,
            'y': 0.0,
            'z': 10000.0,
            'event_id': 1
        },
        {
            'particle_id': 1,
            'pid': 'proton',
            'energy': 20.0,
            'generation': 0,
            'x': -10.0,
            'y': -10.0,
            'z': 9000.0,
            'event_id': 1
        }
    ])

    figure = plot_shower(mock_tracks, title='Track smoke test')
    assert {trace.name for trace in figure.data} == {'gamma', 'proton'}
    assert all(trace.type == 'scatter3d' for trace in figure.data)
    assert all(None in trace.x for trace in figure.data)

    output = tmp_path / 'tracks.html'
    figure.write_html(output)
    html = output.read_text(encoding='utf-8')
    assert 'Track smoke test' in html
    assert 'scatter3d' in html
    assert 'plotly' in html.lower()

def test_plot_shower_handles_an_empty_recording():
    # Construct empty DataFrame with columns
    tracks = pd.DataFrame(columns=['particle_id', 'pid', 'energy', 'generation', 'x', 'y', 'z', 'event_id'])
    figure = plot_shower(tracks)
    assert len(figure.data) == 0
    assert figure.layout.annotations[0].text == 'No particle tracks were recorded'

def test_plot_cherenkov_pool():
    # Construct mock Cherenkov DataFrame
    cherenkov_df = pd.DataFrame({
        'x': np.random.normal(0, 100, 100),
        'y': np.random.normal(0, 100, 100),
        'weight': np.ones(100)
    })
    figure = plot_cherenkov_pool(cherenkov_df, title="Cherenkov Pool Test")
    assert len(figure.data) == 1
    assert figure.data[0].type == 'scatter'
