from sim.shower import ShowerSimulation
from sim.visualize import plot_shower


def test_plot_shower_combines_segments_by_particle_type(tmp_path):
    sim = ShowerSimulation(
        ['gamma', 'proton'], energies=[10.0, 20.0],
        z_starts=10000.0, record_tracks=True,
    )
    sim.step()
    tracks = sim.get_tracks_dataframe()

    figure = plot_shower(tracks, title='Track smoke test')
    assert {trace.name for trace in figure.data} == {'gamma', 'proton'}
    assert all(trace.type == 'scatter3d' for trace in figure.data)
    assert all(None in trace.x for trace in figure.data)
    assert figure.layout.scene.zaxis.autorange != 'reversed'

    output = tmp_path / 'tracks.html'
    figure.write_html(output)
    html = output.read_text(encoding='utf-8')
    assert 'Track smoke test' in html
    assert 'scatter3d' in html
    assert 'plotly' in html.lower()


def test_plot_shower_handles_an_empty_recording():
    tracks = ShowerSimulation('gamma').get_tracks_dataframe()
    figure = plot_shower(tracks)
    assert len(figure.data) == 0
    assert figure.layout.annotations[0].text == 'No particle tracks were recorded'
