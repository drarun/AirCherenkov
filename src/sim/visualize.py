import plotly.graph_objects as go
import plotly.express as px

COLOR_MAP = {
    'gamma': '#FFD700',
    'e-': '#4DA6FF',
    'e+': '#FF4D4D',
    'proton': '#FFFFFF',
    'pi0': '#00FF88',
    'pi_charged': '#FF8C00',
    'mu': '#DA70D6',
}


def plot_shower(df, title="Air Shower 3D Tracks"):
    """
    Plot particle segments as an interactive 3D Plotly figure.

    Segments are combined into one WebGL trace per particle type instead of
    one trace per particle, which keeps large showers responsive in a browser.
    """
    required = {
        'particle_id', 'pid', 'energy', 'generation',
        'x', 'y', 'z', 'event_id',
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Track dataframe is missing columns: {', '.join(sorted(missing))}")

    fig = go.Figure()

    for particle_type in COLOR_MAP:
        particle_rows = df[df['pid'] == particle_type]
        if particle_rows.empty:
            continue

        x_values = []
        y_values = []
        z_values = []
        hover_values = []
        for particle_id, segment in particle_rows.groupby('particle_id', sort=False):
            first = segment.iloc[0]
            hover = (
                f"Type: {particle_type}<br>"
                f"Particle: {int(particle_id)}<br>"
                f"Energy: {first['energy']:.3g} GeV<br>"
                f"Generation: {int(first['generation'])}<br>"
                f"Event: {int(first['event_id'])}"
            )
            x_values.extend([*segment['x'].tolist(), None])
            y_values.extend([*segment['y'].tolist(), None])
            z_values.extend([*segment['z'].tolist(), None])
            hover_values.extend([hover] * len(segment) + [None])

        fig.add_trace(go.Scatter3d(
            x=x_values,
            y=y_values,
            z=z_values,
            mode='lines',
            line=dict(color=COLOR_MAP[particle_type], width=2),
            name=particle_type,
            hoverinfo='text',
            hovertext=hover_values,
            connectgaps=False,
        ))

    unknown_types = sorted(set(df['pid'].unique()) - set(COLOR_MAP))
    for particle_type in unknown_types:
        particle_rows = df[df['pid'] == particle_type]
        x_values, y_values, z_values = [], [], []
        for _, segment in particle_rows.groupby('particle_id', sort=False):
            x_values.extend([*segment['x'].tolist(), None])
            y_values.extend([*segment['y'].tolist(), None])
            z_values.extend([*segment['z'].tolist(), None])
        fig.add_trace(go.Scatter3d(
            x=x_values, y=y_values, z=z_values,
            mode='lines', line=dict(color='gray', width=2),
            name=str(particle_type), connectgaps=False,
        ))

    fig.update_layout(
        title=title,
        template='plotly_dark',
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Altitude (meters)',
            aspectmode='data',
        ),
        legend_title="Particle Type",
        autosize=True,
        height=800,
        margin=dict(l=0, r=0, b=0, t=60),
    )

    if df.empty:
        fig.add_annotation(
            text="No particle tracks were recorded",
            x=0.5, y=0.5, xref='paper', yref='paper', showarrow=False,
        )

    return fig


# Backward-compatible public alias for older callers.
plot_shower_3d = plot_shower


def plot_cherenkov_pool(cherenkov_df, title, range_m=1500.0):
    """Plot an interactive weighted Cherenkov-packet footprint."""
    hover_data = {'weight': ':.3g'} if 'weight' in cherenkov_df.columns else None
    fig = px.scatter(
        cherenkov_df,
        x='x',
        y='y',
        hover_data=hover_data,
        opacity=0.05,
        title=title,
        width=900,
        height=900,
        range_x=[-range_m, range_m],
        range_y=[-range_m, range_m],
        template='plotly_dark',
    )
    fig.update_traces(marker=dict(size=3, color='#FFD700'))
    fig.update_layout(
        paper_bgcolor='#0a0a0a',
        plot_bgcolor='#0a0a0a',
        xaxis_title="Ground X (m)",
        yaxis_title="Ground Y (m)",
        font=dict(family='Courier New', color='#e0e0e0'),
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig
