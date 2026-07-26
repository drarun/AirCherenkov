import plotly.graph_objects as go
import plotly.express as px
from shower import ShowerSimulation

def plot_shower(df, title="Air Shower 3D Tracks"):
    """
    Plots the 3D tracks of particles in an air shower using Plotly.
    """
    # Create a color map for different particle types
    color_map = {
        'gamma': 'yellow',
        'e-': 'blue',
        'e+': 'red',
        'proton': 'white',
        'pi0': 'green',
        'pi_charged': 'orange',
        'mu': 'purple'
    }

    fig = go.Figure()

    # We plot each particle track as a separate 3D line
    # To optimize Plotly, we can group by particle_id
    grouped = df.groupby('particle_id')
    
    for pid, group in grouped:
        particle_type = group['pid'].iloc[0]
        
        fig.add_trace(go.Scatter3d(
            x=group['x'],
            y=group['y'],
            z=group['z'],
            mode='lines',
            line=dict(
                color=color_map.get(particle_type, 'gray'),
                width=2
            ),
            name=particle_type,
            showlegend=False,
            hoverinfo='text',
            hovertext=f"Type: {particle_type}<br>Energy: {group['energy'].iloc[0]:.2f} GeV"
        ))

    # Add custom legend entries
    for ptype, color in color_map.items():
        if ptype in df['pid'].unique():
            fig.add_trace(go.Scatter3d(
                x=[None], y=[None], z=[None],
                mode='lines',
                line=dict(color=color, width=4),
                name=ptype
            ))

    # Reverse Z axis so 0 is at the bottom (ground)
    fig.update_layout(
        title=title,
        template='plotly_dark',
        scene=dict(
            xaxis_title='X (meters)',
            yaxis_title='Y (meters)',
            zaxis_title='Z (Height in meters)',
            aspectmode='data',
            zaxis=dict(autorange="reversed") 
        ),
        legend_title="Particle Type",
        width=900,
        height=800
    )
    
    return fig

if __name__ == "__main__":
    print("Simulating Gamma-ray Shower...")
    gamma_sim = ShowerSimulation('gamma', energy=500.0)
    gamma_sim.run(max_generations=10)
    gamma_df = gamma_sim.get_tracks_dataframe()
    gamma_cherenkov = gamma_sim.get_cherenkov_dataframe()
    
    print("Simulating Hadronic (Proton) Shower...")
    proton_sim = ShowerSimulation('proton', energy=500.0)
    proton_sim.run(max_generations=10)
    proton_df = proton_sim.get_tracks_dataframe()
    proton_cherenkov = proton_sim.get_cherenkov_dataframe()
    
    # Generate HTML files
    print("Generating interactive HTML plots for tracks...")
    fig_gamma = plot_shower(gamma_df, "Gamma-ray Shower (Electromagnetic Cascade)")
    fig_gamma.write_html("gamma_shower.html")
    
    fig_proton = plot_shower(proton_df, "Proton Shower (Hadronic Cascade)")
    fig_proton.write_html("proton_shower.html")
    
    # Generate Cherenkov 2D Plots
    print("Generating Cherenkov ground pool plots...")
    if not gamma_cherenkov.empty:
        fig_c_gamma = px.scatter(gamma_cherenkov, x='x', y='y', opacity=0.1, 
                                 title="Gamma-ray Cherenkov Footprint at Ground",
                                 width=800, height=800, template='plotly_dark')
        fig_c_gamma.update_layout(xaxis_title="Ground X (m)", yaxis_title="Ground Y (m)")
        fig_c_gamma.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_c_gamma.write_html("gamma_cherenkov_pool.html")

    if not proton_cherenkov.empty:
        fig_c_proton = px.scatter(proton_cherenkov, x='x', y='y', opacity=0.1, 
                                  title="Proton Cherenkov Footprint at Ground",
                                  width=800, height=800, template='plotly_dark')
        fig_c_proton.update_layout(xaxis_title="Ground X (m)", yaxis_title="Ground Y (m)")
        fig_c_proton.update_yaxes(scaleanchor="x", scaleratio=1)
        fig_c_proton.write_html("proton_cherenkov_pool.html")
        
    print("Done! You can open the generated .html files in your browser.")
