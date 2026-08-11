import os
import sys

sys.path.insert(0, 'src')
from generate_training_data import generate_training_data

def main():
    output_dir = 'data/crab_mc_750m'
    print("Starting 150,000 event test generation with R_max = 750 meters...")
    generate_training_data(
        num_gammas=50000,
        num_hadrons=100000,
        batch_size=100,
        save_every=1000,
        output_dir=output_dir,
        zenith_deg=20.0,
        azimuth_deg=180.0,
        e_min=10.0,
        e_max=30000.0,
        spectral_index=2.0,
        impact_radius=750.0,
        debug=False
    )
    print("\nGeneration complete. Now generating throw radius plot on OneDrive Desktop...")
    plot_script = r"C:\Users\aruns\.gemini\antigravity-cli\brain\f0e1b905-b5d2-4fce-80a1-d0620d2d7de7\scratch\plot_throw_radius.py"
    os.system(f'python "{plot_script}"')
    print("Done! Plot has been generated on OneDrive Desktop.")

if __name__ == '__main__':
    main()
