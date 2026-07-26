import torch
import numpy as np
import matplotlib.pyplot as plt
from sbi.inference import SNPE
from sbi.utils import BoxUniform
import sbi.analysis as analysis
import os

def build_prior():
    # Parameters: [Phi_0, Gamma]
    # Phi_0 in some arbitrary flux units, e.g., Uniform(1.0, 10.0)
    # Gamma (spectral index), e.g., Uniform(1.5, 3.5)
    prior = BoxUniform(low=torch.tensor([1.0, 1.5]), 
                       high=torch.tensor([10.0, 3.5]))
    return prior

def simulator(theta):
    """
    Simulates a histogram of true energies given spectral parameters.
    theta: 1D tensor [Phi_0, Gamma]
    Returns:
    1D tensor of counts in energy bins
    """
    phi_0 = theta[0].item()
    gamma = theta[1].item()
    
    # Define energy bins (e.g., from 0.1 to 100 TeV in log space)
    E_min = 0.1
    E_max = 100.0
    n_bins = 20
    edges = np.logspace(np.log10(E_min), np.log10(E_max), n_bins + 1)
    
    # Calculate expected number of events in each bin
    # Integral of Phi_0 * E^{-Gamma} dE from E1 to E2
    # = Phi_0 / (1 - Gamma) * (E2**(1 - Gamma) - E1**(1 - Gamma))
    counts = np.zeros(n_bins)
    for i in range(n_bins):
        E1 = edges[i]
        E2 = edges[i+1]
        if abs(gamma - 1.0) < 1e-4:
            # log case
            mu = phi_0 * (np.log(E2) - np.log(E1))
        else:
            mu = phi_0 / (1.0 - gamma) * (E2**(1.0 - gamma) - E1**(1.0 - gamma))
            
        # Optional: scale by some exposure factor to get reasonable counts
        exposure = 1000.0 
        mu *= exposure
        
        # Sample Poisson counts
        counts[i] = np.random.poisson(mu)
        
    return torch.tensor(counts, dtype=torch.float32)

def main():
    print("Setting up prior and simulator...")
    prior = build_prior()
    
    # Test simulator
    test_theta = torch.tensor([5.0, 2.5])
    test_x = simulator(test_theta)
    print(f"Test simulator output shape: {test_x.shape}")
    
    print("Simulating training dataset...")
    # Simulate 5000 datasets for training
    num_simulations = 5000
    theta = prior.sample((num_simulations,))
    x = torch.stack([simulator(t) for t in theta])
    
    print("Training Neural Posterior Estimator (SNPE)...")
    inference = SNPE(prior=prior)
    inference = inference.append_simulations(theta, x)
    density_estimator = inference.train()
    posterior = inference.build_posterior(density_estimator)
    
    print("Creating dummy observed dataset...")
    # True parameters for observation
    theta_true = torch.tensor([4.0, 2.2])
    x_obs = simulator(theta_true)
    
    print("Sampling from posterior...")
    samples = posterior.sample((10000,), x=x_obs)
    
    print("Plotting posterior...")
    fig, axes = analysis.pairplot(
        samples,
        limits=[[1.0, 10.0], [1.5, 3.5]],
        ticks=[[1.0, 10.0], [1.5, 3.5]],
        fig_size=(6, 6),
        labels=[r"$\Phi_0$", r"$\Gamma$"],
        points=theta_true,
        points_colors=["red"],
        points_offdiag={"markersize": 6}
    )
    
    os.makedirs(os.path.join("results"), exist_ok=True)
    plot_path = os.path.join("results", "sbi_posterior.png")
    plt.savefig(plot_path)
    print(f"Posterior plot saved to {plot_path}")

if __name__ == "__main__":
    main()
