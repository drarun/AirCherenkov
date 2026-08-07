import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection

class Camera:
    def __init__(self, n_rings=10, pixel_size=0.1):
        """
        Initialize a hexagonal IACT camera.
        
        Args:
            n_rings (int): Number of concentric hexagonal rings around the center pixel.
                           (e.g., 10 rings roughly gives ~331 pixels).
            pixel_size (float): Diameter (or flat-to-flat distance) of a pixel in degrees.
        """
        self.n_rings = n_rings
        self.pixel_size = pixel_size
        self.pixel_x, self.pixel_y = self._generate_hex_grid()
        self.n_pixels = len(self.pixel_x)
        self._edge_index = None
        
    def _generate_hex_grid(self):
        """Generates x and y coordinates for a hexagonal grid of pixels."""
        x_coords = []
        y_coords = []
        
        # Distance between adjacent pixel centers
        step = self.pixel_size
        # Hexagon height is sqrt(3)/2 * step if step is the distance between centers
        height = np.sqrt(3) / 2 * step 

        for q in range(-self.n_rings, self.n_rings + 1):
            for r in range(-self.n_rings, self.n_rings + 1):
                if -self.n_rings <= q + r <= self.n_rings:
                    x = step * (q + r / 2.0)
                    y = height * r
                    x_coords.append(x)
                    y_coords.append(y)
                    
        return np.array(x_coords), np.array(y_coords)

    def get_neighbor_matrix(self):
        """
        Computes the adjacency matrix for the camera pixels.
        Returns:
            np.ndarray: Boolean matrix of shape (n_pixels, n_pixels) where True means pixels are adjacent.
        """
        # Distance squared between all pairs of pixels
        dist_sq = (self.pixel_x[:, None] - self.pixel_x[None, :])**2 + (self.pixel_y[:, None] - self.pixel_y[None, :])**2
        # Hexagonal centers are exactly `self.pixel_size` apart
        # We use 1.1 as a tolerance margin due to floating point math
        is_neighbor = (dist_sq > 0) & (dist_sq < (self.pixel_size * 1.1)**2)
        return is_neighbor
        
    @property
    def edge_index(self):
        """Returns the PyTorch Geometric edge_index tensor for this camera."""
        if self._edge_index is None:
            adj = self.get_neighbor_matrix()
            self._edge_index = torch.tensor(adj).nonzero(as_tuple=False).t().contiguous()
        return self._edge_index

    def plot_image(self, image_amplitudes, ax=None, title="Camera Image", cmap="viridis"):
        """
        Plot an image on the hexagonal camera.
        
        Args:
            image_amplitudes (np.ndarray): 1D array of length n_pixels with pixel values (e.g., photoelectrons).
            ax (matplotlib.axes.Axes): Axis to plot on.
            title (str): Title of the plot.
            cmap (str): Matplotlib colormap.
        """
        if len(image_amplitudes) != self.n_pixels:
            raise ValueError(f"Expected {self.n_pixels} pixel values, got {len(image_amplitudes)}")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        patches = []
        # Radius from center to vertex of hexagon
        radius = (self.pixel_size / np.sqrt(3))

        for x, y in zip(self.pixel_x, self.pixel_y):
            # RegularPolygon takes center, numVertices, radius
            polygon = RegularPolygon((x, y), numVertices=6, radius=radius, orientation=np.radians(30))
            patches.append(polygon)

        p = PatchCollection(patches, cmap=cmap, alpha=0.9, edgecolor='black', linewidth=0.5)
        p.set_array(image_amplitudes)
        ax.add_collection(p)
        
        ax.set_aspect('equal')
        # Set limits based on the camera size
        limit = self.n_rings * self.pixel_size * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_xlabel("Camera X (degrees)")
        ax.set_ylabel("Camera Y (degrees)")
        ax.set_title(title)
        
        plt.colorbar(p, ax=ax, label="Signal (photoelectrons)")
        return ax

if __name__ == "__main__":
    # Test the camera with a dummy image (e.g. Night Sky Background noise)
    cam = Camera(n_rings=15, pixel_size=0.1) # VERITAS-like camera size
    
    # Generate random Poisson noise mimicking Night Sky Background (NSB)
    nsb_rate = 2.0 # mean photoelectrons per pixel
    noise_image = np.random.poisson(lam=nsb_rate, size=cam.n_pixels)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    cam.plot_image(noise_image, ax=ax, title=f"Empty Camera with NSB (Mean={nsb_rate} p.e.)")
    plt.show()
