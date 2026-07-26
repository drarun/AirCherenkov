import numpy as np
from sim.camera import Camera
from sim.backend import ray_trace_gpu, device_info

class Telescope:
    def __init__(self, x_tel=0.0, y_tel=0.0, mirror_radius=6.0, focal_length=15.0, 
                 mirror_reflectivity=0.82, quantum_efficiency=0.20, pedestal_std=0.5,
                 n_rings=15, pixel_size=0.1):
        self.x_tel = x_tel
        self.y_tel = y_tel
        self.mirror_radius = mirror_radius
        self.focal_length = focal_length
        self.mirror_reflectivity = mirror_reflectivity
        self.quantum_efficiency = quantum_efficiency
        self.pedestal_std = pedestal_std
        
        self.camera = Camera(n_rings=n_rings, pixel_size=pixel_size)
        
    def ray_trace(self, cherenkov_photons, nsb_rate=2.0):
        """
        Ray-trace Cherenkov photons through the telescope optics onto the camera.
        
        Uses GPU-accelerated pixel lookup (torch.cdist) when CUDA is available,
        falling back to scipy KDTree on CPU.
        """
        # Start with NSB (Night Sky Background)
        image = np.random.poisson(lam=nsb_rate, size=self.camera.n_pixels).astype(float)
        
        if len(cherenkov_photons['x_ground']) == 0:
            image += np.random.normal(scale=self.pedestal_std, size=image.shape)
            return image
        
        # Dispatch signal computation to GPU/CPU backend
        signal = ray_trace_gpu(
            cherenkov_photons,
            self.camera.pixel_x, self.camera.pixel_y, self.camera.pixel_size,
            self.x_tel, self.y_tel, self.mirror_radius,
            self.mirror_reflectivity, self.quantum_efficiency
        )
        
        image += signal
        
        # Add electronic pedestal noise
        image += np.random.normal(scale=self.pedestal_std, size=image.shape)
        
        return image

class VeritasTelescope(Telescope):
    def __init__(self, x_tel=0.0, y_tel=0.0):
        super().__init__(x_tel=x_tel, y_tel=y_tel, mirror_radius=6.0, focal_length=12.0, n_rings=12, pixel_size=0.1)

class HessTelescope(Telescope):
    def __init__(self, x_tel=0.0, y_tel=0.0):
        super().__init__(x_tel=x_tel, y_tel=y_tel, mirror_radius=6.0, focal_length=15.0, n_rings=16, pixel_size=0.1)

class CtaLST(Telescope):
    def __init__(self, x_tel=0.0, y_tel=0.0):
        super().__init__(x_tel=x_tel, y_tel=y_tel, mirror_radius=11.5, focal_length=28.0, n_rings=20, pixel_size=0.1)

class CtaMST(Telescope):
    def __init__(self, x_tel=0.0, y_tel=0.0):
        super().__init__(x_tel=x_tel, y_tel=y_tel, mirror_radius=6.0, focal_length=16.0, n_rings=15, pixel_size=0.1)

class CtaSST(Telescope):
    def __init__(self, x_tel=0.0, y_tel=0.0):
        super().__init__(x_tel=x_tel, y_tel=y_tel, mirror_radius=2.0, focal_length=2.2, n_rings=10, pixel_size=0.1)

class TelescopeArray:
    def __init__(self, telescopes):
        self.telescopes = telescopes

    @staticmethod
    def veritas_array():
        return TelescopeArray([
            VeritasTelescope(x_tel=0.0, y_tel=0.0),
            VeritasTelescope(x_tel=100.0, y_tel=0.0),
            VeritasTelescope(x_tel=0.0, y_tel=100.0),
            VeritasTelescope(x_tel=100.0, y_tel=100.0)
        ])

    def ray_trace(self, cherenkov_photons, nsb_rate=2.0):
        results = []
        for tel in self.telescopes:
            results.append(tel.ray_trace(cherenkov_photons, nsb_rate=nsb_rate))
        return results
