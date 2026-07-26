import numpy as np
from dataclasses import dataclass
from typing import Optional, Any

@dataclass
class HillasParameters:
    size: float
    centroid_x: float
    centroid_y: float
    length: float
    width: float
    psi: float
    miss: float
    distance: float
    alpha: float
    asymmetry: float

def compute_hillas(camera: Any, image: np.ndarray, mask: np.ndarray) -> Optional[HillasParameters]:
    """
    Compute standard Hillas parameters for an IACT camera image.
    
    Args:
        camera: Camera object with pixel_x and pixel_y attributes (in degrees).
        image: Array of pixel signals (photoelectrons).
        mask: Boolean array indicating which pixels survive cleaning.
        
    Returns:
        HillasParameters object, or None if fewer than 3 pixels survive.
    """
    if np.sum(mask) < 3:
        return None
        
    x = camera.pixel_x[mask]
    y = camera.pixel_y[mask]
    s = image[mask]
    
    size = float(np.sum(s))
    if size <= 0:
        return None
        
    centroid_x = float(np.sum(s * x) / size)
    centroid_y = float(np.sum(s * y) / size)
    
    vx = x - centroid_x
    vy = y - centroid_y
    
    c_xx = np.sum(s * vx**2) / size
    c_yy = np.sum(s * vy**2) / size
    c_xy = np.sum(s * vx * vy) / size
    
    d = c_xx - c_yy
    z = np.sqrt(d**2 + 4 * c_xy**2)
    
    lambda1 = (c_xx + c_yy + z) / 2.0
    lambda2 = (c_xx + c_yy - z) / 2.0
    
    length = float(np.sqrt(max(lambda1, 0.0)))
    width = float(np.sqrt(max(lambda2, 0.0)))
    
    # Compute orientation angle psi
    if z == 0:
        psi = 0.0
    else:
        psi = 0.5 * np.arctan2(2.0 * c_xy, d)
        
    miss = float(abs(centroid_x * np.sin(psi) - centroid_y * np.cos(psi)))
    distance = float(np.sqrt(centroid_x**2 + centroid_y**2))
    
    if distance > 0:
        # Clamp argument to arcsin to [-1, 1] for numerical stability
        arg = np.clip(miss / distance, -1.0, 1.0)
        alpha = float(np.degrees(np.arcsin(arg)))
    else:
        alpha = 0.0
        
    # Ensure alpha is in [0, 90]
    alpha = min(abs(alpha), 90.0)
    
    # Orient psi so that it points roughly away from the camera center
    # i.e. dot product with centroid vector is positive
    dir_x = np.cos(psi)
    dir_y = np.sin(psi)
    if dir_x * centroid_x + dir_y * centroid_y < 0:
        dir_x = -dir_x
        dir_y = -dir_y
        
    l = vx * dir_x + vy * dir_y
    asymmetry = float(np.sum(s * l**3) / size)
    
    return HillasParameters(
        size=size,
        centroid_x=centroid_x,
        centroid_y=centroid_y,
        length=length,
        width=width,
        psi=psi,
        miss=miss,
        distance=distance,
        alpha=alpha,
        asymmetry=asymmetry
    )

def print_hillas(params: HillasParameters) -> None:
    """
    Print Hillas parameters in a formatted table.
    """
    if params is None:
        print("HillasParameters: None")
        return
        
    print("-" * 40)
    print("Hillas Parameters")
    print("-" * 40)
    print(f"{'Size (pe)':<20} : {params.size:.2f}")
    print(f"{'Centroid X (deg)':<20} : {params.centroid_x:.4f}")
    print(f"{'Centroid Y (deg)':<20} : {params.centroid_y:.4f}")
    print(f"{'Length (deg)':<20} : {params.length:.4f}")
    print(f"{'Width (deg)':<20} : {params.width:.4f}")
    print(f"{'Psi (rad)':<20} : {params.psi:.4f}")
    print(f"{'Miss (deg)':<20} : {params.miss:.4f}")
    print(f"{'Distance (deg)':<20} : {params.distance:.4f}")
    print(f"{'Alpha (deg)':<20} : {params.alpha:.4f}")
    print(f"{'Asymmetry':<20} : {params.asymmetry:.4e}")
    print("-" * 40)
