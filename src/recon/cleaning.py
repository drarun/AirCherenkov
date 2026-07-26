import numpy as np

def tail_cut_clean(camera, image, picture_thresh=5.0, boundary_thresh=2.5, min_neighbors=1):
    """
    Standard Two-Level Tail-Cut Image Cleaning.
    
    Args:
        camera (Camera): The camera object containing pixel geometries.
        image (np.ndarray): The raw pixel amplitudes (photoelectrons).
        picture_thresh (float): Threshold for core shower pixels.
        boundary_thresh (float): Threshold for fringe pixels adjacent to a picture pixel.
        min_neighbors (int): Minimum number of adjacent picture pixels required for core pixels.
        
    Returns:
        np.ndarray: A boolean mask of the same length as the image, where True means the pixel is retained.
    """
    neighbors = camera.get_neighbor_matrix()
    
    # 1. Identify picture pixels
    # Must be > picture_thresh AND have at least `min_neighbors` adjacent pixels also > picture_thresh
    is_above_pic = image >= picture_thresh
    
    # Count how many neighbors are also above picture threshold
    pic_neighbor_counts = np.sum(neighbors & is_above_pic[None, :], axis=1)
    
    picture_mask = is_above_pic & (pic_neighbor_counts >= min_neighbors)
    
    # 2. Identify boundary pixels
    # Must be > boundary_thresh AND adjacent to at least one picture pixel
    is_above_bnd = image >= boundary_thresh
    
    # Check if any neighbor is a valid picture pixel
    adjacent_to_pic = np.any(neighbors & picture_mask[None, :], axis=1)
    
    boundary_mask = is_above_bnd & adjacent_to_pic & (~picture_mask)
    
    # Final cleaned mask
    return picture_mask | boundary_mask


def double_pass_clean(camera, image, 
                      pic1=5.0, bnd1=2.5, 
                      pic2=2.0, bnd2=1.0, 
                      dist_tolerance=0.15):
    """
    Double-Pass Image Cleaning.
    
    Args:
        camera (Camera): The camera object.
        image (np.ndarray): The raw pixel amplitudes.
        pic1, bnd1: Thresholds for the first pass (standard cleaning).
        pic2, bnd2: Lower thresholds for the second pass to recover faint fringes.
        dist_tolerance: Maximum perpendicular distance (in degrees) from the shower axis to accept 2nd pass pixels.
        
    Returns:
        np.ndarray: A boolean mask of retained pixels after both passes.
    """
    # PASS 1: Standard tight cleaning to find the core
    mask_pass1 = tail_cut_clean(camera, image, pic1, bnd1)
    
    if np.sum(mask_pass1) < 3:
        # If the first pass failed to find a shower core, return the empty mask
        return mask_pass1
        
    # Calculate the principal axis (major axis) of the Pass 1 image
    x_core = camera.pixel_x[mask_pass1]
    y_core = camera.pixel_y[mask_pass1]
    w_core = image[mask_pass1]
    
    # Center of gravity
    sum_w = np.sum(w_core)
    cg_x = np.sum(x_core * w_core) / sum_w
    cg_y = np.sum(y_core * w_core) / sum_w
    
    # Covariance matrix for the core pixels
    dx = x_core - cg_x
    dy = y_core - cg_y
    Sxx = np.sum(w_core * dx**2) / sum_w
    Syy = np.sum(w_core * dy**2) / sum_w
    Sxy = np.sum(w_core * dx * dy) / sum_w
    
    # Angle of the major axis (delta)
    # tan(2 * delta) = 2*Sxy / (Sxx - Syy)
    diff = Sxx - Syy
    if diff == 0: diff = 1e-10 # prevent division by zero
    delta = 0.5 * np.arctan2(2 * Sxy, diff)
    
    # Unit vector of the major axis
    vx = np.cos(delta)
    vy = np.sin(delta)
    
    # PASS 2: Lower threshold cleaning
    mask_pass2_candidates = tail_cut_clean(camera, image, pic2, bnd2)
    
    # Filter Pass 2 candidates: they must be close to the Pass 1 major axis
    # Perpendicular distance from a point (x, y) to the line passing through (cg_x, cg_y) with direction (vx, vy):
    # d = |(x - cg_x)*vy - (y - cg_y)*vx|
    all_dx = camera.pixel_x - cg_x
    all_dy = camera.pixel_y - cg_y
    perp_dist = np.abs(all_dx * vy - all_dy * vx)
    
    valid_pass2_mask = mask_pass2_candidates & (perp_dist <= dist_tolerance)
    
    # Final mask is Pass 1 combined with the valid Pass 2 pixels
    return mask_pass1 | valid_pass2_mask
