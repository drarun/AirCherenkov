import torch
import numpy as np

class CameraTrigger:
    def __init__(self, pixel_x, pixel_y, threshold_pe=5.0, window_ns=5.0,
                 min_pixels=3, pixel_size=None):
        """
        Simulates a local coincidence Camera Level Trigger (CLT).
        
        Parameters
        ----------
        pixel_x, pixel_y : array-like
            Coordinates of the camera pixels.
        threshold_pe : float
            Discriminator threshold (in photoelectrons) each pixel must cross.
        window_ns : float
            Maximum arrival time difference between pixels in the coincidence cluster.
        min_pixels : int
            Number of adjacent pixels required to trigger (usually 3).
        """
        pixel_x = np.asarray(pixel_x, dtype=np.float64)
        pixel_y = np.asarray(pixel_y, dtype=np.float64)
        if not isinstance(min_pixels, (int, np.integer)) or isinstance(min_pixels, bool):
            raise TypeError("min_pixels must be an integer")
        if min_pixels != 3:
            raise ValueError(
                "Only compact three-pixel coincidence clusters are currently supported; "
                f"got min_pixels={min_pixels}"
            )
        if pixel_x.ndim != 1 or pixel_y.ndim != 1 or pixel_x.shape != pixel_y.shape:
            raise ValueError("pixel_x and pixel_y must be equally sized one-dimensional arrays")
        if len(pixel_x) < min_pixels:
            raise ValueError("camera geometry has fewer pixels than min_pixels")
        if not np.all(np.isfinite(pixel_x)) or not np.all(np.isfinite(pixel_y)):
            raise ValueError("camera pixel coordinates must be finite")
        if not np.isfinite(threshold_pe) or threshold_pe < 0:
            raise ValueError("threshold_pe must be finite and non-negative")
        if not np.isfinite(window_ns) or window_ns < 0:
            raise ValueError("window_ns must be finite and non-negative")
        self.threshold = float(threshold_pe)
        self.window = float(window_ns)
        self.min_pixels = int(min_pixels)

        # Infer the nearest-neighbour separation from the actual geometry unless
        # the camera supplies it explicitly. This avoids coupling the trigger to
        # the 0.1-degree spacing of one particular camera model.
        pos = np.stack([pixel_x, pixel_y], axis=1)
        delta = pos[:, None, :] - pos[None, :, :]
        distances = np.sqrt(np.sum(delta * delta, axis=2))
        np.fill_diagonal(distances, np.inf)

        if pixel_size is None:
            nearest = np.min(distances, axis=1)
            finite_nearest = nearest[np.isfinite(nearest) & (nearest > 0)]
            if finite_nearest.size == 0:
                raise ValueError("cannot infer pixel spacing from the supplied geometry")
            spacing = float(np.median(finite_nearest))
        else:
            spacing = float(pixel_size)
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError("pixel_size must be finite and greater than zero")

        tolerance = max(spacing * 0.05, np.finfo(np.float64).eps * 32)
        adjacency = np.abs(distances - spacing) <= tolerance
        self.pixel_size = spacing
        self.adj_matrix = torch.from_numpy(adjacency)
        
        # Precompute all valid "3-pixel compact clusters" (triangles of mutually adjacent pixels)
        self.clusters = self._find_trigger_clusters()
        
    def _find_trigger_clusters(self):
        """
        Find all groups of `min_pixels` that are mutually adjacent (e.g. triangles for min_pixels=3).
        """
        clusters = []
        n_pixels = self.adj_matrix.shape[0]
        adj = self.adj_matrix.numpy()
        
        if self.min_pixels == 3:
            for i in range(n_pixels):
                # Find all neighbors of i
                neighbors = np.where(adj[i])[0]
                for j in neighbors:
                    if j > i:
                        # Find common neighbors of i and j
                        common = np.where(adj[i] & adj[j])[0]
                        for k in common:
                            if k > j:
                                clusters.append([i, j, k])
        return np.array(clusters, dtype=np.int32)
        
    def evaluate(self, image, timing):
        """
        Evaluate if the camera triggers on the given event.
        
        Returns
        -------
        triggered : bool
            True if the camera triggered.
        t0 : float or None
            The precise nanosecond timestamp the trigger fired (average time of the first cluster).
        """
        image = self._as_numpy(image)
        timing = self._as_numpy(timing)
        if image.ndim != 1 or timing.ndim != 1:
            raise ValueError("image and timing must be one-dimensional arrays")
        if image.shape != timing.shape or image.shape[0] != self.adj_matrix.shape[0]:
            raise ValueError("image and timing must contain one value per camera pixel")

        # Step 1: Pixel discriminator check
        over_thresh = image > self.threshold
        
        # Step 2: Check each cluster
        if len(self.clusters) == 0:
            return False, None
            
        cluster_charges = over_thresh[self.clusters]
        # Cluster is active if ALL pixels in the cluster are over threshold
        active_clusters_mask = np.all(cluster_charges, axis=1)
        
        active_clusters = self.clusters[active_clusters_mask]
        
        if len(active_clusters) == 0:
            return False, None
            
        # Step 3: Check sliding time window coincidence
        valid_t0s = []
        for cluster in active_clusters:
            t = timing[cluster]
            if np.all(np.isfinite(t)) and np.max(t) - np.min(t) <= self.window:
                # The local trigger logic fires the exact moment the LAST pixel crosses the threshold.
                # Or for normalization purposes, the mean arrival time of the coincidence.
                valid_t0s.append(np.mean(t))
                
        if len(valid_t0s) == 0:
            return False, None
            
        # The camera triggers when the FIRST valid local coincidence cluster fires
        return True, np.min(valid_t0s)

    def evaluate_traces(self, traces, bin_width_ns=2.0):
        """Evaluate discriminator crossings directly from pixel waveforms.

        The threshold is applied to per-sample pulse amplitude rather than to
        charge integrated over the complete readout window. The crossing time
        is the first sample at which each pixel exceeds the discriminator.
        """
        traces = self._as_numpy(traces)
        if traces.ndim != 2 or traces.shape[0] != self.adj_matrix.shape[0]:
            raise ValueError(
                "traces must have shape (n_camera_pixels, n_time_bins)"
            )
        if not np.all(np.isfinite(traces)):
            raise ValueError("traces must contain only finite values")
        if not np.isfinite(bin_width_ns) or bin_width_ns <= 0:
            raise ValueError("bin_width_ns must be finite and greater than zero")

        over_threshold = traces > self.threshold
        peak_amplitude = np.max(traces, axis=1)
        first_crossing_bin = np.argmax(over_threshold, axis=1)
        crossing_time = first_crossing_bin.astype(np.float64) * float(bin_width_ns)
        crossing_time[~np.any(over_threshold, axis=1)] = np.nan
        return self.evaluate(peak_amplitude, crossing_time)

    @staticmethod
    def _as_numpy(values):
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        return np.asarray(values)
