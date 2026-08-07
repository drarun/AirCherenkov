import torch
import numpy as np

class CameraTrigger:
    def __init__(self, camera, threshold_pe=5.0, window_ns=5.0, min_pixels=3):
        """
        Simulates a local coincidence Camera Level Trigger (CLT).
        
        Parameters
        ----------
        camera : sim.camera.Camera
            Camera object defining the geometry and adjacency.
        threshold_pe : float
            Discriminator threshold (in photoelectrons) each pixel must cross.
        window_ns : float
            Maximum arrival time difference between pixels in the coincidence cluster.
        min_pixels : int
            Number of adjacent pixels required to trigger (usually 3).
        """
        self.threshold = threshold_pe
        self.window = window_ns
        self.min_pixels = min_pixels
        self.camera = camera
        
        # Precompute adjacency matrix based on hexagonal pixel spacing
        self.adj_matrix = torch.tensor(self.camera.get_neighbor_matrix())
        
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
        else:
            # Fallback for generic N (not implemented for simplicity, usually N=3 is standard)
            pass
            
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
            if np.max(t) - np.min(t) <= self.window:
                # The local trigger logic fires the exact moment the LAST pixel crosses the threshold.
                # Or for normalization purposes, the mean arrival time of the coincidence.
                valid_t0s.append(np.mean(t))
                
        if len(valid_t0s) == 0:
            return False, None
            
        # The camera triggers when the FIRST valid local coincidence cluster fires
        return True, np.min(valid_t0s)
