import numpy as np

class FADC:
    def __init__(self, sample_rate_mhz=500, window_size=20, 
                 fadc_resolution_bits=10, adc_per_pe=4.0):
        """
        Simulate FADC electronics and PMT pulse shaping.
        
        Parameters
        ----------
        sample_rate_mhz : float
            FADC sampling rate in MHz.
        window_size : int
            Number of FADC samples in the readout window.
        fadc_resolution_bits : int
            Number of bits for the FADC, determines maximum ADC count.
        adc_per_pe : float
            Conversion factor from photoelectrons to ADC counts.
        """
        self.sample_rate_mhz = sample_rate_mhz
        self.window_size = window_size
        self.fadc_max = (1 << fadc_resolution_bits) - 1
        self.adc_per_pe = adc_per_pe
        
        # Time array for the pulse shape
        dt = 1000.0 / self.sample_rate_mhz  # ns per sample
        t = np.arange(self.window_size) * dt
        t_peak = self.window_size * dt / 2.0
        sigma = 2.5  # Typical PMT pulse width ~2.5 ns sigma
        
        # Gaussian PMT pulse shape
        self.pulse_shape = np.exp(-0.5 * ((t - t_peak) / sigma)**2)
        # Normalize so that the sum is 1.0 (charge conservation)
        self.pulse_shape /= np.sum(self.pulse_shape)

    def digitize_image(self, cherenkov_pe, nsb_rate, pedestal_std):
        """
        Apply NSB Poisson noise, PMT pulse shaping, electronic noise, 
        and FADC digitization to a Cherenkov photoelectron image.
        
        Parameters
        ----------
        cherenkov_pe : array-like
            Cherenkov photoelectron counts per pixel.
        nsb_rate : float
            Night Sky Background photoelectrons per pixel in the readout window.
        pedestal_std : float
            Electronic pedestal standard deviation (in ADC counts) for the integrated image.
            
        Returns
        -------
        integrated_adc : array-like
            The integrated image in ADC counts (baseline subtracted).
        """
        n_pixels = cherenkov_pe.shape[0]
        
        # 1. Add NSB Poisson noise
        nsb_pe = np.random.poisson(lam=nsb_rate, size=n_pixels)
        total_pe = cherenkov_pe + nsb_pe
        
        # 2. PMT pulse shaping: convert total PE into waveforms
        true_waveforms = np.outer(total_pe, self.pulse_shape)
        
        # Convert PE waveforms to ADC counts
        adc_waveforms = true_waveforms * self.adc_per_pe
        
        # 3. Add electronic noise and baseline
        # Calculate per-sample noise such that the sum of the window has std = pedestal_std
        sample_noise_std = pedestal_std / np.sqrt(self.window_size)
        noise = np.random.normal(scale=sample_noise_std, size=adc_waveforms.shape)
        
        baseline = 50.0  # ADC counts per sample
        adc_waveforms += noise + baseline
        
        # 4. FADC digitization (quantization and saturation)
        adc_waveforms = np.clip(np.round(adc_waveforms), 0, self.fadc_max)
        
        # Return integrated ADC image, subtracting baseline to keep pedestal near 0
        integrated_adc = np.sum(adc_waveforms, axis=1) - (baseline * self.window_size)
        
        return integrated_adc
