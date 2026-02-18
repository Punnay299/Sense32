import numpy as np
import scipy.signal

class CSISanitizer:
    """
    Sanitization utility for ESP32 CSI Data.
    Implements:
    1. Hampel Filter (Amplitude Outlier Removal)
    2. Phase Unwrapping & Linear Detrending (Phase Sanitization)
    """

    @staticmethod
    def sanitize_amplitude(amp_data, window_size=5, n_sigmas=3):
        """
        Applies Hampel Filter to remove outliers from amplitude data.
        
        Args:
            amp_data (np.array): Shape (Seq, Subcarriers) or (Subcarriers,)
            window_size (int): Sliding window size.
            n_sigmas (float): Threshold for outlier detection.
            
        Returns:
            np.array: Cleaned amplitude data.
        """
        # If 1D, make it 2D for consistent processing (1, Subcarriers)
        is_1d = False
        if amp_data.ndim == 1:
            amp_data = amp_data.reshape(1, -1)
            is_1d = True
            
        # Iterate over subcarriers (columns) if sequence; 
        # BUT Hampel is usually temporal (over time for specific subcarrier).
        # OR is it spectral (over subcarriers for specific packet)?
        # For CSI, spikes occur in both dimensions. 
        # Let's assume we filter TEMPORALLY (Seq dimension) for each subcarrier.
        
        # If Input is (Seq, Subcarriers)
        clean_data = amp_data.copy()
        
        for sc in range(amp_data.shape[1]):
            series = amp_data[:, sc]
            if len(series) < window_size:
                continue
                
            # Rolling Median
            new_series = series.copy()
            k = 1.4826 # Scale factor for Gaussian distribution
            
            rolling_median = scipy.signal.medfilt(series, kernel_size=window_size)
            
            # Estimate standard deviation (MAD)
            # This is a simplifed Hampel. For strict sliding window MAD:
            # We can use pandas but unwanted dependency. 
            # Let's stick to a simple diff check vs global median of window.
            
            # Vectorized approach for speed:
            difference = np.abs(series - rolling_median)
            median_abs_deviation = scipy.signal.medfilt(difference, kernel_size=window_size)
            
            threshold = n_sigmas * k * median_abs_deviation
            
            outlier_idx = difference > threshold
            new_series[outlier_idx] = rolling_median[outlier_idx]
            
            clean_data[:, sc] = new_series
            
        if is_1d:
            return clean_data.flatten()
        return clean_data

    @staticmethod
    def sanitize_phase(phase_data):
        """
        Sanitizes Phase data by:
        1. Unwrapping (Unwrap 2pi jumps)
        2. Linear Fit Removal (Remove Timing Offsets/SFO)
        
        Args:
            phase_data (np.array): Raw Phase (Seq, Subcarriers) in Radians.
            
        Returns:
            np.array: Cleaned Phase.
        """
        # Check input range. ESP32 often gives raw int8/int16. 
        # Assuming input is ALREADY converted to radians or raw values.
        # If raw ~ +/- 3.14, it is radians.
        
        is_1d = False
        if phase_data.ndim == 1:
            phase_data = phase_data.reshape(1, -1)
            is_1d = True
            
        clean_phase = np.zeros_like(phase_data)
        subcarrier_indices = np.arange(phase_data.shape[1])
        
        # Process each PACKET (Time step) individually
        for t in range(phase_data.shape[0]):
            pkt_phase = phase_data[t, :]
            
            # 1. Unwrap Phase (Remove 2pi jumps)
            unwrapped = np.unwrap(pkt_phase)
            
            # 2. Linear Detrending (Remove Time Lag / Sampling Frequency Offset)
            # We fit a line to the unwrapped phase: y = mx + c
            # The slope 'm' corresponds to the time delay.
            # The intercept 'c' is the path constant.
            # By removing this linear trend, we keep the non-linear variations due to multipath/human body.
            
            slope, intercept = np.polyfit(subcarrier_indices, unwrapped, 1)
            linear_trend = slope * subcarrier_indices + intercept
            
            sanitized = unwrapped - linear_trend
            
            # 3. Z-Score / Normalization (Optional but good for Neural Net)
            # Phase range can be arbitrary after detrending. 
            # Normalizing to [-pi, pi] isn't strictly necessary for detrended phase,
            # but usually we want it zero-centered.
            
            clean_phase[t, :] = sanitized
            
        if is_1d:
            return clean_phase.flatten()
        return clean_phase
