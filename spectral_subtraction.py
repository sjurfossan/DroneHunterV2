import numpy as np

class SpectralSubtractionDenoiser:
    def __init__(self, noise_threshold=0.01, alpha=1.0):
        """ Initializes the spectral subtraction denoiser. """
        self.noise_threshold = noise_threshold
        self.alpha = alpha
        self.noise_estimate = None

    def estimate_noise(self, spectrum):
        """ Estimates the noise floor based on an initial spectrum. """
        self.noise_estimate = np.mean(spectrum, axis=0) * self.noise_threshold

    def denoise(self, spectrum):
        """ Applies spectral subtraction to reduce noise. """
        if self.noise_estimate is None:
            self.estimate_noise(spectrum)

        # Subtract noise estimate from spectrum
        clean_spectrum = spectrum - self.alpha * self.noise_estimate
        clean_spectrum[clean_spectrum < 0] = 0  # Ensure no negative values
        return clean_spectrum
