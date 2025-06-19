import numpy as np
import pywt

class WaveletDenoiser:
    """Wavelet transforms on real-time spectrum data"""
    def __init__(self, wavelet='db4', level=3):
        self.wavelet = wavelet
        self.level = level

    def denoise(self, data):
        coeffs = pywt.wavedec(data, self.wavelet, level=self.level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745  # Estimate noise level
        threshold = sigma * np.sqrt(2 * np.log(len(data)))
        coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
        return pywt.waverec(coeffs, self.wavelet)[:len(data)]