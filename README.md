# 🛸 DroneHunterV2

**DroneHunterV2** is a Python-based system for detecting drone RF signals using a HackRF One and signal processing techniques. Developed as part of a master's thesis, it implements multiple denoising and detection strategies to identify potential drone transmissions in noisy environments.
---

## ⚙️ For Evaluation on a Prerecorded Dataset:

For the development og DroneHunterV2, four strateies was evaluated:

### 1. Internal Noise Subtraction + CA-CFAR (internal_noise_reduction_cfar_dataset.py)
- Uses a baseline noise profile if the HackRF One.
- Subtracts this profile from operational RF sweeps to suppress consistent hardware noise.
- Applies CA-CFAR to detect drone signals in the cleaned spectrum.
- See: `sequence_diagram_baseline_dataset.png`

### 2. Wavelet Denoising (db4) + CA-CFAR (wavelet_cfar_dataset.py)
- Applies a Discrete Wavelet Transform (DWT) using the Daubechies-4 (db4) wavelet.
- Removes high-frequency noise while preserving spectral features.
- Runs CA-CFAR detection on the denoised data.
- See: `sequence_diagram_wavelet_dataset.png`

### 3. Combined Subtraction + Wavelet + CA-CFAR (dronehunterV2_final_software_dataset.py)
- Combines both internal subtraction and wavelet denoising sequentially.
- Improves sensitivity by suppressing both persistent and transient noise.
- See: `sequence_diagram_baseline_wavelet_dataset.png`

### 4. CA-CFAR with no other filter applied (cfar_alone_dataset.py)
- CFAR alone with no other filter applied.
  
---
## ⚙️ For the Real-Time Applications (Requires a HackRF One):

For real-time applications:

### 1. Internal Noise Subtraction + CA-CFAR (internal_noise_reduction_cfar_real_time.py)
- Uses a baseline noise profile if the HackRF One.
- Subtracts this profile from operational RF sweeps to suppress consistent hardware noise.
- Applies CA-CFAR to detect drone signals in the cleaned spectrum.
- See: `sequence_diagram_baseline.png`

### 2. Wavelet Denoising (db4) + CA-CFAR (wavelet_cfar_dataset_real_time.py)
- Applies a Discrete Wavelet Transform (DWT) using the Daubechies-4 (db4) wavelet.
- Removes high-frequency noise while preserving spectral features.
- Runs CA-CFAR detection on the denoised data.
- See: `sequence_diagram_wavelet.png`

### 3. Combined Subtraction + Wavelet + CA-CFAR (dronehunterV2_final_software_dataset_real_time.py)
- Combines both internal subtraction and wavelet denoising sequentially.
- Improves sensitivity by suppressing both persistent and transient noise.
- See: `sequence_diagram_baseline_wavelet.png`

### 4. CA-CFAR with no other filter applied (cfar_alone_real_time.py)
- CFAR alone with no other filter applied.
---

## ✅ Requirements

- HackRF One
- Python 3.8+
- Packages: `numpy`, `scipy`, `matplotlib`, `pywt`, `pandas`
