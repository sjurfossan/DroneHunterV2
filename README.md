# 🛸 DroneHunterV2

**DroneHunterV2** is a Python-based system for detecting drone RF signals using a HackRF One and advanced signal processing techniques. Developed as part of a master's thesis, it implements multiple denoising and detection strategies to identify potential drone transmissions in noisy environments.
---

## ⚙️ Methods Implemented

DroneHunterV2 includes three detection pipelines using Constant False Alarm Rate (CA-CFAR) techniques:

### 1. Internal Noise Subtraction + CA-CFAR
- Records a baseline noise profile using HackRF One with a 50-ohm terminator.
- Subtracts this profile from operational RF sweeps to suppress consistent hardware noise.
- Applies CA-CFAR to detect drone signals in the cleaned spectrum.
- See: `sequence_diagram_baseline_dataset.png`

### 2. Wavelet Denoising (db4) + CA-CFAR
- Applies a Discrete Wavelet Transform (DWT) using the Daubechies-4 (db4) wavelet.
- Removes high-frequency noise while preserving spectral features.
- Runs CA-CFAR detection on the denoised data.
- See: `sequence_diagram_wavelet_dataset.png`

### 3. Combined Subtraction + Wavelet + CA-CFAR
- Combines both internal subtraction and wavelet denoising sequentially.
- Improves sensitivity by suppressing both persistent and transient noise.
- See: `sequence_diagram_baseline_wavelet_dataset.png`

---

## 🚀 Getting Started

### ✅ Requirements

- Python 3.8+
- Packages: `numpy`, `scipy`, `matplotlib`, `pywt`, `pandas`
- HackRF tools (`hackrf_sweep` or compatible interface)

### 📥 Installation

```bash
git clone https://github.com/sjurfossan/DroneHunterV2.git
cd DroneHunterV2
pip install -r requirements.txt  # (if requirements.txt exists)

