import os
import numpy as np
import subprocess
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QVBoxLayout, QMainWindow, QLineEdit, QLabel
import atexit
from pyqtgraph import ColorMap
from cfar import CFARProcessor

# ---------------------------------------------------------------------------
# 🔧  Spectrum‑hygiene utilities (spur mask + optional DC notch)
# ---------------------------------------------------------------------------

class SpurMaskCleaner:
    """Subtract a once‑measured spur mask and (optionally) apply a narrow
    median filter along frequency.  Falls back gracefully if the mask does
    not exist so the script always runs.
    """

    def __init__(self, total_bins: int, mask_path: str | None = "spur_mask.npy", median_kernel: int | None = 31, dc_bins: int = 2):
        self.total_bins = total_bins
        self.median_kernel = median_kernel if median_kernel and median_kernel % 2 == 1 else None
        self.dc_bins = dc_bins
        self.mask = np.zeros(total_bins, dtype=np.float32)

        if mask_path and os.path.isfile(mask_path):
            try:
                data = np.load(mask_path)
                if data.shape[0] == total_bins:
                    self.mask = data.astype(np.float32)
                    print(f"[INFO] Spur mask loaded from '{mask_path}'.")
                else:
                    print(f"[WARN] Spur mask length mismatch (found {data.shape[0]}, need {total_bins}). Ignoring.")
            except Exception as e:
                print(f"[WARN] Could not load spur mask ({e}). Ignoring.")

    # ──────────────────────────────────────────────────────────────────
    def _median_filter(self, arr: np.ndarray) -> np.ndarray:
        if self.median_kernel is None:
            return arr
        k = self.median_kernel
        pad = k // 2
        padded = np.pad(arr, pad, mode="edge")
        out = np.empty_like(arr)
        for i in range(arr.size):
            out[i] = np.median(padded[i : i + k])
        return out

    # ──────────────────────────────────────────────────────────────────
    def clean(self, psd: np.ndarray) -> np.ndarray:
        psd = psd.astype(np.float32) - self.mask  # subtract fixed spurs

        # notch centre DC spike unconditionally
        centre = psd.size // 2
        psd[centre - self.dc_bins : centre + self.dc_bins + 1] = -120.0

        if self.median_kernel:
            psd -= self._median_filter(psd)  # flatten residual bumps
        return psd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FREQ_BINS_PER_LINE = 5
WATERFALL_DEPTH = 100

# blue → yellow → red colour map
colors = [
    (0.00, (0, 0, 225)),
    (0.95, (225, 225, 0)),
    (0.97, (225, 0, 0)),
    (1.00, (225, 0, 0)),
]
color_map = pg.ColorMap(*zip(*colors))
lut = color_map.getLookupTable(start=0.0, stop=1.0, nPts=256)

# ---------------------------------------------------------------------------
# Helper classes (FrequencyManager, HackRFController)
# ---------------------------------------------------------------------------

class FrequencyManager:
    """Gets the next sweep span in MHz."""

    def __init__(self):
        self.freq_dict = {
            1: [300, 600],
            2: [900, 1500],
            3: [2000, 3000],
            4: [4500, 5500],
            5: [0, 6000],
        }
        self.counter = 1

    def next_range(self):
        self.counter += 1
        if self.counter > len(self.freq_dict):
            self.counter = 1
        return self.freq_dict[self.counter]


class HackRFController:
    """Minimal wrapper around *hackrf_sweep*."""

    def __init__(self):
        self.lna_gain = 32
        self.gain = 40
        self.process = None

    def start(self, freq_low: int, freq_high: int):
        self.process = subprocess.Popen(
            [
                "hackrf_sweep",
                "-f",
                f"{freq_low}:{freq_high}",
                "-l",
                str(self.lna_gain),
                "-g",
                str(self.gain),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return self.process

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def restart(self, freq_low: int, freq_high: int):
        self.stop()
        self.start(freq_low, freq_high)

    def process_line(self, line: str):
        parts = line.strip().split(", ")
        try:
            vals = list(map(float, parts[-FREQ_BINS_PER_LINE:]))
            return vals if len(vals) == FREQ_BINS_PER_LINE else []
        except ValueError:
            return []

# ---------------------------------------------------------------------------
# Plot widgets (SpectrumPlot, WaterfallPlotWidget)
# ---------------------------------------------------------------------------

class SpectrumPlot:
    def __init__(self, layout, freq_low, freq_high, total_bins):
        self.plot = layout.addPlot(title="Real‑Time Spectrum with Max‑Hold")
        self.plot.setLabel("bottom", "Frequency (Hz)")
        self.plot.setLabel("left", "Power (dB)")
        self.total_bins = total_bins
        self.plot.setXRange(freq_low * 1e6, freq_high * 1e6)
        self.plot.setYRange(-100, 0)
        self.rt_curve = self.plot.plot(pen="y")
        self.max_curve = self.plot.plot(pen="r")

    def update(self, freqs, powers, max_hold):
        if powers.size != self.total_bins:
            return max_hold
        if max_hold is None or max_hold.size != self.total_bins:
            max_hold = np.full(self.total_bins, -100, dtype=np.float32)
        max_hold = np.maximum(max_hold, powers)
        self.rt_curve.setData(freqs, powers)
        self.max_curve.setData(freqs, max_hold)
        return max_hold


class WaterfallPlotWidget:
    """Scrolling waterfall that can be resized when the span changes."""

    def __init__(self, layout, freq_low, freq_high, total_bins):
        self.plot = layout.addPlot()
        self.plot.setLabel("bottom", "Frequency (Hz)")
        self.plot.setLabel("left", "Time")
        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.plot.invertY(True)
        self.img.setTransform(QtGui.QTransform().scale(1e6, 1))

        self.resize(freq_low, freq_high, total_bins)

    # ------------------------------------------------------------------
    def resize(self, freq_low: int, freq_high: int, total_bins: int):
        """Re‑initialise buffer & axes when sweep span changes."""
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.total_bins = total_bins
        self.waterfall = np.full((WATERFALL_DEPTH, total_bins), -100, dtype=np.float32)
        self.plot.setXRange(freq_low * 1e6, freq_high * 1e6)
        self.plot.setYRange(0, WATERFALL_DEPTH)

    def update(self, new_row: np.ndarray):
        if new_row.size != self.total_bins:
            return  # ignore mismatched frame
        self.waterfall[1:, :] = self.waterfall[:-1, :]
        self.waterfall[0, :] = new_row
        mn, mx = self.waterfall.min(), self.waterfall.max()
        if mx - mn < 1e-3:
            scaled = np.zeros_like(self.waterfall)
        else:
            scaled = (self.waterfall - mn) / (mx - mn)
        self.img.setImage(scaled.T, autoLevels=False, lut=lut, levels=(0, 1))
        self.img.setPos(self.freq_low * 1e6, 0)

# ---------------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("HackRF Spectrum Sweep")
        self.resize(1100, 820)

        # ── RF plumbing ───────────────────────────────────────────────
        self.freq_manager = FrequencyManager()
        self.freq_low, self.freq_high = self.freq_manager.freq_dict[self.freq_manager.counter]
        self.total_bins = self.freq_high - self.freq_low
        self.frequencies = np.linspace(self.freq_low * 1e6, self.freq_high * 1e6, self.total_bins)

        self.hackrf = HackRFController()
        self.cleaner = SpurMaskCleaner(self.total_bins)
        self.cfar = CFARProcessor()
        self.max_hold = np.full(self.total_bins, -100, dtype=np.float32)

        # ── GUI layout ────────────────────────────────────────────────
        self.layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.layout)
        self.spectrum_plot = SpectrumPlot(self.layout, self.freq_low, self.freq_high, self.total_bins)
        self.layout.nextRow()
        self.waterfall_plot = WaterfallPlotWidget(self.layout, self.freq_low, self.freq_high, self.total_bins)

        self.init_ui()
        self.start_hackrf()

    # ------------------------------------------------------------------
    def init_ui(self):
        widget = QWidget()
        vbox = QVBoxLayout()

        vbox.addWidget(QLabel("VGA Gain:"))
        self.gain_in = QLineEdit(str(self.hackrf.gain))
        vbox.addWidget(self.gain_in)
        g_btn = QPushButton("Apply Gain")
        g_btn.clicked.connect(self.apply_gain)
        vbox.addWidget(g_btn)

        vbox.addWidget(QLabel("LNA Gain:"))
        self.lna_in = QLineEdit(str(self.hackrf.lna_gain))
        vbox.addWidget(self.lna_in)
        l_btn = QPushButton("Apply LNA Gain")
        l_btn.clicked.connect(self.apply_lna_gain)
        vbox.addWidget(l_btn)

        nxt_btn = QPushButton("Next Frequency Range")
        nxt_btn.clicked.connect(self.set_frequency_range)
        vbox.addWidget(nxt_btn)

        rst_btn = QPushButton("Reset Max Hold")
        rst_btn.clicked.connect(self.reset_max_hold)
        vbox.addWidget(rst_btn)

        widget.setLayout(vbox)
        self.setMenuWidget(widget)

    # ------------------------------------------------------------------
    def start_hackrf(self):
        self.hackrf.start(self.freq_low, self.freq_high)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.read_data)
        self.timer.start(10)

    def read_data(self):
        power_vals = []
        needed = self.total_bins // FREQ_BINS_PER_LINE
        for _ in range(needed):
            line = self.hackrf.process.stdout.readline()
            if not line:
                return
            power_vals.extend(self.hackrf.process_line(line))
        if len(power_vals) != self.total_bins:
            return

        psd = np.array(power_vals, dtype=np.float32)
        clean_psd = self.cleaner.clean(psd)
        detections = self.cfar.apply_cfar(clean_psd)

        self.max_hold = self.spectrum_plot.update(self.frequencies, clean_psd, self.max_hold)
        self.waterfall_plot.update(clean_psd)

        idx = np.where(detections == 1)[0]
        if idx.size:
            det_f = self.frequencies[idx]
            det_p = clean_psd[idx]
            if hasattr(self, "det_curve"):
                self.det_curve.setData(det_f, det_p, pen=None, symbol="o", symbolBrush="r")
            else:
                self.det_curve = self.spectrum_plot.plot.plot(det_f, det_p, pen=None, symbol="o", symbolBrush="r")

    # ------------------------------------------------------------------
    def set_frequency_range(self):
        self.freq_low, self.freq_high = self.freq_manager.next_range()
        self.total_bins = self.freq_high - self.freq_low
        self.frequencies = np.linspace(self.freq_low * 1e6, self.freq_high * 1e6, self.total_bins)

        # resize helper objects
        self.waterfall_plot.resize(self.freq_low, self.freq_high, self.total_bins)
        self.spectrum_plot.total_bins = self.total_bins
        self.spectrum_plot.plot.setXRange(self.freq_low * 1e6, self.freq_high * 1e6)

        self.cleaner = SpurMaskCleaner(self.total_bins)
        self.reset_max_hold()
        self.hackrf.restart(self.freq_low, self.freq_high)

    def reset_max_hold(self):
        self.max_hold = np.full(self.total_bins, -100, dtype=np.float32)

    # ------------------------------------------------------------------
    def apply_gain(self):
        try:
            self.hackrf.gain = int(self.gain_in.text())
            self.hackrf.restart(self.freq_low, self.freq_high)
            self.reset_max_hold()
        except ValueError:
            print("[ERR] Invalid VGA gain value")

    def apply_lna_gain(self):
        try:
            self.hackrf.lna_gain = int(self.lna_in.text())
            self.hackrf.restart(self.freq_low, self.freq_high)
            self.reset_max_hold()
        except ValueError:
            print("[ERR] Invalid LNA gain value")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication([])
    win = MainWindow()
    win.show()
    atexit.register(win.hackrf.stop)
    app.exec_()
