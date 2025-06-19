import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QVBoxLayout, QMainWindow
from cfar import CFARProcessor
from cfar_postprocessor import CfarPostProcessor
from averaging_powers import SpectrumProcessor
import csv


# Constants
FREQ_BINS_PER_LINE = 5
WATERFALL_DEPTH = 100

# Define blue-red colormap
colors = [
    (0, (0, 0, 225)),
    (0.95, (225, 225, 0)),
    (0.97, (225, 0, 0)),
    (1, (225, 0, 0))
]

color_map = pg.ColorMap(*zip(*colors))
lut = color_map.getLookupTable(start=0.0, stop=1.0, nPts=256)


class DroneBandChecker:
    def __init__(self):
        # Each tuple is (start_freq_in_Hz, end_freq_in_Hz)
        self.drone_bands = [
            (2.4e9, 2.4835e9),   # 2.4 GHz to 2.42 GHz
        ]

    def is_drone_band(self, freq_hz):
        for start, end in self.drone_bands:
            if start <= freq_hz <= end:
                return True
        return False

    def check_and_print(self, freq_hz):
        if self.is_drone_band(freq_hz):
            return "YES"
        else:
            return "NO"
    


class DatasetController:
    def __init__(self, filepath):
        self.filepath = filepath
        self.current_index = 0
        self.lines = self.load_dataset()

    def load_dataset(self):
        sweeps = []
        with open(self.filepath, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                try:
                    start_freq = float(row[2])
                    stop_freq = float(row[3])
                    step = float(row[4])
                    powers = [float(x) for x in row[6:]]
                    freqs = np.arange(start_freq, stop_freq, step)
                    if len(powers) == len(freqs):
                        sweeps.append((freqs, powers))

                except ValueError:
                    continue
        return sweeps

    def get_next(self, target_count=6000):
        full_freqs = []
        full_powers = []
        count = 0

        while count < target_count:
            if self.current_index >= len(self.lines):
                if count == 0:
                    return None  # No more data
                break  # Return whatever we have
            freqs, powers = self.lines[self.current_index]
            self.current_index += 1

            needed = target_count - count
            if len(powers) <= needed:
                full_freqs.extend(freqs)
                full_powers.extend(powers)
                count += len(powers)
            else:
                # Split the powers/freqs if too many
                full_freqs.extend(freqs[:needed])
                full_powers.extend(powers[:needed])
                # Save back the remaining for next call
                self.lines[self.current_index - 1] = (freqs[needed:], powers[needed:])
                count += needed

        return np.array(full_freqs), np.array(full_powers)





    

class SpectrumPlot:
    def __init__(self, layout, freq_low, freq_high, total_bins):
        self.plot = layout.addPlot(title='Real-Time Spectrum with Max Hold')
        self.plot.setLabel('bottom', 'Frequency (Hz)')
        self.plot.setLabel('left', 'Power (dB)')
        self.plot.setXRange(freq_low * 1e6, freq_high * 1e6)
        self.plot.setYRange(-100, 0)
        self.total_bins = total_bins

        self.real_time_curve = self.plot.plot(pen='y')
        self.max_hold_curve = self.plot.plot(pen='r')
        self.plot.setTitle(f'Real-Time Spectrum: {freq_low} - {freq_high} MHz')

    def update(self, frequencies, power_values, max_hold):
        if max_hold is None or len(max_hold) != len(power_values):
            max_hold = np.full(len(power_values), -100)
        for i in range(len(power_values)):
            if power_values[i] > max_hold[i]:
                max_hold[i] = power_values[i]
        self.real_time_curve.setData(frequencies, power_values)
        self.max_hold_curve.setData(frequencies, max_hold)


class WaterfallPlotWidget:
    def __init__(self, layout, freq_low, freq_high, total_bins):
        self.layout = layout
        self.history_size = WATERFALL_DEPTH
        self.freq_low = freq_low
        self.freq_high = freq_high
        self.total_bins = total_bins
        self.waterfall_data = np.full((self.history_size, self.total_bins), -100)
        self.create_plot()

    def create_plot(self):
        self.plot = self.layout.addPlot()
        self.plot.setLabel('bottom', 'Frequency (Hz)')
        self.plot.setLabel('left', 'Time')
        self.plot.setYRange(0, self.history_size)
        self.plot.setXRange(self.freq_low * 1e6, self.freq_high * 1e6)
        self.waterfall_img = pg.ImageItem()
        self.plot.addItem(self.waterfall_img)
        self.plot.invertY(True)
        self.waterfall_img.setTransform(QtGui.QTransform().scale(1e6, 1))

    def update(self, power_values):
        self.waterfall_data[1:, :] = self.waterfall_data[:-1, :]
        self.waterfall_data[0, :] = np.array(power_values)
        min_power, max_power = np.min(self.waterfall_data), np.max(self.waterfall_data)
        if max_power - min_power > 1e-3:
            scaled_waterfall = (self.waterfall_data - min_power) / (max_power - min_power)
        else:
            scaled_waterfall = np.zeros_like(self.waterfall_data)
        self.waterfall_img.setImage(scaled_waterfall.T, autoLevels=False, lut=lut, levels=(0, 1))
        self.waterfall_img.setPos(self.freq_low * 1e6, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HackRF Spectrum Sweep')
        self.resize(1000, 800)

        self.datasetcontroller = DatasetController("Bymarka_data/full_sweep_1.csv")
        self.cfar = CFARProcessor()
        self.cfar_postprocessor = CfarPostProcessor()
        self.droneband = DroneBandChecker()
        self.spectrum_processor = SpectrumProcessor()

        self.power_avg = self.spectrum_processor.process_data()

        self.freq_low = 0
        self.freq_high = 6000

        self.count_tp = 0   # True Positive
        self.count_fp = 0   # False Positive
        self.count_fn = 0   # False Negative

        first_sweep = self.datasetcontroller.get_next()
        if first_sweep is None:
            raise ValueError("No data found in dataset.")

        freqs, powers = first_sweep
        self.datasetcontroller.current_index = 0  # reset so first sweep is not skipped

        self.total_bins = len(powers)
        self.frequencies = freqs

        
        self.waterfall_data = np.full((WATERFALL_DEPTH, 100), -100)
        self.max_hold = np.full(self.total_bins, -100)
        self.power_values = []

        self.layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.layout)

        self.spectrum_plot = SpectrumPlot(self.layout, self.freq_low, self.freq_high, self.total_bins)
        self.layout.nextRow()
        self.waterfall_plot = WaterfallPlotWidget(self.layout, self.freq_low, self.freq_high, self.total_bins)

        self.init_ui()

        self.start_dataset_stream()

    def start_dataset_stream(self):
        self.read_timer = QtCore.QTimer()
        self.read_timer.timeout.connect(self.read_data)
        self.read_timer.start(100)  # 100 ms = 10 frames per second


    def init_ui(self):
        self.ui_widget = QWidget()
        self.ui_layout = QVBoxLayout()

        self.reset_button = QPushButton('Reset Max Hold')
        self.reset_button.clicked.connect(self.reset_max_hold)
        self.ui_layout.addWidget(self.reset_button)

        self.ui_widget.setLayout(self.ui_layout)
        self.setMenuWidget(self.ui_widget)


    def read_data(self):
        result = self.datasetcontroller.get_next()
        if result is None:
            return  # End of data

        freqs, powers = result

        # Ensure frequency and power values are aligned and sorted
        sort_indices = np.argsort(freqs)
        freqs = freqs[sort_indices]
        powers = powers[sort_indices]


        power_base = [powers[i] - self.power_avg[i] for i in range(len(powers))]

        detection_result = self.cfar.apply_cfar(power_base)

        # Update max hold & spectrum
        self.spectrum_plot.update(freqs, power_base, self.max_hold)

        # Overlay detections
        
        """--------------------------------------------------------"""
        
        detected_indices = np.where(detection_result == 1)[0]
        detected_freqs = freqs[detected_indices]
        detected_powers = np.array(power_base)[detected_indices]

        final_freqs, final_powers = self.cfar_postprocessor.cluster_with_powers(detected_freqs, detected_powers, bandwidth=40000000)

        print(final_freqs)

        print(self.cfar_postprocessor.count_classification(final_freqs))

        self.count_tp += self.cfar_postprocessor.count_classification(final_freqs)[0]
        self.count_fp += self.cfar_postprocessor.count_classification(final_freqs)[1]
        
        if self.cfar_postprocessor.count_classification(final_freqs)[0] == 0:
            self.count_fn += 1

        if self.count_fn == 0 and self.count_fp == 0 and self.count_tp == 0:
            accuracy_cfar = self.count_tp/1
        else:
            accuracy_cfar = self.count_tp / (self.count_tp + self.count_fp + self.count_fn)

        print(self.count_tp, self.count_fp, self.count_fn)
        print(accuracy_cfar)

        if hasattr(self, "detection_curve"):
            # This updates the detection target points
            self.detection_curve.setData(final_freqs, final_powers, pen=None, symbol='o', symbolBrush='r')
        else:
            # This initiates the detection target points
            self.detection_curve = self.spectrum_plot.plot.plot(final_freqs, final_powers, pen=None, symbol='o', symbolBrush='r')

        # Waterfall update
        self.waterfall_plot.update(power_base)



    def reset_max_hold(self):
        """ Reset the max hold """
        # Calculate the total number of frequency bins
        total_lines = int((self.freq_high - self.freq_low) / FREQ_BINS_PER_LINE)
        total_freq_bins = int(total_lines * FREQ_BINS_PER_LINE)
        
        self.max_hold = np.full(total_freq_bins, -100)


if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
