import numpy as np
import subprocess
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
from PyQt5.QtWidgets import QApplication, QPushButton, QWidget, QVBoxLayout, QMainWindow, QLineEdit, QLabel
import atexit
from pyqtgraph import ColorMap
from cfar import CFARProcessor


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


class FrequencyManager:
    """Manages the available frequency ranges."""
    def __init__(self):
        self.freq_dict = {
            1: [300, 600],
            2: [900, 1500],
            3: [2000, 3000],
            4: [4500, 5500],
            5: [0, 6000]
        }
        self.counter = 1

    def next_range(self):
        self.counter += 1
        temp_freq_dict_val_low = self.freq_dict[self.counter][0]
        temp_freq_dict_val_high = self.freq_dict[self.counter][1]
        if self.counter >= len(self.freq_dict):
            self.counter = 0
        
        return temp_freq_dict_val_low, temp_freq_dict_val_high
    

class HackRFController:
    """Handles HackRF interaction."""
    def __init__(self):
        self.lna_gain = 32
        self.gain = 40
        self.process = None

    def start(self, freq_low, freq_high):
        self.process = subprocess.Popen(
            ["hackrf_sweep", "-f", f"{freq_low}:{freq_high}", "-l", str(self.lna_gain), "-g", str(self.gain)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        return self.process
    
    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait()
            self.process = None

    def restart(self, freq_low, freq_high):
        self.stop()
        self.start(freq_low, freq_high)

    def process_line(self, line):
        parts = line.strip().split(", ")
        try:
            values = list(map(float, parts[-FREQ_BINS_PER_LINE:]))
            return values if len(values) == FREQ_BINS_PER_LINE else []
        except ValueError:
            return []
    

class SpectrumPlot:
    """Handles the real-time spectrum plot."""
    def __init__(self, layout, freq_low, freq_high, total_bins):
        self.plot = layout.addPlot(title='Real-Time Spectrum with Max Hold')
        self.plot.setLabel('bottom', 'Frequency (Hz)')
        self.plot.setLabel('left', 'Power (dB)')
        self.plot.setXRange(freq_low * 1e6, freq_high * 1e6)
        self.plot.setYRange(-100, 0)
        self.total_bins = total_bins

        self.real_time_curve = self.plot.plot(pen='y')
        self.max_hold_curve = self.plot.plot(pen='r')

        # Setting title from start
        self.plot.setTitle(f'Real-Time Spectrum: {freq_low} - {freq_high} MHz')


    def update(self, frequencies, power_values, max_hold):
        if len(power_values) == self.total_bins:
            # Ensure max_hold is initialized correctly
            if max_hold is None or len(max_hold) != self.total_bins:
                max_hold = np.full(self.total_bins, -100)

            # Only update max_hold if new power values are higher than the current values in max_hold
            for i in range(self.total_bins):
                if power_values[i] > max_hold[i]:
                    max_hold[i] = power_values[i]

            # Update the plot with frequencies in Hz (converted to GHz)
            self.real_time_curve.setData(frequencies, power_values)
            self.max_hold_curve.setData(frequencies, max_hold)

            



class WaterfallPlotWidget:
    """Handles the waterfall plot."""
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

        # Scale the iamge in x-direction by a factor of 1/1000
        self.waterfall_img.setTransform(QtGui.QTransform().scale(1e6, 1))

    def update(self, power_values):
        # Update the waterfall plot
        self.waterfall_data[1:, :] = self.waterfall_data[:-1, :]
        self.waterfall_data[0, :] = np.array(power_values)

        # Normalize the waterfall data for better visibility
        min_power, max_power = np.min(self.waterfall_data), np.max(self.waterfall_data)
        if max_power - min_power > 1e-3:  
            scaled_waterfall = (self.waterfall_data - min_power) / (max_power - min_power)
        else:
            scaled_waterfall = np.zeros_like(self.waterfall_data)  

        # Update the image position to move downwards
        self.waterfall_img.setImage(scaled_waterfall.T, autoLevels=False, lut=lut, levels=(0, 1))
        self.waterfall_img.setPos(self.freq_low * 1e6, 0)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('HackRF Spectrum Sweep')
        self.resize(1000, 800)

        self.freq_manager = FrequencyManager()
        self.hackrf = HackRFController()

        self.freq_low = self.freq_manager.freq_dict[self.freq_manager.counter][0]
        self.freq_high = self.freq_manager.freq_dict[self.freq_manager.counter][1]

        self.total_bins = int(self.freq_high - self.freq_low)
        self.frequencies = np.linspace(self.freq_low * 1e6, self.freq_high * 1e6, self.total_bins)
        self.waterfall_data = np.full((WATERFALL_DEPTH, 100), -100)
        self.max_hold = np.full(self.total_bins, -100)
        self.power_values = []

        self.layout = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.layout)

        self.spectrum_plot = SpectrumPlot(self.layout, self.freq_low, self.freq_high, self.total_bins)
        self.layout.nextRow()
        self.waterfall_plot = WaterfallPlotWidget(self.layout, self.freq_low, self.freq_high, self.total_bins)
        self.cfar = CFARProcessor()

        self.init_ui()
        self.start_hackrf()


    def init_ui(self):
        self.ui_widget = QWidget()
        self.ui_layout = QVBoxLayout()

        self.apply_gain_button = QPushButton('Apply Gain')
        self.apply_gain_button.clicked.connect(self.apply_gain)
        self.gain_input = QLineEdit()
        self.gain_input.setText(str(self.hackrf.gain))

        self.apply_lna_gain_button = QPushButton('Apply LNA Gain')
        self.apply_lna_gain_button.clicked.connect(self.apply_lna_gain)
        self.lna_gain_input = QLineEdit()
        self.lna_gain_input.setText(str(self.hackrf.lna_gain))
        
        self.ui_layout.addWidget(QLabel("Enter VGA Gain:"))
        self.ui_layout.addWidget(self.gain_input)
        self.ui_layout.addWidget(self.apply_gain_button)
        self.ui_layout.addWidget(QLabel("Enter LNA Gain:"))
        self.ui_layout.addWidget(self.lna_gain_input)
        self.ui_layout.addWidget(self.apply_lna_gain_button)

        self.freq_button = QPushButton('Set Frequency Range')
        self.freq_button.clicked.connect(self.set_frequency_range)
        self.ui_layout.addWidget(self.freq_button)

        self.reset_button = QPushButton('Reset Max Hold')
        self.reset_button.clicked.connect(self.reset_max_hold)
        self.ui_layout.addWidget(self.reset_button)

        self.ui_widget.setLayout(self.ui_layout)
        self.setMenuWidget(self.ui_widget)

    def start_hackrf(self):
        try:
            self.hackrf.start(self.freq_low, self.freq_high) 
            self.read_data()
                        
            # Set up periodic reading of HackRF data
            self.read_timer = QtCore.QTimer()
            self.read_timer.timeout.connect(self.read_data)
            self.read_timer.start(10)
        except Exception as e:
            print(f"Error: {e}")


    def read_data(self):
        power_values = []
        lines_processed = 0

        for line in self.hackrf.process.stdout:
            if line.strip():
                power_values.extend(self.hackrf.process_line(line))
                lines_processed += 1

                if lines_processed >= self.total_bins / FREQ_BINS_PER_LINE:
                    # Apply CFAR
                    detection_result = self.cfar.apply_cfar(power_values)

                    # Update spectrum plot with detections
                    self.spectrum_plot.update(self.frequencies, power_values, self.max_hold)

                    # Overlay detections (example: mark detections in red)
                    detected_indices = np.where(detection_result == 1)[0]
                    detected_frequencies = self.frequencies[detected_indices]
                    detected_powers = np.array(power_values)[detected_indices]

                    if hasattr(self, "detection_curve"):
                        self.detection_curve.setData(detected_frequencies, detected_powers, pen=None, symbol='o', symbolBrush='r')
                    else:
                        self.detection_curve = self.spectrum_plot.plot.plot(detected_frequencies, detected_powers, pen=None, symbol='o', symbolBrush='r')

                    # Update waterfall plot
                    self.waterfall_plot.update(power_values)
                    break


    def set_frequency_range(self):
        """ Apply the next frequency range from FrequencyManager """
        self.freq_low, self.freq_high = self.freq_manager.next_range()
        self.waterfall_plot.freq_low, self.waterfall_plot.freq_high = self.freq_low, self.freq_high

        # Calculate the total number of frequency bins
        self.total_bins = int(self.freq_high - self.freq_low)
        self.spectrum_plot.total_bins = self.total_bins

        # Update frequency axis for plotting
        self.frequencies = np.linspace(self.freq_low * 1e6, self.freq_high * 1e6, self.total_bins)

        # Reset waterfall and power values to avoid display issues
        self.waterfall_plot.waterfall_data = np.full((WATERFALL_DEPTH, self.total_bins), -100)
        self.power_values = []
        
        # Update the X-range of both the spectrum and waterfall plot
        self.spectrum_plot.plot.setXRange(self.freq_low * 1e6, self.freq_high * 1e6)
        self.waterfall_plot.plot.setXRange(self.freq_low * 1e6, self.freq_high * 1e6)
        
        # Update the Y-range of the waterfall plot to match the depth of the waterfall (time dimension)
        self.waterfall_plot.plot.setYRange(0, WATERFALL_DEPTH)

        # Updating the title of the plot
        self.spectrum_plot.plot.setTitle(f'Real-Time Spectrum: {self.freq_low} - {self.freq_high} MHz')
        
        # Restart the process with the new frequency range
        self.hackrf.restart(self.freq_low, self.freq_high)

        # Reset the max_hold to reflect the new frequency range
        self.reset_max_hold()

        # Forcing a redraw to ensure that the plot reflects the changes immediately
        app.processEvents()


    def reset_max_hold(self):
        """ Reset the max hold """
        # Calculate the total number of frequency bins
        total_lines = int((self.freq_high - self.freq_low) / FREQ_BINS_PER_LINE)
        total_freq_bins = int(total_lines * FREQ_BINS_PER_LINE)
        
        self.max_hold = np.full(total_freq_bins, -100)


    def apply_gain(self):
        """ Apply new gain value and restart the HackRF process """
        try:
            self.hackrf.gain = int(self.gain_input.text())
            self.hackrf.restart(self.freq_low, self.freq_high)
            self.reset_max_hold()
        except ValueError:
            print('Invalid gain input')

    def apply_lna_gain(self):
        """ Apply new lna gain value and restart the HackRF process """
        try:
            self.hackrf.lna_gain = int(self.lna_gain_input.text())
            self.hackrf.restart(self.freq_low, self.freq_high)
            self.reset_max_hold()
        except ValueError:
            print('Invalid LNA gain input')


if __name__ == "__main__":
    app = QApplication([])
    main_window = MainWindow()
    main_window.show()
    atexit.register(main_window.hackrf.stop)
    app.exec_()
