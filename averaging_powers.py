import numpy as np
import csv

class SpectrumProcessor:
    def __init__(self, filepath="Measurement Noise/measurement_noise_hackrf_one.csv", target_count=6000):
        self.filepath = filepath
        self.target_count = target_count
        self.current_index = 0
        self.power_avg = [0 for _ in range(target_count)]
        self.count = 0
        self.lines = self.load_dataset()

    def load_dataset(self):
        """Load the dataset from the CSV file."""
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

    def get_next(self):
        """Get the next set of frequency and power data."""
        full_freqs = []
        full_powers = []
        count = 0

        while count < self.target_count:
            if self.current_index >= len(self.lines):
                if count == 0:
                    return None  # No more data
                break  # Return whatever we have
            freqs, powers = self.lines[self.current_index]
            self.current_index += 1

            needed = self.target_count - count
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

    def read_data(self):
        """Process the data and calculate the averaged powers."""
        result = self.get_next()
        if result is None:
            print("End of data.")
            return self.power_avg  # Return the final averaged powers when done

        freqs, powers = result

        # Calculate running average for each frequency
        self.power_avg = [(self.power_avg[i] + powers[i]/200) for i in range(len(self.power_avg))]


        return None  # Continue processing

    def process_data(self):
        """Process all the data and return the final averaged power values."""
        while True:
            result = self.read_data()
            if result is not None:
                return result  # Return power_avg when processing is complete
            

def plotter():
    import matplotlib.pyplot as plt

    spectrum_processor_hackrf_one = SpectrumProcessor(filepath="Measurement Noise/measurement_noise_hackrf_one.csv")
    spectrum_processor_no_antenna = SpectrumProcessor(filepath="Measurement Noise/measurement_noise_no_antenna.csv")
    spectrum_processor_with_antenna = SpectrumProcessor(filepath="Measurement Noise/measurement_with_antenna.csv")


    x = np.arange(0, 6000, 1)
    y_resistance = spectrum_processor_hackrf_one.process_data()
    y_no_antenna = spectrum_processor_no_antenna.process_data()
    y_with_antenna = spectrum_processor_with_antenna.process_data()

    plt.plot(x, y_resistance)
    plt.title("Data from the HackRF One equipped with 50 Ohm terminator averaged over 200 sweep samples", fontsize=18)
    plt.xlabel("MHz", fontsize=18)
    plt.ylabel("dB", fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize = 18)
    plt.show()

    plt.plot(x, y_no_antenna)
    plt.title("Data from the HackRF One with no antenna averaged over 200 sweep samples", fontsize=18)
    plt.xlabel("MHz", fontsize=18)
    plt.ylabel("dB", fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize = 18)
    plt.show()


    y_difference = [y_resistance[i] - y_no_antenna[i] for i in range(len(y_resistance))]

    plt.plot(x, y_difference)
    plt.title("HackRF One: 50Ω Terminator vs No Antenna (Avg of 200 Sweeps)", fontsize=18)
    plt.xlabel("MHz", fontsize = 18)
    plt.ylabel("dB", fontsize = 18)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize = 18)
    plt.show()

    plt.plot(x, y_with_antenna)
    plt.title("Measurement with antenna averaged over 200 sweep samples", fontsize=18)
    plt.xlabel("MHz", fontsize = 18)
    plt.ylabel("dB", fontsize = 18)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize = 18)
    plt.show()

    y_resistance_vs_antenna = [y_resistance[i] - y_with_antenna[i] for i in range(len(y_resistance))]

    plt.plot(x, y_resistance_vs_antenna)
    plt.title("Measurement difference of resistance vs antenna averaged over 200 sweep samples", fontsize=18)
    plt.xlabel("MHz", fontsize=18)
    plt.ylabel("dB", fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize = 18)
    plt.show()