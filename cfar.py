import numpy as np

class CFARProcessor:
    def __init__(self, num_guard_cells=200, num_training_cells=400, bias=9):
        self.num_guard_cells = num_guard_cells
        self.num_training_cells = num_training_cells
        self.bias = bias

    def apply_cfar(self, power_values):
        """Apply CFAR detection to power values"""
        num_cells = len(power_values)
        detection_result = np.zeros(num_cells)
        
        for i in range(self.num_training_cells, num_cells - self.num_training_cells):
            guard_start = i - self.num_guard_cells
            guard_end = i + self.num_guard_cells
            train_start = guard_start - self.num_training_cells
            train_end = guard_end + self.num_training_cells
            
            if train_start < 0 or train_end >= num_cells:
                continue  # Skip edge cases
            
            training_cells = np.concatenate((power_values[train_start:guard_start], power_values[guard_end:train_end]))
            noise_level = np.mean(training_cells)
            threshold = noise_level + self.bias * np.std(training_cells)
            
            if power_values[i] > threshold:
                detection_result[i] = 1
        
        return detection_result