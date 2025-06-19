class CfarPostProcessor:
    def __init__(self, band_low=2400000000, band_high=2483500000):
        self.band_low = band_low
        self.band_high = band_high

        self.freqs_cluster = []

    def cluster_with_powers(self, freqs, powers, bandwidth=40000000):
        if len(freqs) == 0:
            return [], []

        # Sort by frequency
        sorted_data = sorted(zip(freqs, powers), key=lambda x: x[0])
        clustered_freqs = []
        clustered_powers = []

        cluster = [sorted_data[0]]

        for i in range(1, len(sorted_data)):
            freq_i, power_i = sorted_data[i]
            last_freq_in_cluster = cluster[-1][0]

            if freq_i - last_freq_in_cluster < bandwidth:
                cluster.append((freq_i, power_i))
            else:
                # Select the point in cluster with highest power
                best_freq, best_power = max(cluster, key=lambda x: x[1])
                clustered_freqs.append(best_freq)
                clustered_powers.append(best_power)
                cluster = [(freq_i, power_i)]  # Start new cluster

        # Final cluster
        if cluster:
            best_freq, best_power = max(cluster, key=lambda x: x[1])
            clustered_freqs.append(best_freq)
            clustered_powers.append(best_power)

        return clustered_freqs, clustered_powers


    def count_classification(self, final_freqs):
        tp, fp = 0, 0
        for i in final_freqs:
            if(self.band_low <= i <= self.band_high):
                tp += 1
            else:
                fp += 1

        return tp, fp