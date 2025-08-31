import numpy as np

def variance(data, center_fn):
    return np.sqrt(center_fn(np.abs(data - center_fn(data)) ** 2))

def normalize_counts_to_frequencies(matrix, min_val, max_val):
    range_size = max_val - min_val + 1
    frequencies_matrix = np.zeros((len(matrix), range_size))
    
    for i, row in enumerate(matrix):
        unique_values, counts = np.unique(row, return_counts=True)
        for val, count in zip(unique_values, counts):
            if min_val <= val <= max_val:
                index = val - min_val
                frequencies_matrix[i, index] = count / len(row)
    
    return frequencies_matrix

def calculate_distributions(rssi, rssi_filtered, rssi_range):
    rssi_frequencies = normalize_counts_to_frequencies(rssi, rssi_range[0], rssi_range[-1])
    rssi_filtered_frequencies = normalize_counts_to_frequencies(rssi_filtered, rssi_range[0], rssi_range[-1])

    for row in rssi_frequencies:
        assert abs(row.sum() - 1) < 1e-5
    for row in rssi_filtered_frequencies:
        assert abs(row.sum() - 1) < 1e-5
    
    return rssi_frequencies, rssi_filtered_frequencies

def calculate_mean_vars(rssi):
    means = list(map(np.mean, rssi))
    medians = list(map(np.median, rssi))
    vars = [variance(row, np.mean) for row in rssi]
    median_vars = [variance(row, np.median) for row in rssi]
    
    return means, medians, vars, median_vars
