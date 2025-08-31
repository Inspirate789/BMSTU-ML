import numpy as np
from sklearn.linear_model import LinearRegression
from input import read_input
from matstat import calculate_distributions, calculate_mean_vars

def make_regression(rssi_range, dist_range, rssi_frequencies, rssi_0, dist_0):
    rssi_grid, dist_grid = np.meshgrid(rssi_range, dist_range)

    rssi_data = rssi_grid.ravel().reshape(-1, 1)
    dist_data = dist_grid.ravel().reshape(-1, 1)
    weights = rssi_frequencies.reshape(-1, 1).ravel()

    dist_transformed = np.log10(dist_data / dist_0)  # log10(d/d0)
    rssi_transformed = rssi_data - rssi_0  # P - P0

    model = LinearRegression(fit_intercept=False)
    model.fit(dist_transformed, rssi_transformed, sample_weight=weights)

    k = model.coef_[0][0]
    n = -k / 10
    
    return n

def main():
    rssi, rssi_filtered, dist_range, rssi_range = read_input()
    _, rssi_filtered_frequencies = calculate_distributions(rssi, rssi_filtered, rssi_range)
    means_filtered, _, vars_filtered, _ = calculate_mean_vars(rssi_filtered)

    best_point = np.argmin(vars_filtered[5:]) + 5
    dist_0 = dist_range[best_point]
    rssi_0 = means_filtered[best_point]

    print(f"d0: {dist_0} м")
    print(f"P0: {rssi_0} дБм (var: {vars_filtered[best_point]:.2f} дБм)")

    n = make_regression(rssi_range, dist_range, rssi_filtered_frequencies, rssi_0, dist_0)

    print(f"n: {n:.10f}")

if __name__ == "__main__":
    main()
