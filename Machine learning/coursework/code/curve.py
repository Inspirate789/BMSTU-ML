import os
import matplotlib.pyplot as plt
import math
import numpy as np
from statistics import median

def mean(data, weights = None):
    res = 0
    for i, v in enumerate(data):
        res += v * (1 if weights is None else weights[i])
    return res / (len(data) if weights is None else sum(weights))

def variance(data):
    return median([abs(v-median(data)) for v in data])

def smoothed_rssi(dist, measurements):
    etalon_mean, etalon_dist, etalon_var = 0, 0, float('+inf')
    second_etalon_mean, second_etalon_dist, second_etalon_var = 0, 0, float('+inf')
    
    avg = [0.]*len(measurements)
    var = [0.]*len(measurements)
    
    for i, measurement in enumerate(measurements):
        avg[i] = median(measurement)
        var[i] = variance(measurement)
        if var[i] < etalon_var:
            etalon_mean, etalon_dist, etalon_var = avg[i], dist[i], var[i]
        elif var[i] < second_etalon_var:
            second_etalon_mean, second_etalon_dist, second_etalon_var = avg[i], dist[i], var[i]
    
    print(f"mean: {avg}")
    print(f"var: {var}")
    print(f"first etalon dist {etalon_dist}m, mean {etalon_mean:.2f}, var {etalon_var:.2f}")
    print(f"second etalon dist {second_etalon_dist}m, mean {second_etalon_mean:.2f}, var {second_etalon_var:.2f}")
    
    # etalon_mean, etalon_dist = avg[0], dist[0]
    
    res = [0.]*len(measurements)
    for i in range(len(var)): 
        (m, d) = (etalon_mean, etalon_dist) if dist[i] != etalon_dist else (second_etalon_mean, second_etalon_dist)
        res[i] = (m - avg[i]) / (10*math.log10(dist[i]/d))
        
    weights = [1/v for v in var]
    print("n: " + ",".join([f"({dist[i]}m, {res[i]:.2f}, w: {weights[i]:.2f})" for i in range(len(res))]))
    
    n_avg = mean(res)
    n_weighted_avg = mean(res, weights)
    print(f"average n: {n_avg:.2f}, weighted: {n_weighted_avg:.2f}")
    
    
    for i, d in enumerate(dist):
        plt.plot([d, d], [min(measurements[i]), max(measurements[i])], marker='_', ms=10, color='k', label='Диапазоны значений RSSI' if i == 0 else None)
    
    dists = np.linspace(min(dist), max(dist), 1000)
    plt.plot(dists, [etalon_mean - 10*n_avg*math.log10(d/etalon_dist) for d in dists], '--r', label='Зависимость, основанная на среднем n')
    plt.plot(dists, [etalon_mean - 10*n_weighted_avg*math.log10(d/etalon_dist) for d in dists], 'g', label='Зависимость, основанная на среднем взвешенном n')
    
def main():
    filenames = sorted(os.listdir("data"), key=lambda filename: int(filename.split("m")[0]))
    # columns_count = (len(filenames) + 2) // 3
    # fig, axs = plt.subplots(3, columns_count)
    
    measurements = []
    dist = []
    rssi_avg = []
    rssi_med = []
    
    for i, filename in enumerate(filenames):
        with open(os.path.join("data", filename)) as file:
            measurement = [int(v) for v in file.readlines()]
            # measurements = [v for v in measurements if abs(v - statistics.mean(measurements)) <= math.sqrt(statistics.variance(measurements))]
            
            # intervals_count = math.floor(math.log(len(measurement), 2)) + 1
            # interval_length = (max(measurement) - min(measurement)) / intervals_count
            # axs[i // columns_count][i % columns_count].hist(measurement, bins=max(measurement) - min(measurement) + 1)
            # axs[i // columns_count][i % columns_count].set_title('{}m, {}, {}'.format(
            #     filename.split("m")[0], 
            #     round(mean(measurement), 2),
            #     round(math.sqrt(variance(measurement)), 2),
            # ))
            dist.append(int(filename.split("m")[0]))
            rssi_avg.append(mean(measurement))
            rssi_med.append(median(measurement))
            measurements.append(measurement)
    
    # plt.show()
    plt.xticks(np.arange(min(dist), max(dist) + 1, 1))
    plt.xlabel('Расстояние, м')
    plt.ylabel('RSSI, дБм')
    # plt.plot(dist, rssi_avg, 'mo', ms=10, label='Средние значения RSSI')
    plt.plot(dist, rssi_med, 'bx', ms=10, label='Медианные значения RSSI')
    smoothed_rssi(dist, measurements)
    plt.grid(linestyle='--')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
