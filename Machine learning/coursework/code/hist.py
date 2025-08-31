import os
import matplotlib.pyplot as plt
import math
import numpy as np
import statistics
from scipy.stats import cauchy, norm

def mean(data, weights = None):
    res = 0
    for i, v in enumerate(data):
        res += v * (1 if weights is None else weights[i])
    return res / (len(data) if weights is None else sum(weights))

def variance(data):
    res = 0
    for v in data:
        res += (v-mean(data))**2
    return res / (len(data) - 1)

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0

    def predict(self, u = 0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(self.H, np.dot(self.P, self.H.T))
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.n)
        self.P = np.dot(np.dot(I - np.dot(K, self.H), self.P), 
        	(I - np.dot(K, self.H)).T) + np.dot(np.dot(K, self.R), K.T)
    
def main():
    filenames = sorted(os.listdir("new"), key=lambda filename: float(filename.split("m")[0]))
    with open(os.path.join("new", "1.5m.txt")) as file:
        measurement = [int(v) for v in file.readlines()]
        # measurements = [v for v in measurements if abs(v - statistics.mean(measurements)) <= math.sqrt(statistics.variance(measurements))]
        
        dt = 1.0/60
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        H = np.array([1, 0, 0]).reshape(1, 3)
        Q = np.array([[0.05, 0.05, 0.0], [0.05, 0.05, 0.0], [0.0, 0.0, 0.0]])
        R = np.array([0.5]).reshape(1, 1)
        kf = KalmanFilter(F = F, H = H, Q = Q, R = R)
        predictions = []
        for z in measurement:
            predictions.append(np.dot(H,  kf.predict())[0])
            kf.update(z)
        
        # intervals_count = math.floor(math.log(len(measurement), 2)) + 1
        # interval_length = (max(measurement) - min(measurement)) / intervals_count
        
        # plt.plot(predictions)
        # plt.plot(measurement)
        predictions = [float(v[0]) for v in predictions if float(v[0]) < max(measurement)+1]
        plt.hist(predictions, bins=math.floor(max(predictions) - min(predictions) + 1), density=True, alpha=0.5, color='b', label = 'RSSI после применения фильтра Калмана', hatch='/')
        plt.hist(measurement, bins=max(measurement) - min(measurement) + 1, density=True, alpha=0.5, color = 'r', label = 'Измереннные значения RSSI', hatch='\\')
        print('{}m, {}, {}, {}'.format(
            filenames[0].split("m")[0], 
            statistics.median(measurement),
            mean(measurement),
            round(math.sqrt(variance(measurement)), 2),
        ))
        print('{}m, {}, {}, {}'.format(
            filenames[0].split("m")[0], 
            statistics.median(predictions),
            mean(predictions),
            round(math.sqrt(variance(predictions)), 2),
        ))
        print(statistics.median([abs(v - statistics.median(predictions)) for v in predictions]))
        print(variance(predictions))
        print()
        
        #print(f'{min(predictions)}, {max(predictions)}')
        print(f'{math.floor(min(predictions))}, {math.ceil(max(predictions))}')
        r = [v for v in range(math.floor(min(predictions)), math.ceil(max(predictions))+1)]
        cdf = [sum(1 for p in predictions if p < v)/len(predictions) for v in r]
        #print(cdf)
        ccdf = cauchy.cdf(r, loc=-58.12, scale=2.25)
        ncdf = norm.cdf(r, loc=-58.41, scale=3.14)
        print(max(abs(ccdf[i] - cdf[i]) for i in range(len(cdf))))
        print(max(abs(ncdf[i] - cdf[i]) for i in range(len(cdf))))
        # plt.plot(r, ccdf, linestyle='dotted', label = 'Функция распределения Коши')
        # plt.plot(r, ncdf, linestyle='dashdot', label = 'Функция нормального распределения')
        # plt.plot(r, cdf, label = 'Выборочная функция распределения')
        
        plt.xlabel('RSSI, дБм')
        plt.ylabel('Частота')
        plt.grid(linestyle='--')
        plt.legend()
        
    
    # plt.show()
    # plt.xticks(np.arange(min(dist), max(dist) + 1, 1))
    # plt.xlabel('Расстояние, м')
    # plt.ylabel('RSSI, дБм')
    # plt.plot(dist, rssi, 'bx', ms=10, label='Средние значения RSSI')
    # smoothed_rssi(dist, measurements)
    # plt.grid(linestyle='--')
    # plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
