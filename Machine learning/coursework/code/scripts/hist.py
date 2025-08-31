import numpy as np
import matplotlib.pyplot as plt
from input import read_input
from matstat import calculate_distributions

def main():
    rssi, rssi_filtered, dist_range, rssi_range = read_input()
    rssi_frequencies, rssi_filtered_frequencies = calculate_distributions(rssi, rssi_filtered, rssi_range)

    x, y = np.meshgrid(rssi_range, dist_range)
    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.plot_surface(x, y, rssi_frequencies, cmap=plt.cm.coolwarm)
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('RSSI, дБм')
    ax.set_zlabel('Частота')
    ax.set_title('Исходные данные')
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.plot_surface(x, y, rssi_filtered_frequencies, cmap=plt.cm.coolwarm)
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('RSSI, дБм')
    ax.set_zlabel('Частота')
    ax.set_title('Результат применения фильтра Калмана')

    plt.show()


if __name__ == '__main__':
    main()
