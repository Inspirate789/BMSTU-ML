import matplotlib.pyplot as plt
from input import read_input
from matstat import calculate_mean_vars

def main():
    rssi, rssi_filtered, dist_range, _ = read_input()

    _, medians, _, median_vars = calculate_mean_vars(rssi)
    _, medians_filtered, _, median_vars_filtered = calculate_mean_vars(rssi_filtered)

    fig = plt.figure()

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(dist_range, medians, label='Без фильтрации')
    ax.plot(dist_range, medians_filtered, label='С фильтрацией')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('RSSI, дБм')
    ax.set_title('Медианные значения RSSI')
    ax.legend(loc='best')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(dist_range, median_vars, label='Без фильтрации')
    ax.plot(dist_range, median_vars_filtered, label='С фильтрацией')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('RSSI, дБм$^2$')
    ax.set_title('Среднеквадратичные медианные отклонения RSSI')
    ax.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main()
