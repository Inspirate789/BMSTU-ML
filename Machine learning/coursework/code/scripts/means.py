import matplotlib.pyplot as plt
from input import read_input
from matstat import calculate_mean_vars

def main():
    rssi, rssi_filtered, dist_range, _ = read_input()

    means, _, vars, _ = calculate_mean_vars(rssi)
    means_filtered, _, vars_filtered, _ = calculate_mean_vars(rssi_filtered)

    fig = plt.figure()

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(dist_range, means, label='Без фильтрации')
    ax.plot(dist_range, means_filtered, label='С фильтрацией')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('RSSI, дБм')
    ax.set_title('Средние значения RSSI')
    ax.legend(loc='best')

    ax = fig.add_subplot(2, 1, 2)
    ax.plot(dist_range, vars, label='Без фильтрации')
    ax.plot(dist_range, vars_filtered, label='С фильтрацией')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('RSSI, дБм')
    ax.set_title('Среднеквадратичные отклонения RSSI')
    ax.legend(loc='best')

    plt.show()


if __name__ == '__main__':
    main()
