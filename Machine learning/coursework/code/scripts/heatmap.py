import matplotlib.pyplot as plt
from input import read_input
from matstat import calculate_distributions

def main():
    rssi, rssi_filtered, dist_range, rssi_range = read_input()
    rssi_frequencies, rssi_filtered_frequencies = calculate_distributions(rssi, rssi_filtered, rssi_range)

    fig = plt.figure()

    ax = fig.add_subplot(2, 1, 1)
    im = ax.imshow(rssi_frequencies.transpose(), extent=[dist_range[0], dist_range[-1], rssi_range[-1], rssi_range[0]], cmap=plt.cm.coolwarm)
    ax.set_aspect(0.055)
    cbar = ax.figure.colorbar(im, ax=ax, cmap=plt.cm.coolwarm)
    cbar.ax.set_ylabel('Частота')
    
    ax = fig.add_subplot(2, 1, 2)
    im = ax.imshow(rssi_filtered_frequencies.transpose(), extent=[dist_range[0], dist_range[-1], rssi_range[-1], rssi_range[0]], cmap=plt.cm.coolwarm)
    ax.set_aspect(0.055)
    cbar = ax.figure.colorbar(im, ax=ax, cmap=plt.cm.coolwarm)
    cbar.ax.set_ylabel('Частота')
    ax.set_xlabel('Расстояние, м')
    ax.set_ylabel('RSSI, дБм')

    plt.show()


if __name__ == '__main__':
    main()
