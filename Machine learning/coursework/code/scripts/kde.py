import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
from input import read_input
from matstat import calculate_distributions

def create_samples_from_weights(X_grid, Y_grid, W):
    W_norm = W / W.sum()
    indices = np.random.choice(np.arange(len(W_norm.ravel())), 
                               size=100000, 
                               p=W_norm.ravel())
    x_samples = X_grid.ravel()[indices]
    y_samples = Y_grid.ravel()[indices]
    return np.column_stack([x_samples, y_samples])

def main():
    rssi, rssi_filtered, dist_range, rssi_range = read_input()
    _, rssi_filtered_frequencies = calculate_distributions(rssi, rssi_filtered, rssi_range)

    X, Y = np.meshgrid(rssi_range, dist_range)
    W = rssi_filtered_frequencies

    # Создание выборки
    samples = create_samples_from_weights(X, Y, W)

    # Обучение гауссовой смеси (EM-алгоритм)
    n_components = 4
    kde = KernelDensity()
    kde.fit(samples)
    weights = kde.weights_
    means = kde.means_
    covariances = kde.covariances_

    # Вычисление плотности смеси на сетке
    Z = np.sum([
        weights[i] * multivariate_normal(means[i], covariances[i]).pdf(np.vstack([X.ravel(), Y.ravel()]).T)
        for i in range(n_components)
    ], axis=0).reshape(X.shape)

    # Визуализация
    fig = plt.figure(figsize=(12, 6))

    # 2D контурная диаграмма
    ax1 = fig.add_subplot(121)
    ax1.scatter(samples[:, 0], samples[:, 1], s=1, alpha=0.3)
    ax1.scatter(means[:,0], means[:,1], c='red', s=50, edgecolor='black')
    ax1.contour(X, Y, Z, levels=15, cmap=plt.cm.coolwarm)
    ax1.set_title('Контуры гауссовой смеси')
    ax1.set_xlabel('RSSI, дБм')
    ax1.set_ylabel('Расстояние, м')

    # 3D поверхность плотности
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(X, Y, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    ax2.set_title('3D поверхность плотности')
    ax2.set_xlabel('RSSI, дБм')
    ax2.set_ylabel('Расстояние, м')
    ax2.set_zlabel('Плотность')

    plt.show()

if __name__ == '__main__':
    main()
