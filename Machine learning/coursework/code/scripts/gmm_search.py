import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
from input import read_input
from matstat import calculate_distributions, calculate_mean_vars
from regression import make_regression
# from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_samples_from_weights(X_grid, Y_grid, W):
    W_norm = W / W.sum()
    indices = np.random.choice(np.arange(len(W_norm.ravel())), 
                               size=1000000, 
                               p=W_norm.ravel())
    x_samples = X_grid.ravel()[indices]
    y_samples = Y_grid.ravel()[indices]
    return np.column_stack([x_samples, y_samples])

def main():
    rssi, rssi_filtered, dist_range, rssi_range = read_input()
    _, rssi_filtered_frequencies = calculate_distributions(rssi, rssi_filtered, rssi_range)
    means_filtered, _, vars_filtered, _ = calculate_mean_vars(rssi_filtered)

    best_point = np.argmin(vars_filtered[5:]) + 5
    dist_0 = dist_range[best_point]
    rssi_0 = means_filtered[best_point]

    print(f"d0: {dist_0} м")
    print(f"P0: {rssi_0} дБм (var: {vars_filtered[best_point]:.2f} дБм)")
    best_n = make_regression(rssi_range, dist_range, rssi_filtered_frequencies, rssi_0, dist_0)
    print(f"n: {best_n:.10f}")

    X, Y = np.meshgrid(rssi_range, dist_range)
    W = rssi_filtered_frequencies
    W /= W.sum()

    # Создание выборки
    samples = create_samples_from_weights(X, Y, W)
    #samples_train, samples_test = train_test_split(samples, test_size=0.1, random_state=7)

    Z_test = []
    for sample in samples:
        Z_test.append(W[np.where(np.isclose(dist_range, sample[1])), np.where(np.isclose(rssi_range, sample[0]))][0][0])
    Z_test = np.array(Z_test)

    best_gmm = None
    
    for n_components in range(1, 11):
        if best_gmm is not None and n_components - best_gmm['n_components'] > 3:
            break

        print(f'n_components: {n_components} ... ', end='')
        for _ in range(2):
            gmm = GaussianMixture(n_components=n_components, max_iter=1000)
            gmm.fit(samples)
            weights = gmm.weights_
            means = gmm.means_
            covariances = gmm.covariances_

            Z = np.sum([
                weights[i] * multivariate_normal(means[i], covariances[i]).pdf(np.vstack([X.ravel(), Y.ravel()]).T)
                for i in range(n_components)
            ], axis=0).reshape(X.shape)

            Z_predict = np.exp(gmm.score_samples(samples))
            mse = mean_squared_error(Z_test, Z_predict)

            n = make_regression(rssi_range, dist_range, Z, rssi_0, dist_0)

            if best_gmm is not None:
                mse_diff = mse - best_gmm['mse']
                n_diff_diff = abs(n - best_n) - abs(best_gmm['n'] - best_n)

            if best_gmm is None or \
                mse_diff < -5e-5 and n_diff_diff < 1e-3 or \
                n_diff_diff < -1e-4 and mse_diff < 1e-4:
                best_gmm = {
                    'n_components': n_components,
                    'weights': weights,
                    'means': means,
                    'covariances': covariances,
                    'Z': Z,
                    'n': n,
                    'mse': mse,
                }
        print(f"best n_components: {best_gmm['n_components']}; best n: {best_gmm['n']}; best mse: {best_gmm['mse']:e}")

    
    print(best_gmm)
    means, Z = best_gmm['means'], best_gmm['Z']

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
