import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal

# Генерация синтетических данных (смесь двух Гауссиан)
np.random.seed(42)
n_samples = 1000
means = [[0, 0], [5, 5]]
covs = [[[1, 0.5], [0.5, 1]], [[1, -0.5], [-0.5, 1]]]
points = np.vstack([
    np.random.multivariate_normal(means[i], covs[i], n_samples)
    for i in range(2)
])

# Создание координатной сетки
xgrid = np.linspace(-3, 8, 100)
ygrid = np.linspace(-3, 8, 100)
X, Y = np.meshgrid(xgrid, ygrid)
XY = np.vstack([X.ravel(), Y.ravel()]).T

# Обучение гауссовой смеси (EM-алгоритм)
n_components = 2
gmm = GaussianMixture(n_components=n_components)
gmm.fit(points)
weights = gmm.weights_
means = gmm.means_
covariances = gmm.covariances_

# Вычисление плотности смеси на сетке
Z = np.sum([
    weights[i] * multivariate_normal(means[i], covariances[i]).pdf(XY)
    for i in range(n_components)
], axis=0).reshape(X.shape)

# Визуализация
fig = plt.figure(figsize=(12, 6))

# 2D контурная диаграмма
ax1 = fig.add_subplot(121)
ax1.scatter(points[:, 0], points[:, 1], s=1, alpha=0.3)
ax1.contour(X, Y, Z, levels=10, colors='red')
ax1.set_title('Контуры гауссовой смеси')
ax1.set_xlabel('Признак 1')
ax1.set_ylabel('Признак 2')

# 3D поверхность плотности
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax2.set_title('3D поверхность плотности')
ax2.set_xlabel('Признак 1')
ax2.set_ylabel('Признак 2')
ax2.set_zlabel('Плотность')

plt.tight_layout()
plt.show()
