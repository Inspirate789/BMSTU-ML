import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class GMM2DEM:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        self.K = n_components
        self.max_iter = max_iter
        self.tol = tol
    
    def _initialize(self, X):
        n_samples, n_features = X.shape
        # Инициализация параметров
        idx = np.random.choice(n_samples, self.K, replace=False)
        self.means = X[idx]
        self.covs = [np.eye(n_features) for _ in range(self.K)]
        self.weights = np.ones(self.K)/self.K
        
    def _e_step(self, X):
        n_samples = X.shape[0]
        self.responsibilities = np.zeros((n_samples, self.K))
        
        for k in range(self.K):
            self.responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(
                X, mean=self.means[k], cov=self.covs[k])
        
        self.responsibilities /= self.responsibilities.sum(axis=1, keepdims=True)
    
    def _m_step(self, X):
        n_samples = X.shape[0]
        
        for k in range(self.K):
            # Обновление весов
            self.weights[k] = self.responsibilities[:,k].mean()
            
            # Обновление средних
            self.means[k] = np.average(X, axis=0, 
                                     weights=self.responsibilities[:,k])
            
            # Обновление ковариаций
            diff = X - self.means[k]
            self.covs[k] = (self.responsibilities[:,k]*diff.T) @ diff / self.responsibilities[:,k].sum()
            self.covs[k] += 1e-6 * np.eye(X.shape[1])  # Регуляризация
    
    def fit(self, X):
        self._initialize(X)
        prev_log_likelihood = -np.inf
        
        for _ in range(self.max_iter):
            # E-шаг
            self._e_step(X)
            
            # M-шаг
            self._m_step(X)
            
            # Расчет логарифмического правдоподобия
            log_likelihood = self._compute_log_likelihood(X)
            
            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break
            prev_log_likelihood = log_likelihood
    
    def _compute_log_likelihood(self, X):
        likelihood = np.zeros((X.shape[0], self.K))
        for k in range(self.K):
            likelihood[:,k] = self.weights[k] * multivariate_normal.pdf(X, 
                                                      mean=self.means[k], 
                                                      cov=self.covs[k])
        return np.log(likelihood.sum(axis=1)).sum()

def create_samples_from_weights(X_grid, Y_grid, W):
    """Создание выборки из матрицы весов"""
    W_norm = W / W.sum()
    indices = np.random.choice(np.arange(len(W_norm.ravel())), 
                               size=10000, 
                               p=W_norm.ravel())
    x_samples = X_grid.ravel()[indices]
    y_samples = Y_grid.ravel()[indices]
    return np.column_stack([x_samples, y_samples])

def visualize_results(X_grid, Y_grid, W, model):
    plt.figure(figsize=(12,6))
    
    # Визуализация исходных данных
    plt.subplot(121)
    plt.imshow(W, extent=[X_grid.min(), X_grid.max(), 
                        Y_grid.min(), Y_grid.max()],
             origin='lower', cmap='viridis')
    plt.title('Исходные данные')
    plt.colorbar()
    
    # Визуализация модели
    plt.subplot(122)
    x_min, x_max = X_grid.min(), X_grid.max()
    y_min, y_max = Y_grid.min(), Y_grid.max()
    
    xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
    pos = np.dstack((xx, yy))
    
    Z = np.zeros_like(xx)
    for k in range(model.K):
        rv = multivariate_normal(model.means[k], model.covs[k])
        Z += model.weights[k] * rv.pdf(pos)
    
    plt.contour(xx, yy, Z, levels=15, cmap='magma')
    plt.scatter(model.means[:,0], model.means[:,1], 
              c='red', s=50, edgecolor='black')
    plt.title('Подобранные гауссианы')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Пример использования
# Генерация синтетических данных
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)

# Создание "истинной" гауссовой смеси (для примера)
true_means = [[-2, -2], [2, 2], [0, 0]]
true_covs = [ [[1, 0.5], [0.5, 1]], 
            [[1, -0.7], [-0.7, 1]], 
            [[0.5, 0], [0, 0.5]] ]
W = np.zeros_like(X)
for m, c in zip(true_means, true_covs):
    Z = multivariate_normal.pdf(np.dstack([X,Y]), mean=m, cov=c)
    W += Z
W /= W.sum()

# Создание выборки
samples = create_samples_from_weights(X, Y, W)

# Обучение модели
model = GMM2DEM(n_components=3)
model.fit(samples)

# Визуализация
visualize_results(X, Y, W, model)
