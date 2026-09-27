use std::collections::HashSet;

// model.rs
use tch::{Device, Kind, Tensor};

/// Классический перцептрон Розенблатта с фиксированным A-слоем и обучаемым R-слоем.
pub struct RosenblattPerceptron {
    // A-слой (ассоциативный) – веса заморожены
    a_weights: Tensor, // [num_a, input_dim]
    a_bias: Tensor,    // [num_a]
    // R-слой (реагирующий) – обучаемые веса
    r_weights: Tensor, // [num_classes, num_a]
    r_bias: Tensor,    // [num_classes]
    // Метаданные
    num_a: usize,
    device: Device,
}

impl RosenblattPerceptron {
    pub fn new(input_dim: usize, num_a: usize, num_classes: usize, device: Device) -> Self {
        // Инициализация A-весов (Xavier-like нормальное распределение)
        let a_weights = Tensor::randn([num_a as i64, input_dim as i64], (Kind::Float, device))
            * (2.0 / (input_dim as f64)).sqrt();
        let a_bias = Tensor::zeros([num_a as i64], (Kind::Float, device));

        // Заморозить A-слой
        let a_weights = a_weights.set_requires_grad(false);
        let a_bias = a_bias.set_requires_grad(false);

        // Инициализация R-весов (малые случайные)
        let r_weights =
            Tensor::randn([num_classes as i64, num_a as i64], (Kind::Float, device)) * 0.01;
        let r_bias = Tensor::zeros([num_classes as i64], (Kind::Float, device));

        RosenblattPerceptron {
            a_weights,
            a_bias,
            r_weights,
            r_bias,
            num_a,
            device,
        }
    }

    /// Прямой проход. Возвращает логиты размера [batch, num_classes].
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let a_out = self.a_output(x);
        // Логиты = a_out * r_weights^T + r_bias
        a_out.matmul(&self.r_weights.transpose(0, 1)) + &self.r_bias
    }

    /// Возвращает выходы A-слоя (после сигмоида) размера [batch, num_a].
    pub fn a_output(&self, x: &Tensor) -> Tensor {
        let a_z = x.matmul(&self.a_weights.transpose(0, 1)) + &self.a_bias;
        a_z.sigmoid()
    }

    /// Обнаруживает «мёртвые» A-нейроны (средняя активация ниже порога)
    /// и добавляет их индексы в `removed`.
    pub fn prune_dead(&self, data: &Tensor, threshold: f32, removed: &mut HashSet<usize>) {
        let a_out = self.a_output(data);
        let mean_acts = a_out.mean_dim(0, false, Kind::Float);
        for i in 0..self.num_a {
            if mean_acts.double_value(&[i as i64]) <= threshold as f64 {
                removed.insert(i);
            }
        }
    }

    /// Обнаруживает коррелирующие A-нейроны (|corr| > threshold)
    /// и добавляет индексы подлежащих удалению в `removed`.
    /// Из каждой пары оставляет первый, остальные помечает на удаление.
    pub fn prune_correlated(&self, data: &Tensor, threshold: f32, removed: &mut HashSet<usize>) {
        let a_out = self.a_output(data);
        let n_samples = a_out.size()[0] as f64;
        if n_samples < 2.0 {
            return;
        }

        let mean = a_out.mean_dim(0, true, Kind::Float);
        let centered = a_out - mean;
        let cov = centered.transpose(0, 1).matmul(&centered) / (n_samples - 1.0);
        let std = cov.diag(0).sqrt();
        let corr = cov / (std.unsqueeze(1) * std.unsqueeze(0) + 1e-7);
        let corr_cpu = corr.to_device(Device::Cpu).to_kind(Kind::Float);
        let corr_data: Vec<f32> = corr_cpu.flatten(0, -1).try_into().unwrap();

        for i in 0..self.num_a {
            if removed.contains(&i) {
                continue;
            }
            let base = i * self.num_a;
            for j in (i + 1)..self.num_a {
                if removed.contains(&j) {
                    continue;
                }
                let corr_ij = corr_data[base + j].abs();
                if corr_ij > threshold {
                    removed.insert(j);
                }
            }
        }
    }

    /// Удаляет нейроны по списку индексов и перестраивает слои.
    pub fn remove_neurons(&mut self, removed: &HashSet<usize>) -> bool {
        if removed.is_empty() {
            return false;
        }
        let keep_indices: Vec<i64> = (0..self.num_a)
            .filter(|i| !removed.contains(i))
            .map(|i| i as i64)
            .collect();
        if keep_indices.len() == self.num_a {
            false
        } else {
            self.keep_neurons_by_indices(&keep_indices);
            true
        }
    }

    /// Внутренний метод: оставляет только нейроны с указанными индексами.
    fn keep_neurons_by_indices(&mut self, keep_indices: &[i64]) {
        let keep_idx_tensor = Tensor::from_slice(keep_indices).to_device(self.device);
        let new_a_weights = self.a_weights.index_select(0, &keep_idx_tensor);
        let new_a_bias = self.a_bias.index_select(0, &keep_idx_tensor);
        let new_r_weights = self.r_weights.index_select(1, &keep_idx_tensor);
        self.a_weights = new_a_weights.set_requires_grad(false);
        self.a_bias = new_a_bias.set_requires_grad(false);
        self.r_weights = new_r_weights;
        self.num_a = keep_indices.len();
    }

    /// Возвращает количество A-нейронов.
    pub fn num_a_neurons(&self) -> usize {
        self.num_a
    }

    /// Применить дельту к весам R-слоя.
    pub fn update_r_weights(&mut self, delta: &Tensor) {
        self.r_weights = (&self.r_weights + delta).detach();
        self.r_weights = self.r_weights.set_requires_grad(false);
    }

    /// Применить дельту к смещениям R-слоя.
    pub fn update_r_bias(&mut self, delta: &Tensor) {
        self.r_bias = (&self.r_bias + delta).detach();
        self.r_bias = self.r_bias.set_requires_grad(false);
    }

    /// Получить копию весов R-слоя (для отладки).
    pub fn r_weights(&self) -> Tensor {
        self.r_weights.shallow_clone()
    }
}
