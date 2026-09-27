use tch::{Device, Kind, Tensor};

/// Линейный (полносвязный) слой с весами и смещением.
/// Инициализация весов – Xavier (равномерное распределение).
pub struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    /// Создаёт новый линейный слой.
    /// `in_features` – размер входа, `out_features` – размер выхода.
    pub fn new(in_features: i64, out_features: i64, device: Device) -> Self {
        // Xavier uniform: bound = sqrt(6 / (fan_in + fan_out))
        let bound = (6.0 / (in_features + out_features) as f64).sqrt();

        // Создаем [0, 1], масштабируем до [-bound, bound]
        let weight =
            (Tensor::rand([out_features, in_features], (Kind::Float, device)) * 2.0 * bound)
                - bound;

        let bias = Tensor::zeros([out_features], (Kind::Float, device));

        Self { weight, bias }
    }

    /// Прямой проход: y = x * W^T + b
    /// `x` имеет форму [batch_size, in_features]
    /// Возвращает тензор формы [batch_size, out_features]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.weight.transpose(0, 1)) + &self.bias
    }

    /// Обратный проход: вычисляет градиенты по входу, весам и смещению.
    /// `grad_output` – градиент по выходу слоя (форма [batch_size, out_features])
    /// `input` – вход, который был подан на прямой проход (форма [batch_size, in_features])
    /// Возвращает (grad_input, grad_weight, grad_bias).
    pub fn backward(&self, grad_output: &Tensor, input: &Tensor) -> (Tensor, Tensor, Tensor) {
        // grad_input = grad_output * W
        let grad_input = grad_output.matmul(&self.weight);
        // grad_weight = (grad_output^T) * input
        let grad_weight = grad_output.transpose(0, 1).matmul(input);
        // grad_bias = сумма по батчу от grad_output
        let grad_bias = grad_output.sum_dim_intlist(Some(&[0][..]), false, grad_output.kind());
        (grad_input, grad_weight, grad_bias)
    }

    /// Возвращает ссылки на параметры (веса и смещение).
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.shallow_clone(), self.bias.shallow_clone()]
    }

    /// Возвращает изменяемые ссылки на параметры.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weight, &mut self.bias]
    }
}

/// Тип функции активации.
#[derive(Clone, Copy)]
pub enum Activation {
    ReLU,
    Identity,
}

impl Activation {
    /// Применяет активацию к тензору.
    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => x.relu(),
            Activation::Identity => x.shallow_clone(),
        }
    }

    /// Вычисляет градиент активации по её входу.
    /// `grad_output` – градиент по выходу активации.
    /// `input` – вход активации (до применения).
    pub fn backward(&self, grad_output: &Tensor, input: &Tensor) -> Tensor {
        match self {
            Activation::ReLU => {
                // d(ReLU)/dx = 1, если x > 0, иначе 0.
                let mask = input.gt(0.0).to_kind(Kind::Float);
                grad_output * &mask
            }
            Activation::Identity => grad_output.shallow_clone(),
        }
    }
}

/// Многослойный перцептрон (перцептрон Румельхарта).
/// Состоит из чередующихся линейных слоёв и функций активации.
pub struct Perceptron {
    layers: Vec<Linear>,
    activations: Vec<Activation>,
}

impl Perceptron {
    pub fn new(layer_sizes: &[i64], activations: Vec<Activation>, device: Device) -> Self {
        assert_eq!(
            layer_sizes.len(),
            activations.len() + 1,
            "Количество слоёв должно быть на 1 больше количества активаций"
        );
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            layers.push(Linear::new(layer_sizes[i], layer_sizes[i + 1], device));
        }
        Self {
            layers,
            activations,
        }
    }

    /// Прямой проход. Возвращает (выход, кэш).
    /// Кэш – вектор кортежей (input_linear, pre_act) для каждого слоя.
    /// `input_linear` – вход линейного слоя (выход предыдущей активации или исходный x)
    /// `pre_act` – выход линейного слоя до активации.
    pub fn forward(&self, x: &Tensor) -> (Tensor, Vec<(Tensor, Tensor)>) {
        let mut cache = Vec::with_capacity(self.layers.len());
        let mut current = x.shallow_clone();
        for (i, layer) in self.layers.iter().enumerate() {
            let input_linear = current.shallow_clone();
            let pre_act = layer.forward(&input_linear);
            cache.push((input_linear, pre_act.shallow_clone()));
            current = self.activations[i].forward(&pre_act);
        }
        (current, cache)
    }

    /// Обратный проход.
    /// `grad_output` – градиент по выходу сети.
    /// `cache` – кэш, полученный из forward.
    /// Возвращает вектор (grad_weight, grad_bias) для каждого слоя.
    pub fn backward(
        &self,
        grad_output: &Tensor,
        cache: &[(Tensor, Tensor)],
    ) -> Vec<(Tensor, Tensor)> {
        let mut grads = Vec::with_capacity(self.layers.len());
        let mut grad = grad_output.shallow_clone();
        for i in (0..self.layers.len()).rev() {
            let (input_linear, pre_act) = &cache[i];
            // Градиент через активацию: на вход активации подан pre_act
            let grad_act = self.activations[i].backward(&grad, pre_act);
            // Градиент через линейный слой
            let (grad_input, grad_weight, grad_bias) =
                self.layers[i].backward(&grad_act, input_linear);
            grad = grad_input;
            grads.push((grad_weight, grad_bias));
        }
        grads.reverse(); // чтобы порядок соответствовал порядку слоёв (от первого к последнему)
        grads
    }

    /// Возвращает все параметры модели (веса и смещения) в виде плоского вектора
    /// для передачи оптимизатору.
    pub fn parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| layer.parameters())
            .collect()
    }

    /// Возвращает изменяемые ссылки на параметры.
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        self.layers
            .iter_mut()
            .flat_map(|layer| layer.parameters_mut())
            .collect()
    }
}
