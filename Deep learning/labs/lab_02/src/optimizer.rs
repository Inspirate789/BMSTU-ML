use tch::Tensor;

/// Оптимизатор Нестерова (Nesterov Accelerated Gradient).
pub struct NesterovOptimizer {
    learning_rate: f64,
    momentum: f64,           // коэффициент импульса
    velocities: Vec<Tensor>, // скорость для каждого параметра
}

impl NesterovOptimizer {
    /// Создаёт новый оптимизатор.
    /// `parameters` – список параметров модели (веса и смещения).
    /// `learning_rate` – скорость обучения.
    /// `momentum` – коэффициент импульса.
    pub fn new(parameters: &[Tensor], learning_rate: f64, momentum: f64) -> Self {
        let velocities = parameters.iter().map(Tensor::zeros_like).collect();
        Self {
            learning_rate,
            momentum,
            velocities,
        }
    }

    // Выполняет один шаг оптимизации.
    // `parameters` – изменяемые ссылки на параметры (должны соответствовать порядку при создании).
    // `gradients` – градиенты для каждого параметра (такой же порядок и форма).
    // pub fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &[Tensor]) {
    //     assert_eq!(parameters.len(), self.velocities.len());
    //     assert_eq!(gradients.len(), self.velocities.len());

    //     for i in 0..parameters.len() {
    //         let param = &mut *parameters[i];
    //         let grad = &gradients[i];
    //         let v = &mut self.velocities[i];

    //         // NAG update:
    //         // buf = momentum * v - learning_rate * grad
    //         // param = param + momentum * buf - learning_rate * grad
    //         // v = buf
    //         let buf = (&*v * self.momentum) - (grad * self.learning_rate);
    //         *param = &*param + (&buf * self.momentum) - (grad * self.learning_rate);
    //         *v = buf;
    //     }
    // }

    pub fn step(&mut self, parameters: &mut [&mut Tensor], gradients: &[Tensor]) {
        assert_eq!(parameters.len(), self.velocities.len());
        assert_eq!(gradients.len(), self.velocities.len());

        for i in 0..parameters.len() {
            let param = &mut *parameters[i];
            let grad = &gradients[i];
            let v = &mut self.velocities[i];

            // NAG update (PyTorch style)
            *v = &*v * self.momentum - grad * self.learning_rate;
            *param = &*param + &*v * self.momentum - grad * self.learning_rate;
        }
    }
}
