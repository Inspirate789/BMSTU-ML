// perceptron.rs
// Реализация перцептрона Розенблатта с тремя слоями (S, A, R)

use rand::{Rng, RngExt};
use rayon::prelude::*;

// ------------------------------------------------------------
// Ассоциативный слой (A) со случайными фиксированными весами
// ------------------------------------------------------------
struct AssociativeLayer {
    weights: Vec<Vec<f32>>,
    threshold: f32,
}

impl AssociativeLayer {
    pub fn new(s_input_size: usize, num_a: usize) -> Self {
        let mut rng = rand::rng();
        let weights = (0..num_a)
            .map(|_| {
                (0..s_input_size)
                    .map(|_| rng.random_range(-1.0..1.0))
                    .collect()
            })
            .collect();
        AssociativeLayer {
            weights,
            threshold: 0.0,
        }
    }

    pub fn forward(&self, s_input: &[f32]) -> Vec<f32> {
        self.weights
            .iter()
            .map(|a_weights| {
                let sum: f32 = a_weights.iter().zip(s_input).map(|(w, x)| w * x).sum();
                if sum >= self.threshold { 1.0 } else { 0.0 }
            })
            .collect()
    }
}

// ------------------------------------------------------------
// Реагирующий слой (R) - обучаемый перцептрон
// ------------------------------------------------------------
struct ReactiveLayer {
    weights: Vec<f32>,
    bias: f32,
    lr: f32,
}

impl ReactiveLayer {
    pub fn new(num_a: usize, learning_rate: f32) -> Self {
        let mut rng = rand::rng();
        let weights = (0..num_a).map(|_| rng.random_range(-0.5..0.5)).collect();
        ReactiveLayer {
            weights,
            bias: 0.0,
            lr: learning_rate,
        }
    }

    pub fn predict(&self, a_output: &[f32]) -> i32 {
        let sum: f32 = self.weights.iter().zip(a_output).map(|(w, a)| w * a).sum();
        if sum + self.bias >= 0.0 { 1 } else { 0 }
    }

    pub fn raw_score(&self, a_output: &[f32]) -> f32 {
        self.weights
            .iter()
            .zip(a_output)
            .map(|(w, a)| w * a)
            .sum::<f32>()
            + self.bias
    }

    pub fn train_on_sample(&mut self, a_output: &[f32], target: i32) -> bool {
        let pred = self.predict(a_output);
        if pred == target {
            return false;
        }
        let error = (target - pred) as f32;
        for i in 0..self.weights.len() {
            self.weights[i] += self.lr * error * a_output[i];
        }
        self.bias += self.lr * error;
        true
    }
}

// ------------------------------------------------------------
// Полный перцептрон Розенблатта (S - A - R)
// ------------------------------------------------------------
pub struct RosenblattPerceptron {
    associative: AssociativeLayer,
    reactive: ReactiveLayer,
}

impl RosenblattPerceptron {
    pub fn new(s_size: usize, num_a: usize, learning_rate: f32) -> Self {
        RosenblattPerceptron {
            associative: AssociativeLayer::new(s_size, num_a),
            reactive: ReactiveLayer::new(num_a, learning_rate),
        }
    }

    pub fn predict(&self, s_input: &[f32]) -> i32 {
        let a_out = self.associative.forward(s_input);
        self.reactive.predict(&a_out)
    }

    pub fn raw_score(&self, s_input: &[f32]) -> f32 {
        let a_out = self.associative.forward(s_input);
        self.reactive.raw_score(&a_out)
    }

    pub fn train_on_sample(&mut self, s_input: &[f32], target: i32) -> bool {
        let a_out = self.associative.forward(s_input);
        self.reactive.train_on_sample(&a_out, target)
    }

    pub fn train_epoch(&mut self, data: &[(&[f32], i32)]) -> usize {
        let mut errors = 0;
        for (s_input, target) in data {
            if self.train_on_sample(s_input, *target) {
                errors += 1;
            }
        }
        errors
    }
}

// ------------------------------------------------------------
// Мультиклассовый классификатор (one-vs-all) на перцептронах Розенблатта
// ------------------------------------------------------------
pub struct MultiClassRosenblatt {
    classifiers: Vec<RosenblattPerceptron>,
    num_classes: usize,
}

impl MultiClassRosenblatt {
    pub fn new(s_size: usize, num_a: usize, num_classes: usize, learning_rate: f32) -> Self {
        let classifiers = (0..num_classes)
            .map(|_| RosenblattPerceptron::new(s_size, num_a, learning_rate))
            .collect();
        MultiClassRosenblatt {
            classifiers,
            num_classes,
        }
    }

    /// Обучает все бинарные классификаторы параллельно.
    pub fn train(&mut self, samples: &[(Vec<f32>, usize)], epochs: usize) -> Vec<usize> {
        let mut error_history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let total_errors: usize = self
                .classifiers
                .par_iter_mut()
                .enumerate()
                .map(|(c, clf)| {
                    // Для каждого класса формируем бинарные метки
                    let data_for_class: Vec<(&[f32], i32)> = samples
                        .iter()
                        .map(|(pixels, label)| {
                            let target = if *label == c { 1 } else { 0 };
                            (pixels.as_slice(), target)
                        })
                        .collect();
                    clf.train_epoch(&data_for_class)
                })
                .sum();
            error_history.push(total_errors);
        }
        error_history
    }

    /// Предсказание класса по принципу максимальной сырой оценки (argmax).
    pub fn predict(&self, pixels: &[f32]) -> usize {
        (0..self.num_classes)
            .max_by(|&a, &b| {
                let score_a = self.classifiers[a].raw_score(pixels);
                let score_b = self.classifiers[b].raw_score(pixels);
                score_a.partial_cmp(&score_b).unwrap()
            })
            .unwrap()
    }

    /// Точность на тестовой выборке.
    pub fn evaluate(&self, samples: &[(Vec<f32>, usize)]) -> f64 {
        let correct = samples
            .iter()
            .filter(|(pixels, true_label)| self.predict(pixels) == *true_label)
            .count();
        correct as f64 / samples.len() as f64
    }
}

// ------------------------------------------------------------
// Тесты
// ------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_and_problem() {
        let mut p = RosenblattPerceptron::new(2, 10, 0.1);
        let data = vec![
            (vec![0.0, 0.0], 0),
            (vec![0.0, 1.0], 0),
            (vec![1.0, 0.0], 0),
            (vec![1.0, 1.0], 1),
        ];
        let data_ref: Vec<(&[f32], i32)> = data.iter().map(|(v, l)| (v.as_slice(), *l)).collect();
        for _ in 0..20 {
            p.train_epoch(&data_ref);
        }
        for (v, expected) in data {
            assert_eq!(p.predict(&v), expected);
        }
    }

    #[test]
    fn test_multiclass() {
        let samples = vec![
            (vec![0.0, 0.0], 0),
            (vec![0.5, 0.5], 0),
            (vec![1.0, 0.0], 1),
            (vec![1.2, 0.2], 1),
            (vec![0.0, 1.0], 2),
            (vec![0.2, 1.1], 2),
        ];
        let mut classifier = MultiClassRosenblatt::new(2, 5, 3, 0.1);
        classifier.train(&samples, 30);
        let acc = classifier.evaluate(&samples);
        assert!(acc > 0.8);
    }
}
