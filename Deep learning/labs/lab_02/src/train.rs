use crate::model::Perceptron;
use crate::optimizer::NesterovOptimizer;
use tch::{IndexOp, Kind, Tensor};

/// Вычисляет кросс-энтропийную потерю (logits, targets).
/// logits: [batch_size, num_classes]
/// targets: [batch_size] (int64)
/// Возвращает скалярный тензор (среднее по батчу).
pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    // log_softmax для численной стабильности
    let log_probs = logits.log_softmax(1, Kind::Float);
    // Индексируем логарифмические вероятности для правильных классов
    -log_probs
        .gather(1, &targets.unsqueeze(1), false)
        .mean(Kind::Float)
}

/// Вычисляет градиент кросс-энтропийной потери по logits.
/// Возвращает градиент той же формы, что и logits.
pub fn cross_entropy_grad(logits: &Tensor, targets: &Tensor) -> Tensor {
    // softmax по классам
    let probs = logits.softmax(1, Kind::Float);
    // one-hot кодирование меток
    let num_classes = logits.size()[1];
    let one_hot = Tensor::one_hot(targets, num_classes).to_kind(Kind::Float);
    // Градиент: p_i - y_i
    &probs - one_hot
}

/// Вычисляет точность классификации.
/// logits: [batch_size, num_classes]
/// targets: [batch_size] (int64)
/// Возвращает долю правильных ответов (f64).
pub fn accuracy(logits: &Tensor, targets: &Tensor) -> f64 {
    let predictions = logits.argmax(1, false);
    let correct = predictions.eq_tensor(targets).sum(Kind::Int64);
    let total = targets.size()[0] as f64;
    correct.double_value(&[]) / total
}

/// Обучает модель одну эпоху на тренировочных данных.
/// Возвращает (средняя потеря, точность) за эпоху.
pub fn train_epoch(
    model: &mut Perceptron,
    optimizer: &mut NesterovOptimizer,
    data: &(Tensor, Tensor), // (features, labels)
    batch_size: usize,
) -> (f64, f64) {
    let (features, labels) = data;
    let num_samples = features.size()[0] as usize;
    let num_batches = num_samples.div_ceil(batch_size);
    let mut total_loss = 0.0;
    let mut total_correct = 0;
    let mut total_processed = 0;

    // Итерируем по батчам
    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(num_samples);
        let batch_x = features.narrow(0, start as i64, (end - start) as i64);
        let batch_y = labels.narrow(0, start as i64, (end - start) as i64);

        // 1. Прямой проход
        let (logits, cache) = model.forward(&batch_x);

        // 2. Вычисление значения функции потерь и точности
        let loss = cross_entropy_loss(&logits, &batch_y);
        total_loss += loss.double_value(&[]) * (end - start) as f64;
        let acc = accuracy(&logits, &batch_y);
        total_correct += (acc * (end - start) as f64) as i64;
        total_processed += end - start;

        // 3. Градиент функции потерь по выходам
        let grad_logits = cross_entropy_grad(&logits, &batch_y);
        // 4. Обратный проход: получаем градиенты весов и смещений для каждого слоя
        let layer_grads = model.backward(&grad_logits, &cache);
        // 5. Усреднение градиентов по батчу (разделить на размер батча)
        let batch_size_f = (end - start) as f64;
        let avg_grads: Vec<Tensor> = layer_grads
            .iter()
            .flat_map(|(gw, gb)| vec![gw / batch_size_f, gb / batch_size_f])
            .collect();
        for (i, grad) in avg_grads.iter().enumerate() {
            let nan_count = grad.isnan().any().double_value(&[]);
            let inf_count = grad.isinf().any().double_value(&[]);
            if nan_count > 0.0 || inf_count > 0.0 {
                println!("WARNING: NaN/inf в градиенте {i}");
            }
        }
        // 6. Получить изменяемые ссылки на параметры модели
        let mut params = model.parameters_mut();
        // 7. Шаг оптимизатора Нестерова
        optimizer.step(&mut params, &avg_grads);
    }

    let avg_loss = total_loss / total_processed as f64;
    let avg_acc = total_correct as f64 / total_processed as f64;
    (avg_loss, avg_acc)
}

/// Валидация модели на тестовых данных (без обновления весов).
/// Возвращает (средняя потеря, точность).
pub fn validate(
    model: &Perceptron,
    data: &(Tensor, Tensor), // (features, labels)
    batch_size: usize,
) -> (f64, f64) {
    let (features, labels) = data;
    let num_samples = features.size()[0] as usize;
    let num_batches = num_samples.div_ceil(batch_size);
    let mut total_loss = 0.0;
    let mut total_correct = 0;
    let mut total_processed = 0;

    for batch_idx in 0..num_batches {
        let start = batch_idx * batch_size;
        let end = (start + batch_size).min(num_samples);
        let batch_x = features.narrow(0, start as i64, (end - start) as i64);
        let batch_y = labels.narrow(0, start as i64, (end - start) as i64);

        let (logits, _cache) = model.forward(&batch_x);
        let loss = cross_entropy_loss(&logits, &batch_y);
        total_loss += loss.double_value(&[]) * (end - start) as f64;
        let acc = accuracy(&logits, &batch_y);
        total_correct += (acc * (end - start) as f64) as i64;
        total_processed += end - start;
    }

    let avg_loss = total_loss / total_processed as f64;
    let avg_acc = total_correct as f64 / total_processed as f64;
    (avg_loss, avg_acc)
}

/// Возвращает (предсказания, истинные метки) для всего набора данных.
pub fn collect_predictions(
    model: &Perceptron,
    data: &(Tensor, Tensor),
    batch_size: usize,
) -> (Tensor, Tensor) {
    let (features, labels) = data;
    let num_samples = features.size()[0] as usize;
    let device = features.device();
    let mut all_preds = Vec::new();
    let mut all_labels = Vec::new();

    for start in (0..num_samples).step_by(batch_size) {
        let end = (start + batch_size).min(num_samples);
        let batch_x = features.narrow(0, start as i64, (end - start) as i64);
        let batch_y = labels.narrow(0, start as i64, (end - start) as i64);
        let (logits, _) = model.forward(&batch_x);
        let preds = logits.argmax(1, false);
        for i in 0..(end - start) {
            all_preds.push(preds.int64_value(&[i as i64]));
            all_labels.push(batch_y.int64_value(&[i as i64]));
        }
    }

    let pred_tensor = Tensor::from_slice(&all_preds).to_device(device);
    let label_tensor = Tensor::from_slice(&all_labels).to_device(device);
    (pred_tensor, label_tensor)
}

/// Строит матрицу ошибок (confusion matrix) размера [num_classes, num_classes].
/// `preds` – тензор предсказанных классов (int64) формы [n_samples].
/// `targets` – тензор истинных классов (int64) формы [n_samples].
pub fn confusion_matrix(preds: &Tensor, targets: &Tensor, num_classes: i64) -> Tensor {
    let cm = Tensor::zeros([num_classes, num_classes], (Kind::Int64, preds.device()));

    // Подготавливаем индексы: список тензоров [строки, столбцы]
    let indices = [Some(targets.shallow_clone()), Some(preds.shallow_clone())];

    // Создаем тензор единиц той же длины, что и входные данные
    let ones = Tensor::ones([preds.size()[0]], (Kind::Int64, preds.device()));

    // accumulate: true прибавит +1 к текущему значению в ячейке (t, p)
    cm.index_put(&indices, &ones, true)
}

/// Возвращает векторы precision, recall, f1 для каждого класса.
pub fn per_class_metrics_from_cm(cm: &Tensor) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let num_classes = cm.size()[0] as usize;
    let mut precision = vec![0.0; num_classes];
    let mut recall = vec![0.0; num_classes];
    let mut f1 = vec![0.0; num_classes];

    for i in 0..num_classes {
        let tp = cm.i((i as i64, i as i64)).double_value(&[]);
        let fp = cm.narrow(0, i as i64, 1).sum(Kind::Float).double_value(&[]) - tp;
        let fn_ = cm.narrow(1, i as i64, 1).sum(Kind::Float).double_value(&[]) - tp;

        precision[i] = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
        recall[i] = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { 0.0 };
        f1[i] = if precision[i] + recall[i] > 0.0 {
            2.0 * precision[i] * recall[i] / (precision[i] + recall[i])
        } else {
            0.0
        };
    }
    (precision, recall, f1)
}
