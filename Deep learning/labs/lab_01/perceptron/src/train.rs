use std::collections::HashSet;

// train.rs
use crate::model::RosenblattPerceptron;
use anyhow::Result;
use tch::{Kind, Tensor};

/// Преобразует метки в one-hot кодировку.
/// Возвращает тензор [batch, num_classes] с 1 для правильного класса, иначе 0.
fn one_hot(labels: &Tensor, num_classes: i64) -> Tensor {
    let batch_size = labels.size()[0];
    let zeros = Tensor::zeros([batch_size, num_classes], (Kind::Float, labels.device()));
    zeros.scatter(
        1,
        &labels.unsqueeze(1),
        &Tensor::ones([batch_size, 1], (Kind::Float, labels.device())),
    )
}

/// Вычисляет точность и среднее значение функции потерь (бинарная перекрёстная энтропия для многоклассового случая).
/// Для классического перцептрона loss считается как сумма квадратов ошибок или просто как доля ошибок.
/// Здесь для мониторинга используем cross-entropy (логиты -> softmax).
pub fn evaluate(model: &RosenblattPerceptron, features: &Tensor, labels: &Tensor) -> (f64, f64) {
    let logits = model.forward(features);
    let loss = logits.cross_entropy_for_logits(labels);
    let predicted = logits.argmax(-1, false);
    let correct = predicted.eq_tensor(labels).sum(Kind::Float);
    let total = labels.size()[0] as f64;
    let accuracy = correct.double_value(&[]) / total;
    let loss_value = loss.double_value(&[]);
    (accuracy, loss_value)
}

/// Обучает модель на одной эпохе с использованием классического правила коррекции ошибок.
/// Возвращает среднюю долю ошибок на эпоху (чтобы отслеживать прогресс).
pub fn train_epoch(
    model: &mut RosenblattPerceptron,
    features: &Tensor,
    labels: &Tensor,
    batch_size: usize,
    learning_rate: f64,
) -> f64 {
    let n_samples = features.size()[0] as usize;
    let num_classes = model.r_weights().size()[0];
    let mut total_errors = 0.0;

    for start in (0..n_samples).step_by(batch_size) {
        let end = (start + batch_size).min(n_samples);
        let batch_x = features.narrow(0, start as i64, (end - start) as i64);
        let batch_y = labels.narrow(0, start as i64, (end - start) as i64);
        let batch_size_actual = (end - start) as i64;

        // 1. Прямой проход: выходы A-слоя и логиты R-слоя
        let a_out = model.a_output(&batch_x); // [batch, num_a]
        let logits = model.forward(&batch_x); // [batch, num_classes]

        // 2. Пороговое решение (классический перцептрон): y = 1 если logit > 0, иначе 0
        let y = logits.gt(0.0).to_kind(Kind::Float); // [batch, num_classes]

        // 3. Желаемые выходы (one-hot)
        let d = one_hot(&batch_y, num_classes); // [batch, num_classes]

        // 4. Ошибка: e = d - y
        let e = &d - &y; // [batch, num_classes]

        // 5. Количество ошибок в батче (для статистики)
        let errors_in_batch = e.abs().sum(Kind::Float).double_value(&[]);
        total_errors += errors_in_batch;

        // 6. Обновление весов R-слоя по правилу Δv = η * e_j * a_i
        //    Для весов: delta_weights = η * (e^T @ a_out)
        let delta_weights = (e.transpose(0, 1).matmul(&a_out)
            * (learning_rate / batch_size_actual as f64))
            .detach();
        //    Для смещений: delta_bias = η * sum(e по батчу)
        let delta_bias = (e.sum_dim_intlist(0, false, Kind::Float)
            * (learning_rate / batch_size_actual as f64))
            .detach();

        model.update_r_weights(&delta_weights);
        model.update_r_bias(&delta_bias);
    }

    // Средняя ошибка на примере
    total_errors / n_samples as f64
}

/// Основной цикл обучения с ранней остановкой и pruning.
pub fn train_with_pruning(
    model: &mut RosenblattPerceptron,
    train_features: &Tensor,
    train_labels: &Tensor,
    test_features: &Tensor,
    test_labels: &Tensor,
    learning_rate: f64,
    batch_size: usize,
    max_epochs: usize,
    patience: usize,
    dead_threshold: f32,
    corr_threshold: f32,
) -> Result<()> {
    let mut best_loss = f64::INFINITY;
    let mut best_acc = f64::INFINITY;
    let mut epochs_no_improve = 0;
    let mut epoch = 0;

    println!("Начало обучения классического перцептрона Розенблатта");
    println!("Количество A-нейронов: {}", model.num_a_neurons());
    println!("Скорость обучения: {}", learning_rate);
    println!(
        "{:>6} {:>12} {:>12} {:>12} {:>12}",
        "Эпоха", "Train err", "Test loss", "Test acc", "A нейроны"
    );

    while epoch < max_epochs {
        // Обучаем одну эпоху (используем правило коррекции ошибок)
        let train_err = train_epoch(
            model,
            train_features,
            train_labels,
            batch_size,
            learning_rate,
        );
        let (test_acc, test_loss) = evaluate(model, test_features, test_labels);

        println!(
            "{:6} {:12.6} {:12.6} {:11.4}% {:12}",
            epoch + 1,
            train_err,
            test_loss,
            test_acc * 100.0,
            model.num_a_neurons()
        );

        // Проверка улучшения по loss на тесте
        if test_loss < best_loss - 1e-7 {
            best_loss = test_loss;
            best_acc = test_acc;
            epochs_no_improve = 0;
        } else {
            epochs_no_improve += 1;
        }

        // Если нет улучшения достаточно долго, применяем pruning
        if epochs_no_improve >= patience {
            println!(
                "Нет улучшения в течение {} эпох, запускаем pruning...",
                patience
            );

            let mut removed = HashSet::new();

            model.prune_dead(train_features, dead_threshold, &mut removed);
            let dead_removed = removed.len();
            println!("Удалено {} мёртвых A-нейронов", removed.len());

            model.prune_correlated(train_features, corr_threshold, &mut removed);
            println!(
                "Удалено {} коррелирующих A-нейронов",
                removed.len() - dead_removed
            );

            if !model.remove_neurons(&removed) {
                println!("Ни один нейрон не удалён, дальнейшее обучение бесполезно. Остановка.");
                break;
            }

            println!("Удалено {} A-нейронов", removed.len());
            if model.num_a_neurons() == 0 {
                anyhow::bail!("Все A-нейроны удалены, обучение невозможно.");
            }

            let (test_acc, test_loss) = evaluate(model, test_features, test_labels);

            println!(
                "{:6} {:12.6} {:12.6} {:11.4}% {:12}",
                epoch + 1,
                train_err,
                test_loss,
                test_acc * 100.0,
                model.num_a_neurons()
            );

            if test_acc > best_acc {
                println!("Прореживание повысило точность.");
                break;
            }

            // Сбрасываем счётчик и сохраняем текущий loss как лучший
            epochs_no_improve = 0;
            best_loss = test_loss;
            best_acc = test_acc;
        }

        epoch += 1;
    }

    let (final_acc, final_loss) = evaluate(model, test_features, test_labels);
    println!("\nОбучение завершено.");
    println!("Финальная точность на тесте: {:.4}%", final_acc * 100.0);
    println!("Финальная потеря на тесте: {:.6}", final_loss);
    println!(
        "Оставшееся количество A-нейронов: {}",
        model.num_a_neurons()
    );

    Ok(())
}
