mod data;
mod model;
mod optimizer;
mod train;

use anyhow::Result;
use prettytable::{Table, row};
use std::{f64, path::Path};
use tch::{Device, Kind, Tensor};

// Гиперпараметры
const LEARNING_RATE: f64 = 0.001;
const MOMENTUM: f64 = 0.85;
const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 100;

fn main() -> Result<()> {
    // 1. Загрузка данных
    println!("Загрузка тренировочных данных...");
    let train_root = Path::new("data/train");
    let (train_features, train_labels) = data::load_dataset(train_root)?;
    println!(
        "Загружено {} тренировочных изображений",
        train_features.len()
    );

    println!("Загрузка тестовых данных...");
    let test_root = Path::new("data/test");
    let (test_features, test_labels) = data::load_dataset(test_root)?;
    println!("Загружено {} тестовых изображений", test_features.len());

    // Определяем количество классов (уникальные метки)
    let num_classes = *train_labels.iter().max().unwrap() + 1;
    println!("Количество классов: {}", num_classes);

    let mut label_counts = vec![0; num_classes];
    for &l in &train_labels {
        label_counts[l] += 1;
    }
    println!(
        "Распределение меток в тренировочном наборе: {:?}",
        label_counts
    );

    // 2. Преобразование в тензоры
    let device = Device::Mps;

    let train_features_tensor = features_to_tensor(&train_features, device)?;
    let train_labels_tensor = labels_to_tensor(&train_labels, device)?;
    let test_features_tensor = features_to_tensor(&test_features, device)?;
    let test_labels_tensor = labels_to_tensor(&test_labels, device)?;

    // 3. Нормализация тренировочных и тестовых данных
    let mean = train_features_tensor.mean(Kind::Float);
    let std = train_features_tensor.std(true);
    println!("Mean: {:?}, Std: {:?}", mean, std);
    let train_features_tensor = (train_features_tensor - &mean) / (&std + 1e-7);
    let test_features_tensor = (test_features_tensor - &mean) / (&std + 1e-7);

    // 4. Архитектура модели
    let input_dim = data::INPUT_DIM as i64;
    let hidden1 = 1024;
    let hidden2 = 512;
    let layer_sizes = &[input_dim, hidden1, hidden2, num_classes as i64];
    let activations = vec![
        model::Activation::ReLU,
        model::Activation::ReLU,
        model::Activation::ReLU,
    ];
    let mut model = model::Perceptron::new(layer_sizes, activations, device);
    println!(
        "Модель создана: {:?} -> {} -> {} -> {}",
        input_dim, hidden1, hidden2, num_classes
    );

    // 5. Оптимизатор
    let params = model.parameters();
    let mut optimizer = optimizer::NesterovOptimizer::new(&params, LEARNING_RATE, MOMENTUM);

    // 6. Цикл обучения
    let train_data = (train_features_tensor, train_labels_tensor);
    let test_data = (test_features_tensor, test_labels_tensor);

    println!("Начинаем обучение на {} эпох...", EPOCHS);
    let mut last_test_losses = [f64::INFINITY; 3];
    for epoch in 1..=EPOCHS {
        // Обучение на одну эпоху
        let (train_loss, train_acc) =
            train::train_epoch(&mut model, &mut optimizer, &train_data, BATCH_SIZE);
        // Валидация на тестовых данных
        let (test_loss, test_acc) = train::validate(&model, &test_data, BATCH_SIZE);

        println!(
            "Эпоха {:3}/{} | Train Loss: {:.4}, Train Acc: {:.2}% | Test Loss: {:.4}, Test Acc: {:.2}%",
            epoch,
            EPOCHS,
            train_loss,
            train_acc * 100.0,
            test_loss,
            test_acc * 100.0
        );

        last_test_losses = [last_test_losses[1], last_test_losses[2], test_loss];
        if test_loss >= last_test_losses.iter().sum::<f64>() / last_test_losses.len() as f64 {
            break;
        }
    }

    println!("Обучение завершено.");

    let (preds, targets) = train::collect_predictions(&model, &test_data, BATCH_SIZE);
    let cm = train::confusion_matrix(&preds, &targets, num_classes as i64);
    let (precision, recall, f1) = train::per_class_metrics_from_cm(&cm);

    // Получаем названия классов (если у вас есть список имён папок)
    // Допустим, вы сохранили имена классов при загрузке. Для примера просто по индексам:
    let class_dirs = data::load_class_dirs(test_root)?;
    let class_names: Vec<String> = class_dirs
        .iter()
        .map(|p| p.file_name().unwrap().to_string_lossy().to_string())
        .collect();

    // Вывод в виде таблицы
    let mut table = Table::new();
    table.add_row(row!["Class", "Precision", "Recall", "F1"]);

    for i in 0..num_classes {
        table.add_row(row![
            &class_names[i],
            format!("{:.4}", precision[i]),
            format!("{:.4}", recall[i]),
            format!("{:.4}", f1[i]),
        ]);
    }

    table.printstd();

    Ok(())
}

/// Преобразует Vec<Vec<f32>> в тензор размером [n_samples, input_dim].
fn features_to_tensor(features: &[Vec<f32>], device: Device) -> Result<Tensor> {
    if features.is_empty() {
        anyhow::bail!("Нет данных для преобразования в тензор");
    }
    let n_samples = features.len();
    let input_dim = features[0].len();
    let mut flat_data = Vec::with_capacity(n_samples * input_dim);
    for sample in features {
        flat_data.extend_from_slice(sample);
    }
    let tensor = Tensor::from_slice(&flat_data)
        .reshape([n_samples as i64, input_dim as i64])
        .to_device(device);
    Ok(tensor)
}

/// Преобразует Vec<usize> в тензор int64 размером [n_samples].
fn labels_to_tensor(labels: &[usize], device: Device) -> Result<Tensor> {
    let labels_i64: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
    let tensor = Tensor::from_slice(&labels_i64)
        .to_device(device)
        .to_kind(Kind::Int64);
    Ok(tensor)
}
