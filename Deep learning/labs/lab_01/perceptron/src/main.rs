mod data;
mod model;
mod train;

use anyhow::Result;
use std::{f64, path::Path};
use tch::{Device, Kind, Tensor};

use crate::{data::INPUT_DIM, model::RosenblattPerceptron, train::train_with_pruning};

// Гиперпараметры
const LEARNING_RATE: f64 = 0.0001;
const BATCH_SIZE: usize = 64;
const EPOCHS: usize = 250;

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

    let mut model = RosenblattPerceptron::new(INPUT_DIM, INPUT_DIM * 8, num_classes, device);

    train_with_pruning(
        &mut model,
        &train_features_tensor,
        &train_labels_tensor,
        &test_features_tensor,
        &test_labels_tensor,
        LEARNING_RATE,
        BATCH_SIZE,
        EPOCHS,
        3,    // patience
        0.25, // dead_threshold (25% средней активации)
        0.7,  // corr_threshold (|корреляция| > 0.7)
    )?;

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
