mod data;
mod perceptron;

use anyhow::Result;
use perceptron::RosenblattPerceptron;
use std::path::Path;

fn main() -> Result<()> {
    // 1. Пути к данным (предполагаем, что запуск из корня проекта, где есть папка data)
    let train_root = Path::new("data/train");
    let test_root = Path::new("data/test");

    println!("Загрузка тренировочных данных...");
    let (train_features, train_labels) = data::load_dataset(train_root)?;
    println!(
        "Загружено {} тренировочных изображений",
        train_features.len()
    );

    println!("Загрузка тестовых данных...");
    let (test_features, test_labels) = data::load_dataset(test_root)?;
    println!("Загружено {} тестовых изображений", test_features.len());

    let target_class = 12; // бинарная классификация: 1 – этот класс, 0 – все остальные
    println!("Целевой класс: индекс {}", target_class);

    // Преобразуем метки в бинарные (1 для target_class, 0 иначе)
    let train_binary: Vec<i32> = train_labels
        .iter()
        .map(|&l| if l == target_class { 1 } else { 0 })
        .collect();
    let test_binary: Vec<i32> = test_labels
        .iter()
        .map(|&l| if l == target_class { 1 } else { 0 })
        .collect();

    // 3. Создаём перцептрон Розенблатта
    let input_size = data::INPUT_DIM; // 56*56*3 = 9408
    let num_a = 2000; // количество ассоциативных нейронов
    let learning_rate = 0.1;
    let epochs = 3;

    let mut perceptron = RosenblattPerceptron::new(input_size, num_a, learning_rate);
    println!("\nНачало обучения...");

    // Подготовка данных для обучения: срез признаков и бинарных меток
    let train_data: Vec<(&[f32], i32)> = train_features
        .iter()
        .zip(train_binary.iter())
        .map(|(feat, &label)| (feat.as_slice(), label))
        .collect();

    for epoch in 0..epochs {
        let errors = perceptron.train_epoch(&train_data);
        println!("Эпоха {:2}: ошибок = {}", epoch + 1, errors);
    }

    // 4. Оценка на тестовых данных
    println!("\nОценка на тестовой выборке...");
    let mut tp = 0; // True Positive
    let mut fp = 0; // False Positive
    let mut tn = 0; // True Negative
    let mut fn_ = 0; // False Negative

    for (pixels, &true_label) in test_features.iter().zip(test_binary.iter()) {
        let pred = perceptron.predict(pixels);
        if pred == 1 && true_label == 1 {
            tp += 1;
        } else if pred == 1 && true_label == 0 {
            fp += 1;
        } else if pred == 0 && true_label == 0 {
            tn += 1;
        } else if pred == 0 && true_label == 1 {
            fn_ += 1;
        }
    }

    let precision = tp as f64 / (tp + fp) as f64;
    let recall = tp as f64 / (tp + fn_) as f64;
    let f1 = 2.0 * precision * recall / (precision + recall);
    let accuracy = (tp + tn) as f64 / (tp + tn + fp + fn_) as f64;

    println!(
        "\nРезультаты бинарной классификации (класс {}):",
        target_class
    );
    println!("  True Positives:  {}", tp);
    println!("  False Positives: {}", fp);
    println!("  True Negatives:  {}", tn);
    println!("  False Negatives: {}", fn_);
    println!("  Precision: {:.3}", precision);
    println!("  Recall:    {:.3}", recall);
    println!("  F1-score:  {:.3}", f1);
    println!("  Accuracy:  {:.3}", accuracy);

    Ok(())
}
