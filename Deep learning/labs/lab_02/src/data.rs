use anyhow::{Context, Result};
use glob::glob;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Размер изображения после ресайза (ширина, высота)
const IMG_WIDTH: u32 = 56;
const IMG_HEIGHT: u32 = 56;
/// Количество цветовых каналов (RGB)
const CHANNELS: usize = 3;
/// Размер входного вектора для одного изображения
pub const INPUT_DIM: usize = (IMG_WIDTH * IMG_HEIGHT) as usize * CHANNELS;

pub fn load_class_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    // Получаем список поддиректорий (классов) в алфавитном порядке
    let mut class_dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .with_context(|| format!("Не удалось прочитать {:?}", root))?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .map(|entry| entry.path())
        .collect();
    class_dirs.sort(); // стабильный порядок классов
    Ok(class_dirs)
}

/// Загружает все изображения из одной директории (одного класса),
/// преобразует в плоские векторы f32 с нормализацией [0..1].
fn load_images_from_dir(dir: &Path) -> Result<Vec<Vec<f32>>> {
    let pattern = dir.join("*.jpg").to_str().unwrap().to_string();
    let paths: Vec<PathBuf> = glob(&pattern)
        .with_context(|| format!("Ошибка чтения паттерна {}", pattern))?
        .filter_map(|entry| entry.ok())
        .collect();

    let mut images = Vec::with_capacity(paths.len());

    for path in paths {
        let img = image::open(&path).with_context(|| format!("Не удалось открыть {:?}", path))?;
        let img = img.resize_exact(IMG_WIDTH, IMG_HEIGHT, image::imageops::FilterType::Lanczos3);
        let rgb = img.to_rgb8();
        let pixels = rgb.as_raw();

        // Нормализация в [0,1] и преобразование в Vec<f32>
        let vec: Vec<f32> = pixels.iter().map(|&p| p as f32 / 255.0).collect();
        images.push(vec);
    }

    Ok(images)
}

/// Загружает тренировочный датасет из папки `root`.
/// Возвращает (признаки, метки), где признаки – плоские векторы изображений,
/// метки – usize от 0 до num_classes-1.
pub fn load_dataset(root: &Path) -> Result<(Vec<Vec<f32>>, Vec<usize>)> {
    let class_dirs = load_class_dirs(root)?;

    let num_classes = class_dirs.len();
    if num_classes == 0 {
        anyhow::bail!("В папке {:?} нет поддиректорий с классами", root);
    }

    // Параллельная загрузка изображений каждого класса
    let class_data: Vec<(Vec<Vec<f32>>, usize)> = class_dirs
        .into_par_iter()
        .enumerate()
        .map(|(label, dir)| {
            let images = load_images_from_dir(&dir)
                .with_context(|| format!("Ошибка загрузки класса {:?}", dir))
                .unwrap(); // в реальном проекте лучше обработать ошибку аккуратно, здесь упрощаем
            (images, label)
        })
        .collect();

    // Сборка общих векторов
    let mut all_features = Vec::new();
    let mut all_labels = Vec::new();

    for (features, label) in class_data {
        for feat in features {
            // Проверка размера (на всякий случай)
            assert_eq!(
                feat.len(),
                INPUT_DIM,
                "Размер изображения не соответствует заданному"
            );
            all_features.push(feat);
            all_labels.push(label);
        }
    }

    Ok((all_features, all_labels))
}
