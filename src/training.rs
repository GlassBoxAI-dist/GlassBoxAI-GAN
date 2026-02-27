/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Training control — port of GF_Train_ functions.
 */

use crate::loss::*;
use crate::matrix::create_matrix;
use crate::network::*;
use crate::random::{generate_noise, random_uniform};
use crate::types::*;
use std::fs;
use std::io::Write;

pub fn train_step(
    gen: &mut Network,
    disc: &mut Network,
    real_batch: &TMatrix,
    noise: &TMatrix,
    cfg: &GANConfig,
) -> GANMetrics {
    let bs = real_batch.len();

    // D step
    let disc_real = network_forward(disc, real_batch);
    let fake_data = network_forward(gen, noise);
    let disc_fake = network_forward(disc, &fake_data);

    match cfg.loss_type {
        LossType::BCE => {
            let mut real_labels = create_matrix(bs as i32, 1);
            for i in 0..bs {
                real_labels[i][0] = 1.0;
            }
            if cfg.use_label_smoothing {
                real_labels = apply_label_smoothing(&real_labels, 0.0, 0.9);
            }
            let d_grad = bce_gradient(&disc_real, &real_labels);
            network_forward(disc, real_batch);
            network_backward(disc, &d_grad);
            network_update_weights(disc);

            let fake_labels = create_matrix(bs as i32, 1);
            let d_grad = bce_gradient(&disc_fake, &fake_labels);
            network_forward(disc, &fake_data);
            network_backward(disc, &d_grad);
            network_update_weights(disc);
        }
        LossType::WGANGP => {
            let d_grad = wgan_disc_grad(&disc_real, true);
            network_forward(disc, real_batch);
            network_backward(disc, &d_grad);
            network_update_weights(disc);

            let d_grad = wgan_disc_grad(&disc_fake, false);
            network_forward(disc, &fake_data);
            network_backward(disc, &d_grad);
            network_update_weights(disc);
        }
        _ => {
            let labels = create_matrix(bs as i32, 1);
            let d_grad = bce_gradient(&disc_real, &labels);
            network_forward(disc, real_batch);
            network_backward(disc, &d_grad);
            network_update_weights(disc);
        }
    }

    // G step
    let fake_data = network_forward(gen, noise);
    let disc_gen = network_forward(disc, &fake_data);

    let g_grad = match cfg.loss_type {
        LossType::BCE => {
            let mut real_labels = create_matrix(bs as i32, 1);
            for i in 0..bs {
                real_labels[i][0] = 1.0;
            }
            bce_gradient(&disc_gen, &real_labels)
        }
        LossType::WGANGP => wgan_gen_grad(&disc_gen),
        _ => {
            let labels = create_matrix(bs as i32, 1);
            bce_gradient(&disc_gen, &labels)
        }
    };

    network_forward(disc, &fake_data);
    let d_grad_thru = network_backward(disc, &g_grad);
    network_forward(gen, noise);
    network_backward(gen, &d_grad_thru);
    network_update_weights(gen);

    // Compute and return metrics
    let ones = {
        let mut m = create_matrix(bs as i32, 1);
        for i in 0..bs { m[i][0] = 1.0; }
        m
    };
    let zeros = create_matrix(bs as i32, 1);
    let (d_loss_real, d_loss_fake) = match cfg.loss_type {
        LossType::BCE => (
            binary_cross_entropy(&disc_real, &ones),
            binary_cross_entropy(&disc_fake, &zeros),
        ),
        LossType::WGANGP  => (wgan_disc_loss(&disc_real, &disc_fake), 0.0),
        LossType::Hinge   => (hinge_disc_loss(&disc_real, &disc_fake), 0.0),
        LossType::LeastSquares => (ls_disc_loss(&disc_real, &disc_fake), 0.0),
    };
    let g_loss = match cfg.loss_type {
        LossType::BCE          => binary_cross_entropy(&disc_gen, &ones),
        LossType::WGANGP       => wgan_gen_loss(&disc_gen),
        LossType::Hinge        => hinge_gen_loss(&disc_gen),
        LossType::LeastSquares => ls_gen_loss(&disc_gen),
    };
    GANMetrics { d_loss_real, d_loss_fake, g_loss, ..GANMetrics::default() }
}

pub fn train_full(
    gen: &mut Network,
    disc: &mut Network,
    ds: &Dataset,
    cfg: &GANConfig,
) -> GANMetrics {
    let noise_depth = cfg.noise_depth;
    let batch_size = cfg.batch_size.max(1);
    let mut last_metrics = GANMetrics::default();

    for epoch in 0..cfg.epochs {
        let num_batches = (ds.count / batch_size).max(1);

        for batch_idx in 0..num_batches {
            let start = (batch_idx * batch_size) as usize;
            let end = ((batch_idx + 1) * batch_size).min(ds.count) as usize;
            let actual_bs = end - start;
            if actual_bs == 0 {
                continue;
            }

            // Build real batch
            let features = if !ds.samples.is_empty() && !ds.samples[0].is_empty() {
                ds.samples[0][0].len()
            } else {
                1
            };
            let mut real_batch = create_matrix(actual_bs as i32, features as i32);
            for i in 0..actual_bs {
                let si = start + i;
                if si < ds.samples.len() && !ds.samples[si].is_empty() {
                    for j in 0..features.min(ds.samples[si][0].len()) {
                        real_batch[i][j] = ds.samples[si][0][j];
                    }
                }
            }

            let mut noise = vec![];
            generate_noise(&mut noise, actual_bs as i32, noise_depth, cfg.noise_type);

            last_metrics = train_step(gen, disc, &real_batch, &noise, cfg);
        }

        if !cfg.output_dir.is_empty() && cfg.checkpoint_interval > 0 && (epoch + 1) % cfg.checkpoint_interval == 0 {
            save_checkpoint(gen, disc, epoch + 1, &cfg.output_dir);
        }

        // Print progress
        print!(
            "\rEpoch {}/{} complete",
            epoch + 1,
            cfg.epochs
        );
        std::io::stdout().flush().ok();
    }
    println!();
    last_metrics
}

// --- Data ---

pub fn load_dataset(path: &str, dt: DataType) -> Dataset {
    match dt {
        DataType::Image => load_bmp_dataset(path),
        DataType::Audio => load_wav_dataset(path),
        DataType::Vector => {
            // Try to load CSV
            let mut ds = Dataset::default();
            ds.data_type = dt;
            if let Ok(content) = fs::read_to_string(path) {
                for line in content.lines() {
                    let vals: Vec<f32> = line
                        .split(',')
                        .filter_map(|s| s.trim().parse::<f32>().ok())
                        .collect();
                    if !vals.is_empty() {
                        ds.samples.push(vec![vals]);
                        ds.count += 1;
                    }
                }
            }
            ds
        }
    }
}

pub fn load_bmp_dataset(_path: &str) -> Dataset {
    // Stub — BMP loading
    Dataset::default()
}

pub fn load_wav_dataset(_path: &str) -> Dataset {
    // Stub — WAV loading
    Dataset::default()
}

pub fn create_synthetic_dataset(count: i32, features: i32) -> Dataset {
    let mut ds = Dataset::default();
    ds.count = count;
    ds.data_type = DataType::Vector;
    for _ in 0..count {
        let row: Vec<f32> = (0..features).map(|_| random_uniform(0.0, 1.0)).collect();
        ds.samples.push(vec![row]);
    }
    ds
}

pub fn augment_sample(sample: &TMatrix, _dt: DataType) -> TMatrix {
    if sample.is_empty() {
        return vec![];
    }
    let mut result = sample.clone();
    // Simple noise augmentation
    for row in result.iter_mut() {
        for val in row.iter_mut() {
            *val += random_uniform(-0.01, 0.01);
        }
    }
    result
}

// --- Metrics ---

pub fn compute_fid(real_s: &TMatrixArray, fake_s: &TMatrixArray) -> f32 {
    if real_s.is_empty() || fake_s.is_empty() {
        return 0.0;
    }
    // Simplified FID: mean L2 distance between real and fake feature means
    let features = if !real_s[0].is_empty() {
        real_s[0][0].len()
    } else {
        return 0.0;
    };

    let mut real_mean = vec![0.0f32; features];
    let mut fake_mean = vec![0.0f32; features];

    for s in real_s {
        if !s.is_empty() {
            for j in 0..features.min(s[0].len()) {
                real_mean[j] += s[0][j];
            }
        }
    }
    for s in fake_s {
        if !s.is_empty() {
            for j in 0..features.min(s[0].len()) {
                fake_mean[j] += s[0][j];
            }
        }
    }
    for j in 0..features {
        real_mean[j] /= real_s.len() as f32;
        fake_mean[j] /= fake_s.len() as f32;
    }

    let mut dist = 0.0f32;
    for j in 0..features {
        let d = real_mean[j] - fake_mean[j];
        dist += d * d;
    }
    dist.sqrt()
}

pub fn compute_is(samples: &TMatrixArray) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    // Simplified IS: measure of diversity via variance
    let features = if !samples[0].is_empty() {
        samples[0][0].len()
    } else {
        return 0.0;
    };

    let mut mean = vec![0.0f32; features];
    for s in samples {
        if !s.is_empty() {
            for j in 0..features.min(s[0].len()) {
                mean[j] += s[0][j];
            }
        }
    }
    for j in 0..features {
        mean[j] /= samples.len() as f32;
    }

    let mut var_sum = 0.0f32;
    for s in samples {
        if !s.is_empty() {
            for j in 0..features.min(s[0].len()) {
                let d = s[0][j] - mean[j];
                var_sum += d * d;
            }
        }
    }
    let avg_var = var_sum / (samples.len() * features) as f32;
    avg_var.sqrt().exp()
}

pub fn log_metrics(met: &GANMetrics, filename: &str) {
    let header = !std::path::Path::new(filename).exists();
    if let Ok(mut f) = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(filename)
    {
        if header {
            writeln!(
                f,
                "epoch,batch,dLossReal,dLossFake,gLoss,fidScore,isScore,gradPenalty"
            )
            .ok();
        }
        writeln!(
            f,
            "{},{},{},{},{},{},{},{}",
            met.epoch,
            met.batch,
            met.d_loss_real,
            met.d_loss_fake,
            met.g_loss,
            met.fid_score,
            met.is_score,
            met.grad_penalty
        )
        .ok();
    }
}

// --- I/O ---

pub fn save_network_binary(net: &Network, filename: &str) {
    if let Ok(data) = serde_json::to_vec(net) {
        fs::write(filename, data).ok();
    }
}

pub fn load_network_binary(net: &mut Network, filename: &str) {
    if let Ok(data) = fs::read(filename) {
        if let Ok(loaded) = serde_json::from_slice::<Network>(&data) {
            *net = loaded;
        }
    }
}

pub fn save_gan_to_json(gen: &Network, disc: &Network, filename: &str) {
    #[derive(serde::Serialize)]
    struct GANPair<'a> {
        generator: &'a Network,
        discriminator: &'a Network,
    }
    if let Ok(json) = serde_json::to_string_pretty(&GANPair {
        generator: gen,
        discriminator: disc,
    }) {
        fs::write(filename, json).ok();
    }
}

pub fn load_gan_from_json(gen: &mut Network, disc: &mut Network, filename: &str) {
    #[derive(serde::Deserialize)]
    struct GANPair {
        generator: Network,
        discriminator: Network,
    }
    if let Ok(data) = fs::read_to_string(filename) {
        if let Ok(pair) = serde_json::from_str::<GANPair>(&data) {
            *gen = pair.generator;
            *disc = pair.discriminator;
        }
    }
}

pub fn save_checkpoint(gen: &Network, disc: &Network, ep: i32, dir: &str) {
    fs::create_dir_all(dir).ok();
    save_network_binary(gen, &format!("{}/gen_ep{}.bin", dir, ep));
    save_network_binary(disc, &format!("{}/disc_ep{}.bin", dir, ep));
}

pub fn load_checkpoint(gen: &mut Network, disc: &mut Network, ep: i32, dir: &str) {
    load_network_binary(gen, &format!("{}/gen_ep{}.bin", dir, ep));
    load_network_binary(disc, &format!("{}/disc_ep{}.bin", dir, ep));
}

pub fn save_generated_samples(
    gen: &mut Network,
    ep: i32,
    dir: &str,
    noise_dim: i32,
    nt: NoiseType,
) {
    let samples = gf_gen_sample(gen, 10, noise_dim, nt);
    let path = format!("{}/samples_ep{}.csv", dir, ep);
    if let Ok(mut f) = fs::File::create(&path) {
        for row in &samples {
            let line: Vec<String> = row.iter().map(|v| format!("{}", v)).collect();
            writeln!(f, "{}", line.join(",")).ok();
        }
    }
}

pub fn plot_loss_csv(filename: &str, d_loss: &[f32], g_loss: &[f32], cnt: i32) {
    if let Ok(mut f) = fs::File::create(filename) {
        writeln!(f, "epoch,d_loss,g_loss").ok();
        for i in 0..cnt as usize {
            let dl = if i < d_loss.len() { d_loss[i] } else { 0.0 };
            let gl = if i < g_loss.len() { g_loss[i] } else { 0.0 };
            writeln!(f, "{},{},{}", i, dl, gl).ok();
        }
    }
}

pub fn print_loss_bar(d_loss: f32, g_loss: f32, width: i32) {
    let w = width as usize;
    let d_bar = ((d_loss.min(1.0).max(0.0)) * w as f32) as usize;
    let g_bar = ((g_loss.min(1.0).max(0.0)) * w as f32) as usize;

    print!("D[");
    for i in 0..w {
        print!("{}", if i < d_bar { '#' } else { '-' });
    }
    print!("] G[");
    for i in 0..w {
        print!("{}", if i < g_bar { '#' } else { '-' });
    }
    print!("]");
}
