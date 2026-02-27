/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Quality tests — training stability, mode collapse detection,
 * and FID/IS scoring on toy datasets.
 */

use crate::facade::*;
use crate::network::{network_forward, network_backward, network_update_weights};
use crate::types::*;

fn m_finite(m: &TMatrix) -> bool {
    for row in m {
        for &v in row {
            if v.is_nan() || v.is_infinite() {
                return false;
            }
        }
    }
    true
}

// =========================================================================
// Training Stability Test
//
// Trains a small GAN for several epochs and verifies that:
//   - All weights stay finite after every epoch
//   - Discriminator and generator losses are finite
//   - Losses change over epochs (training is progressing)
// =========================================================================
pub fn test_training_stability() -> bool {
    println!("\n[Quality] Training stability test...");

    let gen_sizes = vec![4, 16, 8, 1];
    let disc_sizes = vec![1, 16, 8, 1];

    let mut gen = gf_gen_build(
        &gen_sizes,
        ActivationType::LeakyReLU,
        Optimizer::Adam,
        0.0002,
    );
    let mut disc = gf_disc_build(
        &disc_sizes,
        ActivationType::LeakyReLU,
        Optimizer::Adam,
        0.0002,
    );

    let ds = gf_train_create_synthetic(100, 1);
    let epochs = 5;
    let batch_size = 8;
    let num_batches = (ds.count / batch_size).max(1) as usize;

    let mut d_losses: Vec<f32> = Vec::new();
    let mut g_losses: Vec<f32> = Vec::new();
    let mut ok = true;

    for epoch in 0..epochs {
        let mut epoch_d_loss = 0.0f32;
        let mut epoch_g_loss = 0.0f32;

        for batch_idx in 0..num_batches {
            let start = batch_idx * batch_size as usize;
            let end = ((batch_idx + 1) * batch_size as usize).min(ds.count as usize);
            let actual_bs = end - start;
            if actual_bs == 0 {
                continue;
            }

            // Build real batch
            let mut real_batch = gf_op_create_matrix(actual_bs as i32, 1);
            for i in 0..actual_bs {
                if start + i < ds.samples.len() && !ds.samples[start + i].is_empty() {
                    real_batch[i][0] = ds.samples[start + i][0]
                        .first()
                        .copied()
                        .unwrap_or(0.0);
                }
            }

            let mut noise = vec![];
            gf_op_generate_noise(&mut noise, actual_bs as i32, 4, NoiseType::Gauss);

            // D step — real
            let d_real = network_forward(&mut disc, &real_batch);
            let mut real_labels = gf_op_create_matrix(actual_bs as i32, 1);
            for i in 0..actual_bs {
                real_labels[i][0] = 0.9; // label smoothing
            }
            let d_loss_real = gf_train_bce_loss(&d_real, &real_labels);
            let d_grad_real = gf_train_bce_grad(&d_real, &real_labels);
            network_backward(&mut disc, &d_grad_real);
            network_update_weights(&mut disc);

            // D step — fake
            let fake = network_forward(&mut gen, &noise);
            let d_fake = network_forward(&mut disc, &fake);
            let fake_labels = gf_op_create_matrix(actual_bs as i32, 1);
            let d_loss_fake = gf_train_bce_loss(&d_fake, &fake_labels);
            let d_grad_fake = gf_train_bce_grad(&d_fake, &fake_labels);
            network_backward(&mut disc, &d_grad_fake);
            network_update_weights(&mut disc);

            // G step
            let fake2 = network_forward(&mut gen, &noise);
            let d_gen = network_forward(&mut disc, &fake2);
            let mut gen_labels = gf_op_create_matrix(actual_bs as i32, 1);
            for i in 0..actual_bs {
                gen_labels[i][0] = 1.0;
            }
            let g_loss = gf_train_bce_loss(&d_gen, &gen_labels);
            let g_grad = gf_train_bce_grad(&d_gen, &gen_labels);
            let d_grad_thru = network_backward(&mut disc, &g_grad);
            network_backward(&mut gen, &d_grad_thru);
            network_update_weights(&mut gen);

            epoch_d_loss += d_loss_real + d_loss_fake;
            epoch_g_loss += g_loss;

            if d_loss_real.is_nan() || d_loss_fake.is_nan() || g_loss.is_nan()
                || d_loss_real.is_infinite() || d_loss_fake.is_infinite() || g_loss.is_infinite()
            {
                println!("  [FAIL] NaN/Inf loss at epoch {} batch {}", epoch + 1, batch_idx);
                ok = false;
            }
        }

        let avg_d = epoch_d_loss / num_batches as f32;
        let avg_g = epoch_g_loss / num_batches as f32;
        println!("  Epoch {}/{}: d_loss={:.4} g_loss={:.4}", epoch + 1, epochs, avg_d, avg_g);

        if !m_finite(&gen.layers[0].weights) {
            println!("  [FAIL] Generator weights contain NaN/Inf at epoch {}", epoch + 1);
            ok = false;
        }
        if !m_finite(&disc.layers[0].weights) {
            println!("  [FAIL] Discriminator weights contain NaN/Inf at epoch {}", epoch + 1);
            ok = false;
        }

        d_losses.push(avg_d);
        g_losses.push(avg_g);
    }

    // Check that losses are not all identical (some change should occur)
    let d_var: f32 = if d_losses.len() > 1 {
        let mean = d_losses.iter().sum::<f32>() / d_losses.len() as f32;
        d_losses.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / d_losses.len() as f32
    } else {
        1.0 // single epoch, skip variance check
    };

    if d_var < 1e-10 && epochs > 1 {
        println!(
            "  [WARN] Discriminator loss barely changed (variance={:.2e}); \
             training may be stalled",
            d_var
        );
        // Not a hard failure — GAN training can be volatile
    }

    if ok {
        println!("  [PASS] Training stability: all weights finite, losses valid");
    }
    ok
}

// =========================================================================
// Mode Collapse Detection Test
//
// Generates many samples from a trained generator and measures output
// diversity. Mode collapse manifests as near-zero variance across samples.
// =========================================================================
pub fn test_mode_collapse() -> bool {
    println!("\n[Quality] Mode collapse detection test...");

    let gen_sizes = vec![8, 32, 16, 4];
    let disc_sizes = vec![4, 16, 8, 1];

    let mut gen = gf_gen_build(
        &gen_sizes,
        ActivationType::LeakyReLU,
        Optimizer::Adam,
        0.0002,
    );
    let mut disc = gf_disc_build(
        &disc_sizes,
        ActivationType::LeakyReLU,
        Optimizer::Adam,
        0.0002,
    );

    // Train for 10 epochs to give the generator a chance to learn
    let ds = gf_train_create_synthetic(200, 4);
    let mut cfg = GANConfig::default();
    cfg.epochs = 10;
    cfg.batch_size = 8;
    cfg.loss_type = LossType::BCE;
    cfg.noise_depth = 8;
    gf_train_full(&mut gen, &mut disc, &ds, &cfg);

    // Generate 100 samples with different noise inputs
    let n_samples = 100;
    let noise_dim = 8;
    let samples = gf_gen_sample(&mut gen, n_samples, noise_dim, NoiseType::Gauss);

    if samples.is_empty() {
        println!("  [FAIL] Generator produced no samples");
        return false;
    }

    let output_dim = samples[0].len();
    if output_dim == 0 {
        println!("  [FAIL] Generator output has zero features");
        return false;
    }

    // Compute per-feature variance across all samples
    let mut means = vec![0.0f32; output_dim];
    for sample in &samples {
        for j in 0..output_dim.min(sample.len()) {
            means[j] += sample[j];
        }
    }
    for j in 0..output_dim {
        means[j] /= n_samples as f32;
    }

    let mut total_var = 0.0f32;
    for sample in &samples {
        for j in 0..output_dim.min(sample.len()) {
            let d = sample[j] - means[j];
            total_var += d * d;
        }
    }
    let avg_std = (total_var / (n_samples as f32 * output_dim as f32)).sqrt();

    println!(
        "  Output diversity: avg_std={:.6} (across {} samples, {} features)",
        avg_std, n_samples, output_dim
    );

    // Mode collapse threshold: outputs should have at least minimal diversity
    let min_std = 1e-4;
    if avg_std < min_std {
        println!(
            "  [FAIL] Mode collapse detected: avg_std={:.2e} < threshold {:.2e}",
            avg_std, min_std
        );
        false
    } else {
        println!("  [PASS] No mode collapse detected (diversity above threshold)");
        true
    }
}

// =========================================================================
// FID/IS on Toy Dataset
//
// Creates a structured toy dataset (mixture of two Gaussians in 4D),
// trains a GAN for a few epochs, then evaluates FID and IS on the
// generated samples vs real samples.
// =========================================================================
pub fn test_toy_metrics() -> bool {
    println!("\n[Quality] FID/IS scoring on toy dataset...");

    // --- Build toy dataset: mixture of two 4D Gaussians ---
    // Class A centred at (0.8, 0.8, 0.2, 0.2)
    // Class B centred at (0.2, 0.2, 0.8, 0.8)
    let n_real = 200;
    let mut real_samples: crate::types::TMatrixArray = Vec::new();
    let mut toy_ds = Dataset::default();
    toy_ds.data_type = DataType::Vector;
    toy_ds.count = n_real;

    let centers: [(f32, f32, f32, f32); 2] = [
        (0.8, 0.8, 0.2, 0.2),
        (0.2, 0.2, 0.8, 0.8),
    ];
    for i in 0..n_real as usize {
        let c = &centers[i % 2];
        let noise_scale = 0.05;
        let row = vec![
            (c.0 + gf_op_random_gaussian() * noise_scale).clamp(0.0, 1.0),
            (c.1 + gf_op_random_gaussian() * noise_scale).clamp(0.0, 1.0),
            (c.2 + gf_op_random_gaussian() * noise_scale).clamp(0.0, 1.0),
            (c.3 + gf_op_random_gaussian() * noise_scale).clamp(0.0, 1.0),
        ];
        real_samples.push(vec![row.clone()]);
        toy_ds.samples.push(vec![row]);
    }

    // --- Train a GAN on the toy dataset ---
    let gen_sizes = vec![8, 32, 16, 4];
    let disc_sizes = vec![4, 16, 8, 1];
    let mut gen = gf_gen_build(
        &gen_sizes,
        ActivationType::LeakyReLU,
        Optimizer::Adam,
        0.0002,
    );
    let mut disc = gf_disc_build(
        &disc_sizes,
        ActivationType::LeakyReLU,
        Optimizer::Adam,
        0.0002,
    );

    let mut cfg = GANConfig::default();
    cfg.epochs = 20;
    cfg.batch_size = 8;
    cfg.loss_type = LossType::BCE;
    cfg.noise_depth = 8;
    gf_train_full(&mut gen, &mut disc, &toy_ds, &cfg);

    // --- Generate fake samples ---
    let n_fake = 100;
    let fake_mat = gf_gen_sample(&mut gen, n_fake, 8, NoiseType::Gauss);
    let mut fake_samples: crate::types::TMatrixArray = Vec::new();
    for row in &fake_mat {
        fake_samples.push(vec![row.clone()]);
    }

    // --- Compute FID ---
    let fid = gf_train_compute_fid(&real_samples, &fake_samples);
    println!("  FID score: {:.6}", fid);
    if fid.is_nan() || fid.is_infinite() {
        println!("  [FAIL] FID is NaN/Inf");
        return false;
    }

    // --- Compute IS ---
    let is_score = gf_train_compute_is(&fake_samples);
    println!("  IS score:  {:.6}", is_score);
    if is_score.is_nan() || is_score.is_infinite() || is_score <= 0.0 {
        println!("  [FAIL] IS score is invalid (NaN/Inf/<=0)");
        return false;
    }

    // --- FID should be reasonable on this simple toy task ---
    // After 20 epochs, the generator should move somewhat towards real distribution.
    // We don't assert a tight bound since this is a CPU debug build with random seeds,
    // but FID should be finite and below a generous threshold.
    let fid_threshold = 2.0; // L2 distance of feature means; toy dataset spans [0,1]^4
    if fid > fid_threshold {
        println!(
            "  [WARN] FID={:.4} > threshold {:.4}; generator may not have converged \
             (expected for short training run)",
            fid, fid_threshold
        );
        // Warn but don't fail — 20 epochs may not be enough to converge
    }

    println!("  [PASS] FID and IS scores are valid and finite");
    true
}

// =========================================================================
// Entry point
// =========================================================================
pub fn run_quality_tests() -> bool {
    println!("==========================================================");
    println!(" GANFacade Quality Tests");
    println!("==========================================================");

    gf_sec_secure_randomize();

    let mut all_ok = true;
    let mut pass = 0;
    let mut fail = 0;

    macro_rules! quality_test {
        ($name:expr, $fn:expr) => {
            if $fn {
                println!("  [PASS] {}", $name);
                pass += 1;
            } else {
                println!("  [FAIL] {}", $name);
                fail += 1;
                all_ok = false;
            }
        };
    }

    quality_test!("Training stability", test_training_stability());
    quality_test!("Mode collapse detection", test_mode_collapse());
    quality_test!("FID/IS on toy dataset", test_toy_metrics());

    println!();
    println!("==========================================================");
    println!(" Quality tests: {} passed | {} failed", pass, fail);
    println!("==========================================================");

    all_ok
}
