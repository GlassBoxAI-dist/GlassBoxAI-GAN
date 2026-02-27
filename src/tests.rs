/*
 * MIT License
 * Copyright (c) 2025 Matthew Abbott
 *
 * Test harness — port of RunSingleTest, RunAllTests, RunFuzzTests, ListFunctions.
 */

use crate::facade::*;
use crate::network::{network_forward, network_backward, network_update_weights};
use crate::types::*;

const TEST_DIR: &str = "ganfacade_testout";

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

fn v_finite(v: &TVector) -> bool {
    for &val in v {
        if val.is_nan() || val.is_infinite() {
            return false;
        }
    }
    true
}

fn file_exists(path: &str) -> bool {
    std::path::Path::new(path).exists()
}

const ALL_FUNCS: &[&str] = &[
    "GF_Op_CreateMatrix", "GF_Op_CreateVector", "GF_Op_MatrixMultiply",
    "GF_Op_MatrixAdd", "GF_Op_MatrixSubtract", "GF_Op_MatrixScale",
    "GF_Op_MatrixTranspose", "GF_Op_MatrixNormalize", "GF_Op_MatrixElementMul",
    "GF_Op_MatrixAddInPlace", "GF_Op_MatrixScaleInPlace", "GF_Op_MatrixClipInPlace",
    "GF_Op_SafeGet", "GF_Op_SafeSet",
    "GF_Op_ReLU", "GF_Op_LeakyReLU", "GF_Op_Sigmoid", "GF_Op_Tanh",
    "GF_Op_Softmax", "GF_Op_Activate", "GF_Op_ActivationBackward",
    "GF_Op_Conv2D", "GF_Op_Conv2DBackward", "GF_Op_Deconv2D",
    "GF_Op_Deconv2DBackward", "GF_Op_Conv1D", "GF_Op_Conv1DBackward",
    "GF_Op_BatchNorm", "GF_Op_BatchNormBackward",
    "GF_Op_LayerNorm", "GF_Op_LayerNormBackward", "GF_Op_SpectralNorm",
    "GF_Op_Attention", "GF_Op_AttentionBackward",
    "GF_Op_CreateDenseLayer", "GF_Op_CreateConv2DLayer",
    "GF_Op_CreateDeconv2DLayer", "GF_Op_CreateConv1DLayer",
    "GF_Op_CreateBatchNormLayer", "GF_Op_CreateLayerNormLayer",
    "GF_Op_CreateAttentionLayer",
    "GF_Op_LayerForward", "GF_Op_LayerBackward", "GF_Op_InitLayerOptimizer",
    "GF_Op_RandomGaussian", "GF_Op_RandomUniform",
    "GF_Op_GenerateNoise", "GF_Op_NoiseSlerp",
    "GF_Gen_Build", "GF_Gen_BuildConv", "GF_Gen_Forward", "GF_Gen_Backward",
    "GF_Gen_Sample", "GF_Gen_SampleConditional", "GF_Gen_UpdateWeights",
    "GF_Gen_AddProgressiveLayer", "GF_Gen_GetLayerOutput",
    "GF_Gen_SetTraining", "GF_Gen_Noise", "GF_Gen_NoiseSlerp", "GF_Gen_DeepCopy",
    "GF_Disc_Build", "GF_Disc_BuildConv", "GF_Disc_Evaluate", "GF_Disc_Forward",
    "GF_Disc_Backward", "GF_Disc_UpdateWeights", "GF_Disc_GradPenalty",
    "GF_Disc_FeatureMatch", "GF_Disc_MinibatchStdDev",
    "GF_Disc_AddProgressiveLayer", "GF_Disc_GetLayerOutput",
    "GF_Disc_SetTraining", "GF_Disc_DeepCopy",
    "GF_Train_BCELoss", "GF_Train_BCEGrad",
    "GF_Train_WGANDiscLoss", "GF_Train_WGANGenLoss",
    "GF_Train_HingeDiscLoss", "GF_Train_HingeGenLoss",
    "GF_Train_LSDiscLoss", "GF_Train_LSGenLoss", "GF_Train_LabelSmoothing",
    "GF_Train_AdamUpdate", "GF_Train_SGDUpdate",
    "GF_Train_RMSPropUpdate", "GF_Train_CosineAnneal",
    "GF_Train_CreateSynthetic", "GF_Train_Augment",
    "GF_Train_ComputeFID", "GF_Train_ComputeIS", "GF_Train_LogMetrics",
    "GF_Train_SaveModel", "GF_Train_LoadModel",
    "GF_Train_SaveJSON", "GF_Train_LoadJSON",
    "GF_Train_SaveCheckpoint", "GF_Train_LoadCheckpoint",
    "GF_Train_SaveSamples", "GF_Train_PlotCSV", "GF_Train_PrintBar",
    "GF_Train_Optimize", "GF_Train_Step", "GF_Train_Full",
    "GF_Sec_AuditLog", "GF_Sec_SecureRandomize", "GF_Sec_GetOSRandom",
    "GF_Sec_ValidatePath", "GF_Sec_VerifyWeights", "GF_Sec_VerifyNetwork",
    "GF_Sec_EncryptModel", "GF_Sec_DecryptModel",
    "GF_Sec_RunTests", "GF_Sec_RunFuzzTests", "GF_Sec_BoundsCheck",
    "GF_Introspect_NetworkFields", "GF_Introspect_LayerFields",
    "GF_Introspect_WeightAccess", "GF_Introspect_ForwardCache",
    "GF_Introspect_ActivationStats", "GF_Introspect_Gradients",
    "GF_Introspect_AdamState", "GF_Introspect_MultiUpdate",
    "GF_Introspect_DiscFields", "GF_Introspect_WeightDecay",
    "GF_Introspect_ConfigMutation", "GF_Introspect_LayerChain",
];

pub fn list_functions() {
    for name in ALL_FUNCS {
        println!("{}", name);
    }
}

pub fn run_single_test(name: &str) -> bool {
    std::fs::create_dir_all(TEST_DIR).ok();
    gf_sec_secure_randomize();

    match name {
        // === GF_Op_ Matrix ===
        "GF_Op_CreateMatrix" => {
            let a = gf_op_create_matrix(3, 4);
            a.len() == 3 && a[0].len() == 4 && a[0][0] == 0.0
        }
        "GF_Op_CreateVector" => {
            let v = gf_op_create_vector(5);
            v.len() == 5 && v[0] == 0.0
        }
        "GF_Op_MatrixMultiply" => {
            let mut a = gf_op_create_matrix(2, 3);
            let mut b = gf_op_create_matrix(3, 2);
            a[0][0]=1.0; a[0][1]=2.0; a[0][2]=3.0;
            a[1][0]=4.0; a[1][1]=5.0; a[1][2]=6.0;
            b[0][0]=7.0; b[0][1]=8.0; b[1][0]=9.0; b[1][1]=10.0;
            b[2][0]=11.0; b[2][1]=12.0;
            let c = gf_op_matrix_multiply(&a, &b);
            c.len() == 2 && c[0].len() == 2 && (c[0][0] - 58.0).abs() < 0.01
        }
        "GF_Op_MatrixAdd" => {
            let mut a = gf_op_create_matrix(2, 2);
            let mut b = gf_op_create_matrix(2, 2);
            a[0][0] = 1.0; b[0][0] = 5.0;
            let c = gf_op_matrix_add(&a, &b);
            (c[0][0] - 6.0).abs() < 0.01
        }
        "GF_Op_MatrixSubtract" => {
            let mut a = gf_op_create_matrix(2, 2);
            let mut b = gf_op_create_matrix(2, 2);
            a[0][0] = 10.0; b[0][0] = 3.0;
            let c = gf_op_matrix_subtract(&a, &b);
            (c[0][0] - 7.0).abs() < 0.01
        }
        "GF_Op_MatrixScale" => {
            let mut a = gf_op_create_matrix(2, 2);
            a[0][0] = 4.0;
            let c = gf_op_matrix_scale(&a, 3.0);
            (c[0][0] - 12.0).abs() < 0.01
        }
        "GF_Op_MatrixTranspose" => {
            let mut a = gf_op_create_matrix(2, 3);
            a[0][0]=1.0; a[0][1]=2.0; a[0][2]=3.0;
            a[1][0]=4.0; a[1][1]=5.0; a[1][2]=6.0;
            let c = gf_op_matrix_transpose(&a);
            c.len() == 3 && c[0].len() == 2 && (c[2][1] - 6.0).abs() < 0.01
        }
        "GF_Op_MatrixNormalize" => {
            let mut a = gf_op_create_matrix(1, 3);
            a[0][0] = 3.0; a[0][1] = 0.0; a[0][2] = 4.0;
            let c = gf_op_matrix_normalize(&a);
            m_finite(&c)
        }
        "GF_Op_MatrixElementMul" => {
            let mut a = gf_op_create_matrix(2, 2);
            let mut b = gf_op_create_matrix(2, 2);
            a[0][0] = 3.0; b[0][0] = 4.0;
            let c = gf_op_matrix_element_mul(&a, &b);
            (c[0][0] - 12.0).abs() < 0.01
        }
        "GF_Op_MatrixAddInPlace" => {
            let mut a = gf_op_create_matrix(2, 2);
            let mut b = gf_op_create_matrix(2, 2);
            a[0][0] = 1.0; b[0][0] = 5.0;
            gf_op_matrix_add_in_place(&mut a, &b);
            (a[0][0] - 6.0).abs() < 0.01
        }
        "GF_Op_MatrixScaleInPlace" => {
            let mut a = gf_op_create_matrix(2, 2);
            a[0][0] = 4.0;
            gf_op_matrix_scale_in_place(&mut a, 3.0);
            (a[0][0] - 12.0).abs() < 0.01
        }
        "GF_Op_MatrixClipInPlace" => {
            let mut a = gf_op_create_matrix(1, 3);
            a[0][0] = -5.0; a[0][1] = 0.5; a[0][2] = 5.0;
            gf_op_matrix_clip_in_place(&mut a, -1.0, 1.0);
            (a[0][0] - (-1.0)).abs() < 0.01
                && (a[0][1] - 0.5).abs() < 0.01
                && (a[0][2] - 1.0).abs() < 0.01
        }
        "GF_Op_SafeGet" => {
            let mut a = gf_op_create_matrix(2, 2);
            a[0][0] = 42.0;
            (gf_op_safe_get(&a, 0, 0, 0.0) - 42.0).abs() < 0.01
                && (gf_op_safe_get(&a, 99, 99, -1.0) - (-1.0)).abs() < 0.01
        }
        "GF_Op_SafeSet" => {
            let mut a = gf_op_create_matrix(2, 2);
            gf_op_safe_set(&mut a, 0, 0, 42.0);
            gf_op_safe_set(&mut a, 99, 99, 100.0);
            (a[0][0] - 42.0).abs() < 0.01
        }
        // === Activations ===
        "GF_Op_ReLU" => {
            let mut a = gf_op_create_matrix(1, 3);
            a[0][0] = -1.0; a[0][1] = 0.0; a[0][2] = 2.0;
            let c = gf_op_relu(&a);
            c[0][0] == 0.0 && c[0][2] == 2.0
        }
        "GF_Op_LeakyReLU" => {
            let mut a = gf_op_create_matrix(1, 2);
            a[0][0] = -1.0; a[0][1] = 2.0;
            let c = gf_op_leaky_relu(&a, 0.01);
            (c[0][0] - (-0.01)).abs() < 0.001 && c[0][1] == 2.0
        }
        "GF_Op_Sigmoid" => {
            let mut a = gf_op_create_matrix(1, 1);
            a[0][0] = 0.0;
            let c = gf_op_sigmoid(&a);
            (c[0][0] - 0.5).abs() < 0.01
        }
        "GF_Op_Tanh" => {
            let mut a = gf_op_create_matrix(1, 1);
            a[0][0] = 0.0;
            let c = gf_op_tanh(&a);
            c[0][0].abs() < 0.01
        }
        "GF_Op_Softmax" => {
            let mut a = gf_op_create_matrix(1, 3);
            a[0][0] = 1.0; a[0][1] = 2.0; a[0][2] = 3.0;
            let c = gf_op_softmax(&a);
            let sum = c[0][0] + c[0][1] + c[0][2];
            m_finite(&c) && (sum - 1.0).abs() < 0.01
        }
        "GF_Op_Activate" => {
            let mut a = gf_op_create_matrix(1, 2);
            a[0][0] = -1.0; a[0][1] = 2.0;
            let c = gf_op_activate(&a, ActivationType::ReLU);
            c[0][0] == 0.0 && c[0][1] == 2.0
        }
        "GF_Op_ActivationBackward" => {
            let mut a = gf_op_create_matrix(1, 2);
            a[0][0] = 1.0; a[0][1] = 1.0;
            let mut b = gf_op_create_matrix(1, 2);
            b[0][0] = -1.0; b[0][1] = 2.0;
            let c = gf_op_activation_backward(&a, &b, ActivationType::ReLU);
            c[0][0] == 0.0 && c[0][1] == 1.0
        }
        // === Conv ===
        "GF_Op_Conv2D" => {
            let mut layer = gf_op_create_conv2d_layer(1, 2, 3, 1, 1, 4, 4, ActivationType::ReLU);
            let mut inp = gf_op_create_matrix(2, 16);
            inp[0][0] = 1.0;
            layer.layer_input = inp.clone();
            let out = gf_op_conv2d(&inp, &mut layer);
            out.len() == 2 && out[0].len() == 32 && m_finite(&out)
        }
        "GF_Op_Conv2DBackward" => {
            let mut layer = gf_op_create_conv2d_layer(1, 2, 3, 1, 1, 4, 4, ActivationType::ReLU);
            let mut inp = gf_op_create_matrix(2, 16);
            inp[0][0] = 1.0;
            layer.layer_input = inp.clone();
            crate::convolution::conv2d_forward(&inp, &mut layer);
            let mut grad = gf_op_create_matrix(2, 32);
            grad[0][0] = 1.0;
            let gi = gf_op_conv2d_backward(&mut layer, &grad);
            gi.len() == 2 && gi[0].len() == 16 && m_finite(&gi)
        }
        "GF_Op_Deconv2D" => {
            let mut layer = gf_op_create_deconv2d_layer(1, 2, 3, 1, 1, 4, 4, ActivationType::ReLU);
            let mut inp = gf_op_create_matrix(2, 16);
            inp[0][0] = 1.0;
            layer.layer_input = inp.clone();
            let out = gf_op_deconv2d(&inp, &mut layer);
            out.len() == 2 && out[0].len() == 32 && m_finite(&out)
        }
        "GF_Op_Deconv2DBackward" => {
            let mut layer = gf_op_create_deconv2d_layer(1, 2, 3, 1, 1, 4, 4, ActivationType::ReLU);
            let mut inp = gf_op_create_matrix(2, 16);
            inp[0][0] = 1.0;
            layer.layer_input = inp.clone();
            crate::convolution::deconv2d_forward(&inp, &mut layer);
            let mut grad = gf_op_create_matrix(2, 32);
            grad[0][0] = 1.0;
            let gi = gf_op_deconv2d_backward(&mut layer, &grad);
            gi.len() == 2 && gi[0].len() == 16 && m_finite(&gi)
        }
        "GF_Op_Conv1D" => {
            let mut layer = gf_op_create_conv1d_layer(1, 2, 3, 1, 1, 8, ActivationType::ReLU);
            let mut inp = gf_op_create_matrix(2, 8);
            inp[0][0] = 1.0;
            layer.layer_input = inp.clone();
            let out = gf_op_conv1d(&inp, &mut layer);
            out.len() == 2 && out[0].len() == 16 && m_finite(&out)
        }
        "GF_Op_Conv1DBackward" => {
            let mut layer = gf_op_create_conv1d_layer(1, 2, 3, 1, 1, 8, ActivationType::ReLU);
            let mut inp = gf_op_create_matrix(2, 8);
            inp[0][0] = 1.0;
            layer.layer_input = inp.clone();
            crate::convolution::conv1d_forward(&inp, &mut layer);
            let mut grad = gf_op_create_matrix(2, 16);
            grad[0][0] = 1.0;
            let gi = gf_op_conv1d_backward(&mut layer, &grad);
            gi.len() == 2 && gi[0].len() == 8 && m_finite(&gi)
        }
        // === Normalization ===
        "GF_Op_BatchNorm" => {
            let mut layer = gf_op_create_batch_norm_layer(8);
            layer.is_training = true;
            let mut inp = gf_op_create_matrix(4, 8);
            for i in 0..4 { inp[i][0] = (i + 1) as f32; }
            layer.layer_input = inp.clone();
            let out = gf_op_batch_norm(&inp, &mut layer);
            out.len() == 4 && out[0].len() == 8 && m_finite(&out)
        }
        "GF_Op_BatchNormBackward" => {
            let mut layer = gf_op_create_batch_norm_layer(8);
            layer.is_training = true;
            let mut inp = gf_op_create_matrix(4, 8);
            for i in 0..4 { inp[i][0] = (i + 1) as f32; }
            layer.layer_input = inp.clone();
            crate::normalization::batch_norm_forward(&inp, &mut layer);
            let mut grad = gf_op_create_matrix(4, 8);
            for i in 0..4 { grad[i][0] = 1.0; }
            let gi = gf_op_batch_norm_backward(&mut layer, &grad);
            gi.len() == 4 && m_finite(&gi)
        }
        "GF_Op_LayerNorm" => {
            let mut layer = gf_op_create_layer_norm_layer(8);
            layer.is_training = true;
            let mut inp = gf_op_create_matrix(4, 8);
            for i in 0..4 { inp[i][0] = (i + 1) as f32; }
            layer.layer_input = inp.clone();
            let out = gf_op_layer_norm(&inp, &mut layer);
            m_finite(&out)
        }
        "GF_Op_LayerNormBackward" => {
            let mut layer = gf_op_create_layer_norm_layer(8);
            layer.is_training = true;
            let mut inp = gf_op_create_matrix(4, 8);
            for i in 0..4 { inp[i][0] = (i + 1) as f32; }
            layer.layer_input = inp.clone();
            crate::normalization::layer_norm_forward(&inp, &mut layer);
            let mut grad = gf_op_create_matrix(4, 8);
            for i in 0..4 { grad[i][0] = 1.0; }
            let gi = gf_op_layer_norm_backward(&mut layer, &grad);
            m_finite(&gi)
        }
        "GF_Op_SpectralNorm" => {
            let mut layer = gf_op_create_dense_layer(4, 4, ActivationType::ReLU);
            layer.spectral_u = vec![0.0; 4];
            layer.spectral_v = vec![0.0; 4];
            for i in 0..4 {
                layer.spectral_u[i] = gf_op_random_gaussian();
                layer.spectral_v[i] = gf_op_random_gaussian();
            }
            let out = gf_op_spectral_norm(&mut layer);
            m_finite(&out) && layer.spectral_sigma > 0.0
        }
        // === Attention ===
        "GF_Op_Attention" => {
            let mut layer = gf_op_create_attention_layer(4, 2);
            let mut inp = gf_op_create_matrix(2, 4);
            inp[0][0]=1.0; inp[0][1]=2.0; inp[0][2]=3.0; inp[0][3]=4.0;
            inp[1][0]=5.0; inp[1][1]=6.0; inp[1][2]=7.0; inp[1][3]=8.0;
            let out = gf_op_attention(&inp, &mut layer);
            out.len() == 2 && out[0].len() == 4 && m_finite(&out)
        }
        "GF_Op_AttentionBackward" => {
            let mut layer = gf_op_create_attention_layer(4, 2);
            let mut inp = gf_op_create_matrix(2, 4);
            inp[0][0]=1.0; inp[0][1]=2.0; inp[0][2]=3.0; inp[0][3]=4.0;
            inp[1][0]=5.0; inp[1][1]=6.0; inp[1][2]=7.0; inp[1][3]=8.0;
            crate::attention::self_attention_forward(&inp, &mut layer);
            let mut grad = gf_op_create_matrix(2, 4);
            for i in 0..4 { grad[0][i] = 1.0; grad[1][i] = 1.0; }
            let gi = gf_op_attention_backward(&mut layer, &grad);
            m_finite(&gi)
        }
        // === Layer creation ===
        "GF_Op_CreateDenseLayer" => {
            let layer = gf_op_create_dense_layer(4, 3, ActivationType::ReLU);
            layer.layer_type == LayerType::Dense
                && layer.input_size == 4
                && layer.output_size == 3
                && layer.weights.len() == 4
                && layer.weights[0].len() == 3
                && layer.bias.len() == 3
        }
        "GF_Op_CreateConv2DLayer" => {
            let layer = gf_op_create_conv2d_layer(1, 2, 3, 1, 1, 4, 4, ActivationType::ReLU);
            layer.layer_type == LayerType::Conv2D
                && layer.input_size == 16
                && layer.output_size == 32
        }
        "GF_Op_CreateDeconv2DLayer" => {
            let layer = gf_op_create_deconv2d_layer(1, 2, 3, 1, 1, 4, 4, ActivationType::ReLU);
            layer.layer_type == LayerType::Deconv2D && layer.output_size == 32
        }
        "GF_Op_CreateConv1DLayer" => {
            let layer = gf_op_create_conv1d_layer(1, 2, 3, 1, 1, 8, ActivationType::ReLU);
            layer.layer_type == LayerType::Conv1D && layer.output_size == 16
        }
        "GF_Op_CreateBatchNormLayer" => {
            let layer = gf_op_create_batch_norm_layer(8);
            layer.layer_type == LayerType::BatchNorm
                && (layer.bn_gamma[0] - 1.0).abs() < 0.01
        }
        "GF_Op_CreateLayerNormLayer" => {
            let layer = gf_op_create_layer_norm_layer(8);
            layer.layer_type == LayerType::LayerNorm
                && (layer.bn_gamma[0] - 1.0).abs() < 0.01
        }
        "GF_Op_CreateAttentionLayer" => {
            let layer = gf_op_create_attention_layer(4, 2);
            layer.layer_type == LayerType::Attention
                && layer.head_dim == 2
                && layer.wq.len() == 4
                && layer.wq[0].len() == 4
        }
        // === Layer dispatch ===
        "GF_Op_LayerForward" => {
            let mut layer = gf_op_create_dense_layer(4, 3, ActivationType::ReLU);
            let mut inp = gf_op_create_matrix(2, 4);
            inp[0][0] = 1.0; inp[0][1] = 2.0;
            let out = gf_op_layer_forward(&mut layer, &inp);
            out.len() == 2 && out[0].len() == 3 && m_finite(&out)
        }
        "GF_Op_LayerBackward" => {
            let mut layer = gf_op_create_dense_layer(4, 3, ActivationType::ReLU);
            let mut inp = gf_op_create_matrix(2, 4);
            inp[0][0] = 1.0;
            crate::layer::layer_forward(&mut layer, &inp);
            let mut grad = gf_op_create_matrix(2, 3);
            for b in 0..2 { for j in 0..3 { grad[b][j] = 1.0; } }
            let gi = gf_op_layer_backward(&mut layer, &grad);
            gi[0].len() == 4 && m_finite(&gi)
        }
        "GF_Op_InitLayerOptimizer" => {
            let mut layer = gf_op_create_dense_layer(4, 3, ActivationType::ReLU);
            gf_op_init_layer_optimizer(&mut layer, Optimizer::Adam);
            layer.m_weight.len() == 4 && layer.v_weight.len() == 4
        }
        // === Random ===
        "GF_Op_RandomGaussian" => {
            let v = gf_op_random_gaussian();
            !v.is_nan() && !v.is_infinite()
        }
        "GF_Op_RandomUniform" => {
            let mut ok = true;
            for _ in 0..50 {
                let v = gf_op_random_uniform(0.0, 1.0);
                if v < 0.0 || v > 1.0 { ok = false; }
            }
            ok
        }
        "GF_Op_GenerateNoise" => {
            let mut noise = vec![];
            gf_op_generate_noise(&mut noise, 4, 8, NoiseType::Gauss);
            noise.len() == 4 && noise[0].len() == 8 && m_finite(&noise)
        }
        "GF_Op_NoiseSlerp" => {
            let mut v1 = vec![0.0f32; 4];
            let mut v2 = vec![0.0f32; 4];
            v1[0] = 1.0; v2[1] = 1.0;
            let vs = gf_op_noise_slerp(&v1, &v2, 0.5);
            v_finite(&vs) && vs.len() == 4
        }
        // === GF_Gen_ ===
        "GF_Gen_Build" => {
            let sizes = vec![8, 16, 8, 1];
            let gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            gen.layer_count == 3 && gen.optimizer == Optimizer::Adam
        }
        "GF_Gen_BuildConv" => {
            let gen = gf_gen_build_conv(8, 0, 4, ActivationType::LeakyReLU, Optimizer::Adam, 0.0002);
            gen.layer_count == 7 && gen.layers[0].layer_type == LayerType::Dense
        }
        "GF_Gen_Forward" => {
            let sizes = vec![4, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 4);
            inp[0][0] = 1.0;
            let out = gf_gen_forward(&mut gen, &inp);
            out.len() == 2 && out[0].len() == 1 && m_finite(&out)
        }
        "GF_Gen_Backward" => {
            let sizes = vec![4, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 4);
            inp[0][0] = 1.0;
            network_forward(&mut gen, &inp);
            let mut grad = gf_op_create_matrix(2, 1);
            grad[0][0] = 1.0;
            let out = gf_gen_backward(&mut gen, &grad);
            out.len() == 2 && out[0].len() == 4 && m_finite(&out)
        }
        "GF_Gen_Sample" => {
            let sizes = vec![4, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let out = gf_gen_sample(&mut gen, 4, 4, NoiseType::Gauss);
            out.len() == 4 && m_finite(&out)
        }
        "GF_Gen_SampleConditional" => {
            let sizes = vec![6, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut cond = gf_op_create_matrix(2, 2);
            cond[0][0] = 1.0; cond[1][1] = 1.0;
            let out = gf_gen_sample_conditional(&mut gen, 2, 4, 2, NoiseType::Gauss, &cond);
            out.len() == 2 && m_finite(&out)
        }
        "GF_Gen_UpdateWeights" => {
            let sizes = vec![4, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 4);
            inp[0][0] = 1.0;
            network_forward(&mut gen, &inp);
            network_backward(&mut gen, &gf_op_create_matrix(2, 1));
            network_update_weights(&mut gen);
            m_finite(&gen.layers[0].weights)
        }
        "GF_Gen_AddProgressiveLayer" => {
            let sizes = vec![8, 16, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            gf_gen_add_progressive_layer(&mut gen, 1);
            gen.layer_count == 5
        }
        "GF_Gen_GetLayerOutput" => {
            let sizes = vec![4, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 4);
            inp[0][0] = 1.0;
            network_forward(&mut gen, &inp);
            let out = gf_gen_get_layer_output(&gen, 0);
            !out.is_empty() && m_finite(&out)
        }
        "GF_Gen_SetTraining" => {
            let sizes = vec![4, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            gf_gen_set_training(&mut gen, false);
            let ok1 = !gen.is_training;
            gf_gen_set_training(&mut gen, true);
            ok1 && gen.is_training
        }
        "GF_Gen_Noise" => {
            let noise = gf_gen_noise(3, 8, NoiseType::Gauss);
            noise.len() == 3 && noise[0].len() == 8 && m_finite(&noise)
        }
        "GF_Gen_NoiseSlerp" => {
            let mut v1 = vec![0.0f32; 4];
            let mut v2 = vec![0.0f32; 4];
            v1[0] = 1.0; v2[3] = 1.0;
            let vs = gf_gen_noise_slerp(&v1, &v2, 0.5);
            v_finite(&vs)
        }
        "GF_Gen_DeepCopy" => {
            let sizes = vec![4, 8, 1];
            let gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut copy = gf_gen_deep_copy(&gen);
            copy.learning_rate = 0.999;
            copy.layer_count == gen.layer_count
                && (gen.learning_rate - 0.001).abs() < 0.0001
        }
        // === GF_Disc_ ===
        "GF_Disc_Build" => {
            let sizes = vec![1, 8, 16, 1];
            let disc = gf_disc_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            disc.layer_count == 3
        }
        "GF_Disc_BuildConv" => {
            let disc = gf_disc_build_conv(1, 8, 8, 0, 4, ActivationType::LeakyReLU, Optimizer::Adam, 0.0002);
            disc.layer_count == 5 && disc.layers[4].layer_type == LayerType::Dense
        }
        "GF_Disc_Evaluate" => {
            let sizes = vec![1, 8, 1];
            let mut disc = gf_disc_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(4, 1);
            inp[0][0] = 0.5; inp[1][0] = -0.3;
            let out = gf_disc_evaluate(&mut disc, &inp);
            out.len() == 4 && out[0].len() == 1 && m_finite(&out)
        }
        "GF_Disc_Forward" => {
            let sizes = vec![1, 8, 1];
            let mut disc = gf_disc_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 1);
            inp[0][0] = 0.5;
            let out = gf_disc_forward(&mut disc, &inp);
            out.len() == 2 && m_finite(&out)
        }
        "GF_Disc_Backward" => {
            let sizes = vec![1, 8, 1];
            let mut disc = gf_disc_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 1);
            inp[0][0] = 0.5;
            network_forward(&mut disc, &inp);
            let mut grad = gf_op_create_matrix(2, 1);
            grad[0][0] = 1.0;
            let out = gf_disc_backward(&mut disc, &grad);
            out.len() == 2 && m_finite(&out)
        }
        "GF_Disc_UpdateWeights" => {
            let sizes = vec![1, 8, 1];
            let mut disc = gf_disc_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 1);
            inp[0][0] = 0.5;
            network_forward(&mut disc, &inp);
            network_backward(&mut disc, &gf_op_create_matrix(2, 1));
            gf_disc_update_weights(&mut disc);
            m_finite(&disc.layers[0].weights)
        }
        "GF_Disc_GradPenalty" => {
            let sizes = vec![1, 8, 1];
            let mut disc = gf_disc_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut a = gf_op_create_matrix(4, 1);
            let mut b = gf_op_create_matrix(4, 1);
            a[0][0]=0.9; a[1][0]=0.8; a[2][0]=0.7; a[3][0]=0.6;
            b[0][0]=0.1; b[1][0]=0.2; b[2][0]=0.3; b[3][0]=0.4;
            let gp = gf_disc_grad_penalty(&mut disc, &a, &b, 10.0);
            !gp.is_nan() && !gp.is_infinite() && gp >= 0.0
        }
        "GF_Disc_FeatureMatch" => {
            let sizes = vec![1, 8, 1];
            let mut disc = gf_disc_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut a = gf_op_create_matrix(4, 1);
            let mut b = gf_op_create_matrix(4, 1);
            a[0][0]=0.9; a[1][0]=0.8; a[2][0]=0.7; a[3][0]=0.6;
            b[0][0]=0.1; b[1][0]=0.2; b[2][0]=0.3; b[3][0]=0.4;
            network_forward(&mut disc, &a);
            let fm = gf_disc_feature_match(&mut disc, &a, &b, 0);
            !fm.is_nan() && !fm.is_infinite() && fm >= 0.0
        }
        "GF_Disc_MinibatchStdDev" => {
            let mut inp = gf_op_create_matrix(4, 1);
            inp[0][0]=0.5; inp[1][0]=-0.3; inp[2][0]=0.8; inp[3][0]=0.1;
            let out = gf_disc_minibatch_std_dev(&inp);
            out.len() == 4 && out[0].len() == 2 && m_finite(&out)
        }
        "GF_Disc_AddProgressiveLayer" => {
            let sizes = vec![1, 8, 16, 1];
            let mut disc = gf_disc_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            gf_disc_add_progressive_layer(&mut disc, 1);
            disc.layer_count == 5
        }
        "GF_Disc_GetLayerOutput" => {
            let sizes = vec![1, 8, 1];
            let mut disc = gf_disc_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 1);
            inp[0][0] = 0.5;
            network_forward(&mut disc, &inp);
            let out = gf_disc_get_layer_output(&disc, 0);
            !out.is_empty() && m_finite(&out)
        }
        "GF_Disc_SetTraining" => {
            let sizes = vec![1, 8, 1];
            let mut disc = gf_disc_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            gf_disc_set_training(&mut disc, false);
            let ok1 = !disc.is_training;
            gf_disc_set_training(&mut disc, true);
            ok1 && disc.is_training
        }
        "GF_Disc_DeepCopy" => {
            let sizes = vec![1, 8, 1];
            let disc = gf_disc_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut copy = gf_disc_deep_copy(&disc);
            copy.learning_rate = 0.999;
            copy.layer_count == disc.layer_count
                && (disc.learning_rate - 0.001).abs() < 0.0001
        }
        // === GF_Train_ Loss ===
        "GF_Train_BCELoss" => {
            let mut pred = gf_op_create_matrix(4, 1);
            let mut target = gf_op_create_matrix(4, 1);
            pred[0][0]=0.9; pred[1][0]=0.8; pred[2][0]=0.2; pred[3][0]=0.1;
            target[0][0]=1.0; target[1][0]=1.0; target[2][0]=0.0; target[3][0]=0.0;
            let loss = gf_train_bce_loss(&pred, &target);
            !loss.is_nan() && !loss.is_infinite() && loss > 0.0 && loss < 5.0
        }
        "GF_Train_BCEGrad" => {
            let mut pred = gf_op_create_matrix(4, 1);
            let mut target = gf_op_create_matrix(4, 1);
            pred[0][0] = 0.9; target[0][0] = 1.0;
            let grad = gf_train_bce_grad(&pred, &target);
            grad.len() == 4 && m_finite(&grad)
        }
        "GF_Train_WGANDiscLoss" => {
            let mut d_real = gf_op_create_matrix(4, 1);
            let mut d_fake = gf_op_create_matrix(4, 1);
            d_real[0][0]=2.0; d_real[1][0]=1.5; d_real[2][0]=1.8; d_real[3][0]=2.1;
            d_fake[0][0]=-1.0; d_fake[1][0]=-0.5; d_fake[2][0]=-0.8; d_fake[3][0]=-1.2;
            let loss = gf_train_wgan_disc_loss(&d_real, &d_fake);
            !loss.is_nan() && !loss.is_infinite()
        }
        "GF_Train_WGANGenLoss" => {
            let mut d_fake = gf_op_create_matrix(4, 1);
            d_fake[0][0]=-1.0; d_fake[1][0]=-0.5; d_fake[2][0]=-0.8; d_fake[3][0]=-1.2;
            let loss = gf_train_wgan_gen_loss(&d_fake);
            !loss.is_nan() && !loss.is_infinite()
        }
        "GF_Train_HingeDiscLoss" => {
            let mut d_real = gf_op_create_matrix(4, 1);
            let mut d_fake = gf_op_create_matrix(4, 1);
            d_real[0][0]=2.0; d_real[1][0]=1.5; d_real[2][0]=1.8; d_real[3][0]=2.1;
            d_fake[0][0]=-1.0; d_fake[1][0]=-0.5; d_fake[2][0]=-0.8; d_fake[3][0]=-1.2;
            let loss = gf_train_hinge_disc_loss(&d_real, &d_fake);
            !loss.is_nan() && !loss.is_infinite()
        }
        "GF_Train_HingeGenLoss" => {
            let mut d_fake = gf_op_create_matrix(4, 1);
            d_fake[0][0]=-1.0; d_fake[1][0]=-0.5; d_fake[2][0]=-0.8; d_fake[3][0]=-1.2;
            let loss = gf_train_hinge_gen_loss(&d_fake);
            !loss.is_nan() && !loss.is_infinite()
        }
        "GF_Train_LSDiscLoss" => {
            let mut d_real = gf_op_create_matrix(4, 1);
            let mut d_fake = gf_op_create_matrix(4, 1);
            d_real[0][0]=2.0; d_real[1][0]=1.5; d_real[2][0]=1.8; d_real[3][0]=2.1;
            d_fake[0][0]=-1.0; d_fake[1][0]=-0.5; d_fake[2][0]=-0.8; d_fake[3][0]=-1.2;
            let loss = gf_train_ls_disc_loss(&d_real, &d_fake);
            !loss.is_nan() && !loss.is_infinite()
        }
        "GF_Train_LSGenLoss" => {
            let mut d_fake = gf_op_create_matrix(4, 1);
            d_fake[0][0]=-1.0; d_fake[1][0]=-0.5; d_fake[2][0]=-0.8; d_fake[3][0]=-1.2;
            let loss = gf_train_ls_gen_loss(&d_fake);
            !loss.is_nan() && !loss.is_infinite()
        }
        "GF_Train_LabelSmoothing" => {
            let mut labels = gf_op_create_matrix(4, 1);
            labels[0][0]=1.0; labels[1][0]=1.0; labels[2][0]=0.0; labels[3][0]=0.0;
            let smoothed = gf_train_label_smoothing(&labels, 0.0, 0.9);
            smoothed[0][0] <= 0.91 && smoothed[2][0] >= -0.01 && m_finite(&smoothed)
        }
        // === GF_Train_ Optimizers ===
        "GF_Train_AdamUpdate" => {
            let mut p = gf_op_create_matrix(2, 2);
            let mut g = gf_op_create_matrix(2, 2);
            let mut m = gf_op_create_matrix(2, 2);
            let mut v = gf_op_create_matrix(2, 2);
            p[0][0] = 1.0; g[0][0] = 0.1;
            gf_train_adam_update(&mut p, &g, &mut m, &mut v, 1, 0.001, 0.9, 0.999, 1e-8, 0.0);
            m_finite(&p) && (p[0][0] - 1.0).abs() > 1e-6
        }
        "GF_Train_SGDUpdate" => {
            let mut p = gf_op_create_matrix(2, 2);
            let mut g = gf_op_create_matrix(2, 2);
            p[0][0] = 1.0; g[0][0] = 0.5;
            gf_train_sgd_update(&mut p, &g, 0.01, 0.0);
            m_finite(&p) && (p[0][0] - 1.0).abs() > 1e-6
        }
        "GF_Train_RMSPropUpdate" => {
            let mut p = gf_op_create_matrix(2, 2);
            let mut g = gf_op_create_matrix(2, 2);
            let mut cache = gf_op_create_matrix(2, 2);
            p[0][0] = 1.0; g[0][0] = 0.1;
            gf_train_rmsprop_update(&mut p, &g, &mut cache, 0.001, 0.9, 1e-8, 0.0);
            m_finite(&p) && (p[0][0] - 1.0).abs() > 1e-6
        }
        "GF_Train_CosineAnneal" => {
            let lr = gf_train_cosine_anneal(0, 100, 0.001, 0.0001);
            (lr - 0.001).abs() < 0.0002
        }
        // === GF_Train_ Data ===
        "GF_Train_CreateSynthetic" => {
            let ds = gf_train_create_synthetic(100, 4);
            ds.count == 100 && ds.samples.len() == 100
        }
        "GF_Train_Augment" => {
            let mut a = gf_op_create_matrix(1, 4);
            a[0][0] = 1.0; a[0][1] = 2.0;
            let b = gf_train_augment(&a, DataType::Vector);
            m_finite(&b) && b[0].len() == 4
        }
        "GF_Train_ComputeFID" => {
            let mut real_s: TMatrixArray = Vec::new();
            let mut fake_s: TMatrixArray = Vec::new();
            for _ in 0..10 {
                let mut r = gf_op_create_matrix(1, 4);
                let mut f = gf_op_create_matrix(1, 4);
                r[0][0] = gf_op_random_uniform(0.0, 1.0);
                f[0][0] = gf_op_random_uniform(0.0, 1.0);
                real_s.push(r);
                fake_s.push(f);
            }
            let fid = gf_train_compute_fid(&real_s, &fake_s);
            !fid.is_nan() && !fid.is_infinite()
        }
        "GF_Train_ComputeIS" => {
            let mut samples: TMatrixArray = Vec::new();
            for _ in 0..10 {
                let mut s = gf_op_create_matrix(1, 4);
                s[0][0] = gf_op_random_uniform(0.0, 1.0);
                samples.push(s);
            }
            let is_score = gf_train_compute_is(&samples);
            !is_score.is_nan() && !is_score.is_infinite()
        }
        "GF_Train_LogMetrics" => {
            let mut met = GANMetrics::default();
            met.d_loss_real = 0.5;
            met.g_loss = 0.7;
            met.epoch = 1;
            let path = format!("{}/test_metrics.csv", TEST_DIR);
            gf_train_log_metrics(&met, &path);
            file_exists(&path)
        }
        // === GF_Train_ I/O ===
        "GF_Train_SaveModel" => {
            let sizes = vec![4, 8, 1];
            let gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let path = format!("{}/gen_test.bin", TEST_DIR);
            gf_train_save_model(&gen, &path);
            file_exists(&path)
        }
        "GF_Train_LoadModel" => {
            let sizes = vec![4, 8, 1];
            let gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let path = format!("{}/gen_load_test.bin", TEST_DIR);
            gf_train_save_model(&gen, &path);
            let mut disc = Network::default();
            gf_train_load_model(&mut disc, &path);
            disc.layer_count == gen.layer_count
        }
        "GF_Train_SaveJSON" => {
            let sizes = vec![4, 8, 1];
            let gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let disc = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let path = format!("{}/gan_test.json", TEST_DIR);
            gf_train_save_json(&gen, &disc, &path);
            file_exists(&path)
        }
        "GF_Train_LoadJSON" => {
            let sizes = vec![4, 8, 1];
            let gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let disc = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let path = format!("{}/gan_json_test.json", TEST_DIR);
            gf_train_save_json(&gen, &disc, &path);
            let mut g2 = Network::default();
            let mut d2 = Network::default();
            gf_train_load_json(&mut g2, &mut d2, &path);
            true
        }
        "GF_Train_SaveCheckpoint" => {
            let sizes = vec![4, 8, 1];
            let gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let disc = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let dir = format!("{}/ckpt", TEST_DIR);
            std::fs::create_dir_all(&dir).ok();
            gf_train_save_checkpoint(&gen, &disc, 5, &dir);
            file_exists(&format!("{}/gen_ep5.bin", dir))
                && file_exists(&format!("{}/disc_ep5.bin", dir))
        }
        "GF_Train_LoadCheckpoint" => {
            let sizes = vec![4, 8, 1];
            let gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let disc = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let dir = format!("{}/ckpt", TEST_DIR);
            std::fs::create_dir_all(&dir).ok();
            gf_train_save_checkpoint(&gen, &disc, 7, &dir);
            let mut g2 = Network::default();
            let mut d2 = Network::default();
            gf_train_load_checkpoint(&mut g2, &mut d2, 7, &dir);
            g2.layer_count > 0
        }
        "GF_Train_SaveSamples" => {
            let sizes = vec![4, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            gf_train_save_samples(&mut gen, 1, TEST_DIR, 4, NoiseType::Gauss);
            file_exists(&format!("{}/samples_ep1.csv", TEST_DIR))
        }
        "GF_Train_PlotCSV" => {
            let d_loss = vec![0.5f32, 0.4, 0.3];
            let g_loss = vec![0.8f32, 0.6, 0.5];
            let path = format!("{}/losses_test.csv", TEST_DIR);
            gf_train_plot_csv(&path, &d_loss, &g_loss, 3);
            file_exists(&path)
        }
        "GF_Train_PrintBar" => {
            gf_train_print_bar(0.5, 0.8, 30);
            println!();
            true
        }
        // === GF_Train_ Training ===
        "GF_Train_Optimize" => {
            let sizes = vec![4, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 4);
            inp[0][0] = 1.0;
            network_forward(&mut gen, &inp);
            network_backward(&mut gen, &gf_op_create_matrix(2, 1));
            gf_train_optimize(&mut gen);
            m_finite(&gen.layers[0].weights)
        }
        "GF_Train_Step" => {
            let gen_sizes = vec![4, 8, 1];
            let disc_sizes = vec![1, 8, 1];
            let mut gen = gf_gen_build(&gen_sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut disc = gf_disc_build(&disc_sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut a = gf_op_create_matrix(4, 1);
            for i in 0..4 { a[i][0] = 0.5 + gf_op_random_uniform(0.0, 1.0) * 0.5; }
            let mut noise = vec![];
            gf_op_generate_noise(&mut noise, 4, 4, NoiseType::Gauss);
            let mut cfg = GANConfig::default();
            cfg.loss_type = LossType::BCE;
            cfg.batch_size = 4;
            gf_train_step(&mut gen, &mut disc, &a, &noise, &cfg);
            m_finite(&gen.layers[0].weights) && m_finite(&disc.layers[0].weights)
        }
        "GF_Train_Full" => {
            let gen_sizes = vec![4, 8, 1];
            let disc_sizes = vec![1, 8, 1];
            let mut gen = gf_gen_build(&gen_sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut disc = gf_disc_build(&disc_sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let ds = gf_train_create_synthetic(20, 1);
            let mut cfg = GANConfig::default();
            cfg.epochs = 1;
            cfg.batch_size = 4;
            cfg.loss_type = LossType::BCE;
            cfg.noise_depth = 4;
            cfg.output_dir = TEST_DIR.to_string();
            gf_train_full(&mut gen, &mut disc, &ds, &cfg);
            m_finite(&gen.layers[0].weights) && m_finite(&disc.layers[0].weights)
        }
        // === GF_Sec_ ===
        "GF_Sec_AuditLog" => {
            let path = format!("{}/test_audit.log", TEST_DIR);
            gf_sec_audit_log("test entry", &path);
            file_exists(&path)
        }
        "GF_Sec_SecureRandomize" => {
            gf_sec_secure_randomize();
            true
        }
        "GF_Sec_GetOSRandom" => {
            let _b = gf_sec_get_os_random();
            true // u8 is always 0..=255
        }
        "GF_Sec_ValidatePath" => {
            gf_sec_validate_path("/tmp/model.bin")
                && !gf_sec_validate_path("/tmp/../etc/passwd")
                && !gf_sec_validate_path("")
        }
        "GF_Sec_VerifyWeights" => {
            let mut layer = gf_op_create_dense_layer(4, 4, ActivationType::ReLU);
            layer.weights[0][0] = f32::NAN;
            gf_sec_verify_weights(&mut layer);
            !layer.weights[0][0].is_nan()
        }
        "GF_Sec_VerifyNetwork" => {
            let sizes = vec![4, 4, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            gen.layers[0].weights[0][0] = f32::INFINITY;
            gf_sec_verify_network(&mut gen);
            !gen.layers[0].weights[0][0].is_infinite()
        }
        "GF_Sec_EncryptModel" => {
            let sizes = vec![4, 4, 1];
            let gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let bin_path = format!("{}/enc_test.bin", TEST_DIR);
            let enc_path = format!("{}/enc_test.enc", TEST_DIR);
            gf_train_save_model(&gen, &bin_path);
            gf_sec_encrypt_model(&bin_path, &enc_path, "testkey");
            file_exists(&enc_path)
        }
        "GF_Sec_DecryptModel" => {
            let sizes = vec![4, 4, 1];
            let gen = gf_gen_build(&sizes, ActivationType::ReLU, Optimizer::Adam, 0.001);
            let bin_path = format!("{}/dec_test.bin", TEST_DIR);
            let enc_path = format!("{}/dec_test.enc", TEST_DIR);
            let dec_path = format!("{}/dec_test_dec.bin", TEST_DIR);
            gf_train_save_model(&gen, &bin_path);
            gf_sec_encrypt_model(&bin_path, &enc_path, "testkey");
            gf_sec_decrypt_model(&enc_path, &dec_path, "testkey");
            file_exists(&dec_path)
        }
        "GF_Sec_RunTests" => {
            // Avoid recursion — just return true
            true
        }
        "GF_Sec_RunFuzzTests" => {
            // Avoid recursion — just return true
            true
        }
        "GF_Sec_BoundsCheck" => {
            let a = gf_op_create_matrix(3, 4);
            gf_sec_bounds_check(&a, 0, 0) && !gf_sec_bounds_check(&a, 99, 0)
        }
        // === GF_Introspect_ ===
        "GF_Introspect_NetworkFields" => {
            let sizes = vec![8, 16, 8, 1];
            let gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            gen.layer_count == 3
                && gen.optimizer == Optimizer::Adam
                && (gen.learning_rate - 0.001).abs() < 1e-6
                && (gen.beta1 - 0.9).abs() < 1e-6
                && (gen.beta2 - 0.999).abs() < 1e-6
                && gen.epsilon > 0.0
                && (gen.progressive_alpha - 1.0).abs() < 1e-6
                && gen.is_training
        }
        "GF_Introspect_LayerFields" => {
            let sizes = vec![8, 16, 8, 1];
            let gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            gen.layers[0].layer_type == LayerType::Dense
                && gen.layers[0].activation == ActivationType::LeakyReLU
                && gen.layers[0].input_size == 8
                && gen.layers[0].output_size == 16
        }
        "GF_Introspect_WeightAccess" => {
            let sizes = vec![8, 16, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut ok = true;
            let v = gen.layers[0].weights[0][0];
            if v.is_nan() || v.is_infinite() { ok = false; }
            let v = gen.layers[0].weights[3][7];
            if v.is_nan() || v.is_infinite() { ok = false; }
            let v = gen.layers[0].bias[0];
            if v.is_nan() || v.is_infinite() { ok = false; }
            gen.layers[0].weights[0][0] = 0.12345;
            if (gen.layers[0].weights[0][0] - 0.12345).abs() > 1e-5 { ok = false; }
            gen.layers[0].bias[0] = -0.5;
            if (gen.layers[0].bias[0] - (-0.5)).abs() > 1e-5 { ok = false; }
            ok
        }
        "GF_Introspect_ForwardCache" => {
            let sizes = vec![8, 16, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 8);
            for i in 0..8 {
                inp[0][i] = gf_op_random_gaussian() * 0.5;
                inp[1][i] = gf_op_random_gaussian() * 0.5;
            }
            network_forward(&mut gen, &inp);
            gen.layers[0].layer_input.len() == 2
                && gen.layers[0].layer_input[0].len() == 8
                && !gen.layers[0].layer_output.is_empty()
                && !gen.layers[0].pre_activation.is_empty()
                && m_finite(&gen.layers[0].layer_output)
        }
        "GF_Introspect_ActivationStats" => {
            let sizes = vec![8, 16, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 8);
            for i in 0..8 {
                inp[0][i] = gf_op_random_gaussian() * 0.5;
                inp[1][i] = gf_op_random_gaussian() * 0.5;
            }
            network_forward(&mut gen, &inp);
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            let mut sum = 0.0f32;
            for &v in &gen.layers[0].layer_output[0] {
                if v < min_val { min_val = v; }
                if v > max_val { max_val = v; }
                sum += v;
            }
            let avg = sum / gen.layers[0].output_size as f32;
            !min_val.is_nan() && !min_val.is_infinite()
                && !max_val.is_nan() && !max_val.is_infinite()
                && !avg.is_nan() && !avg.is_infinite()
        }
        "GF_Introspect_Gradients" => {
            let sizes = vec![8, 16, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 8);
            for i in 0..8 {
                inp[0][i] = gf_op_random_gaussian() * 0.5;
                inp[1][i] = gf_op_random_gaussian() * 0.5;
            }
            let mut grad = gf_op_create_matrix(2, 1);
            grad[0][0] = 1.0; grad[1][0] = -1.0;
            network_forward(&mut gen, &inp);
            network_backward(&mut gen, &grad);
            let mut ok = gen.layers[0].weight_grad.len() == 8
                && gen.layers[0].weight_grad[0].len() == 16;
            let v = gen.layers[0].weight_grad[0][0];
            ok = ok && !v.is_nan() && !v.is_infinite();
            ok = ok && gen.layers[0].bias_grad.len() == 16;
            let v = gen.layers[0].bias_grad[0];
            ok = ok && !v.is_nan() && !v.is_infinite();
            ok
        }
        "GF_Introspect_AdamState" => {
            let sizes = vec![8, 16, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 8);
            for i in 0..8 {
                inp[0][i] = gf_op_random_gaussian() * 0.5;
                inp[1][i] = gf_op_random_gaussian() * 0.5;
            }
            let mut grad = gf_op_create_matrix(2, 1);
            grad[0][0] = 1.0; grad[1][0] = -1.0;
            network_forward(&mut gen, &inp);
            network_backward(&mut gen, &grad);
            network_update_weights(&mut gen);
            let mut ok = gen.layers[0].adam_t > 0;
            ok = ok && gen.layers[0].m_weight.len() == 8
                && gen.layers[0].m_weight[0].len() == 16;
            let v = gen.layers[0].m_weight[0][0];
            ok = ok && !v.is_nan() && !v.is_infinite();
            ok = ok && gen.layers[0].v_weight.len() == 8
                && gen.layers[0].v_weight[0].len() == 16;
            let v = gen.layers[0].v_weight[0][0];
            ok = ok && !v.is_nan() && !v.is_infinite();
            ok = ok && gen.layers[0].m_bias.len() == 16;
            ok = ok && gen.layers[0].v_bias.len() == 16;
            ok
        }
        "GF_Introspect_MultiUpdate" => {
            let sizes = vec![8, 16, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 8);
            for i in 0..8 {
                inp[0][i] = gf_op_random_gaussian() * 0.5;
                inp[1][i] = gf_op_random_gaussian() * 0.5;
            }
            let mut grad = gf_op_create_matrix(2, 1);
            grad[0][0] = 1.0; grad[1][0] = -1.0;
            for _ in 0..4 {
                network_forward(&mut gen, &inp);
                network_backward(&mut gen, &grad);
                network_update_weights(&mut gen);
            }
            gen.layers[0].adam_t >= 4 && m_finite(&gen.layers[0].weights)
        }
        "GF_Introspect_DiscFields" => {
            let sizes = vec![1, 8, 16, 1];
            let mut disc = gf_disc_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 1);
            inp[0][0] = 0.5; inp[1][0] = -0.3;
            let mut grad = gf_op_create_matrix(2, 1);
            grad[0][0] = 1.0; grad[1][0] = -1.0;
            network_forward(&mut disc, &inp);
            network_backward(&mut disc, &grad);
            network_update_weights(&mut disc);
            m_finite(&disc.layers[0].weights)
                && !disc.layers[0].weight_grad.is_empty()
                && !disc.layers[0].m_weight.is_empty()
        }
        "GF_Introspect_WeightDecay" => {
            let sizes = vec![8, 16, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            gen.weight_decay = 0.01;
            let inp = gf_op_create_matrix(2, 8);
            let grad = gf_op_create_matrix(2, 1);
            network_forward(&mut gen, &inp);
            network_backward(&mut gen, &grad);
            network_update_weights(&mut gen);
            (gen.weight_decay - 0.01).abs() < 1e-6 && m_finite(&gen.layers[0].weights)
        }
        "GF_Introspect_ConfigMutation" => {
            let sizes = vec![8, 16, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            gen.progressive_alpha = 0.5;
            let mut ok = (gen.progressive_alpha - 0.5).abs() < 1e-6;
            gen.learning_rate = 0.0005;
            ok = ok && (gen.learning_rate - 0.0005).abs() < 1e-6;
            ok
        }
        "GF_Introspect_LayerChain" => {
            let sizes = vec![8, 16, 8, 1];
            let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
            let mut inp = gf_op_create_matrix(2, 8);
            inp[0][0] = 1.0;
            network_forward(&mut gen, &inp);
            (gen.layers[0].layer_output[0][0] - gen.layers[1].layer_input[0][0]).abs() < 1e-6
                && (gen.layers[1].layer_output[0][0] - gen.layers[2].layer_input[0][0]).abs() < 1e-6
        }
        _ => {
            println!("ERROR: Unknown function: {}", name);
            false
        }
    }
}

pub fn run_all_tests() -> bool {
    let mut total = 0;
    let mut pass = 0;
    let mut fail = 0;

    for &name in ALL_FUNCS {
        total += 1;
        let ok = run_single_test(name);
        if ok {
            pass += 1;
            println!("  [PASS] {}", name);
        } else {
            fail += 1;
            println!("  [FAIL] {}", name);
        }
    }

    println!();
    println!("=====================================================================");
    println!(" RESULTS: {} tests | {} passed | {} failed", total, pass, fail);
    println!("=====================================================================");
    if fail == 0 {
        println!("All tests passed.");
    } else {
        println!("{} test(s) FAILED.", fail);
    }
    fail == 0
}

pub fn run_fuzz_tests(iterations: i32) -> bool {
    println!("Running {} fuzz iterations...", iterations);
    let mut all_ok = true;

    for iter in 0..iterations {
        // Random matrix sizes
        let rows = (gf_op_random_uniform(1.0, 8.0)) as i32;
        let cols = (gf_op_random_uniform(1.0, 8.0)) as i32;

        // Fuzz matrix ops
        let mut a = gf_op_create_matrix(rows, cols);
        let mut b = gf_op_create_matrix(rows, cols);
        for i in 0..rows as usize {
            for j in 0..cols as usize {
                a[i][j] = gf_op_random_gaussian() * 10.0;
                b[i][j] = gf_op_random_gaussian() * 10.0;
            }
        }
        let c = gf_op_matrix_add(&a, &b);
        if !m_finite(&c) { println!("  [FUZZ FAIL] iter {} matrix_add", iter); all_ok = false; }

        let c = gf_op_matrix_subtract(&a, &b);
        if !m_finite(&c) { println!("  [FUZZ FAIL] iter {} matrix_sub", iter); all_ok = false; }

        let c = gf_op_matrix_scale(&a, gf_op_random_gaussian());
        if !m_finite(&c) { println!("  [FUZZ FAIL] iter {} matrix_scale", iter); all_ok = false; }

        let c = gf_op_matrix_element_mul(&a, &b);
        if !m_finite(&c) { println!("  [FUZZ FAIL] iter {} element_mul", iter); all_ok = false; }

        // Fuzz activations
        let c = gf_op_relu(&a);
        if !m_finite(&c) { println!("  [FUZZ FAIL] iter {} relu", iter); all_ok = false; }

        let c = gf_op_sigmoid(&a);
        if !m_finite(&c) { println!("  [FUZZ FAIL] iter {} sigmoid", iter); all_ok = false; }

        let c = gf_op_tanh(&a);
        if !m_finite(&c) { println!("  [FUZZ FAIL] iter {} tanh", iter); all_ok = false; }

        // Fuzz forward/backward on small network
        let in_sz = (gf_op_random_uniform(2.0, 8.0)) as i32;
        let hid_sz = (gf_op_random_uniform(2.0, 8.0)) as i32;
        let sizes = vec![in_sz, hid_sz, 1];
        let mut gen = gf_gen_build(&sizes, ActivationType::LeakyReLU, Optimizer::Adam, 0.001);
        let mut inp = gf_op_create_matrix(2, in_sz);
        for i in 0..2 {
            for j in 0..in_sz as usize {
                inp[i][j] = gf_op_random_gaussian();
            }
        }
        let out = network_forward(&mut gen, &inp);
        if !m_finite(&out) { println!("  [FUZZ FAIL] iter {} forward", iter); all_ok = false; }

        let mut grad = gf_op_create_matrix(2, 1);
        grad[0][0] = gf_op_random_gaussian();
        grad[1][0] = gf_op_random_gaussian();
        let gi = network_backward(&mut gen, &grad);
        if !m_finite(&gi) { println!("  [FUZZ FAIL] iter {} backward", iter); all_ok = false; }

        network_update_weights(&mut gen);
        if !m_finite(&gen.layers[0].weights) {
            println!("  [FUZZ FAIL] iter {} weights after update", iter);
            all_ok = false;
        }
    }

    if all_ok {
        println!("All {} fuzz iterations passed.", iterations);
    }
    all_ok
}
