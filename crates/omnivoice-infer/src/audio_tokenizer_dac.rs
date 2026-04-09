use candle_core::{DType, Module, Tensor};
use candle_nn::{Conv1d, Conv1dConfig, Embedding, Linear, VarBuilder};

use crate::{
    audio_tokenizer::{load_linear, AudioTokenizerModelConfig},
    error::Result,
};

#[derive(Debug)]
pub(crate) struct AcousticEncoder {
    conv1: Conv1d,
    blocks: Vec<AcousticEncoderBlock>,
    snake1: Snake1d,
    conv2: Conv1d,
}

impl AcousticEncoder {
    pub(crate) fn load(config: &AudioTokenizerModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let mut blocks = Vec::with_capacity(config.acoustic_downsampling_ratios.len());
        for (stride_index, stride) in config
            .acoustic_downsampling_ratios
            .iter()
            .copied()
            .enumerate()
        {
            blocks.push(AcousticEncoderBlock::load(
                config,
                stride,
                stride_index + 1,
                vb.pp("block").pp(stride_index),
            )?);
        }
        let d_model = config.acoustic_encoder_hidden_size
            * (1usize << config.acoustic_downsampling_ratios.len());
        Ok(Self {
            conv1: load_conv1d(
                vb.pp("conv1"),
                1,
                config.acoustic_encoder_hidden_size,
                7,
                1,
                3,
                1,
                true,
                1,
            )?,
            blocks,
            snake1: Snake1d::load(d_model, vb.pp("snake1"))?,
            conv2: load_conv1d(
                vb.pp("conv2"),
                d_model,
                config.acoustic_hidden_size,
                3,
                1,
                1,
                1,
                true,
                1,
            )?,
        })
    }

    pub(crate) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut hidden_states = self.conv1.forward(hidden_states)?;
        for block in &self.blocks {
            hidden_states = block.forward(&hidden_states)?;
        }
        let hidden_states = self.snake1.forward(&hidden_states)?;
        self.conv2.forward(&hidden_states).map_err(Into::into)
    }
}

#[derive(Debug)]
struct AcousticEncoderBlock {
    res_unit1: ResidualUnit,
    res_unit2: ResidualUnit,
    res_unit3: ResidualUnit,
    snake1: Snake1d,
    conv1: Conv1d,
}

impl AcousticEncoderBlock {
    fn load(
        config: &AudioTokenizerModelConfig,
        stride: usize,
        stride_index: usize,
        vb: VarBuilder<'_>,
    ) -> Result<Self> {
        let dimension = config.acoustic_encoder_hidden_size * (1usize << stride_index);
        Ok(Self {
            res_unit1: ResidualUnit::load(dimension / 2, 1, vb.pp("res_unit1"))?,
            res_unit2: ResidualUnit::load(dimension / 2, 3, vb.pp("res_unit2"))?,
            res_unit3: ResidualUnit::load(dimension / 2, 9, vb.pp("res_unit3"))?,
            snake1: Snake1d::load(dimension / 2, vb.pp("snake1"))?,
            conv1: load_conv1d(
                vb.pp("conv1"),
                dimension / 2,
                dimension,
                2 * stride,
                stride,
                stride.div_ceil(2),
                1,
                true,
                1,
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = self.res_unit1.forward(hidden_states)?;
        let hidden_states = self.res_unit2.forward(&hidden_states)?;
        let hidden_states = self.res_unit3.forward(&hidden_states)?;
        let hidden_states = self.snake1.forward(&hidden_states)?;
        self.conv1.forward(&hidden_states).map_err(Into::into)
    }
}

#[derive(Debug)]
struct ResidualUnit {
    snake1: Snake1d,
    conv1: Conv1d,
    snake2: Snake1d,
    conv2: Conv1d,
}

impl ResidualUnit {
    fn load(dimension: usize, dilation: usize, vb: VarBuilder<'_>) -> Result<Self> {
        let padding = ((7 - 1) * dilation) / 2;
        Ok(Self {
            snake1: Snake1d::load(dimension, vb.pp("snake1"))?,
            conv1: load_conv1d(
                vb.pp("conv1"),
                dimension,
                dimension,
                7,
                1,
                padding,
                dilation,
                true,
                1,
            )?,
            snake2: Snake1d::load(dimension, vb.pp("snake2"))?,
            conv2: load_conv1d(vb.pp("conv2"), dimension, dimension, 1, 1, 0, 1, true, 1)?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let residual = hidden_states.clone();
        let hidden_states = self.snake1.forward(hidden_states)?;
        let hidden_states = self.conv1.forward(&hidden_states)?;
        let hidden_states = self.snake2.forward(&hidden_states)?;
        let hidden_states = self.conv2.forward(&hidden_states)?;
        let pad = (residual.dim(candle_core::D::Minus1)?
            - hidden_states.dim(candle_core::D::Minus1)?)
            / 2;
        let residual = if pad > 0 {
            residual.narrow(
                candle_core::D::Minus1,
                pad,
                hidden_states.dim(candle_core::D::Minus1)?,
            )?
        } else {
            residual
        };
        residual.broadcast_add(&hidden_states).map_err(Into::into)
    }
}

#[derive(Debug)]
struct Snake1d {
    alpha: Tensor,
}

impl Snake1d {
    fn load(channels: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            alpha: vb.get((1, channels, 1), "alpha")?,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let shape = hidden_states.shape();
        let hidden_states = hidden_states.flatten_from(2)?;
        let sin = self.alpha.broadcast_mul(&hidden_states)?.sin()?;
        let sin = (&sin * &sin)?;
        (hidden_states + (&self.alpha + 1e-9)?.recip()?.broadcast_mul(&sin)?)?
            .reshape(shape.dims())
            .map_err(Into::into)
    }
}

#[derive(Debug)]
pub(crate) struct ResidualVectorQuantizer {
    quantizers: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizer {
    pub(crate) fn load(config: &AudioTokenizerModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let mut quantizers = Vec::with_capacity(config.num_quantizers());
        for index in 0..config.num_quantizers() {
            quantizers.push(VectorQuantizer::load(
                config,
                vb.pp("quantizers").pp(index),
            )?);
        }
        Ok(Self { quantizers })
    }

    pub(crate) fn encode(&self, embeddings: &Tensor) -> Result<Tensor> {
        let mut residual = embeddings.clone();
        let mut all_indices = Vec::with_capacity(self.quantizers.len());
        for quantizer in &self.quantizers {
            let indices = quantizer.encode(&residual)?;
            let quantized = quantizer.decode(&indices)?;
            residual = residual.broadcast_sub(&quantized)?;
            all_indices.push(indices);
        }
        let refs = all_indices.iter().collect::<Vec<_>>();
        Tensor::stack(&refs, 0)?.transpose(0, 1).map_err(Into::into)
    }
}

#[derive(Debug)]
struct VectorQuantizer {
    codebook: Embedding,
    project_in: Linear,
    project_out: Linear,
}

impl VectorQuantizer {
    fn load(config: &AudioTokenizerModelConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            codebook: Embedding::new(
                vb.get(
                    (config.codebook_size, config.codebook_dim),
                    "codebook.embed",
                )?,
                config.codebook_dim,
            ),
            project_in: load_linear(
                vb.pp("project_in"),
                config.acoustic_hidden_size + config.semantic_hidden_size,
                config.codebook_dim,
                true,
            )?,
            project_out: load_linear(
                vb.pp("project_out"),
                config.codebook_dim,
                config.acoustic_hidden_size + config.semantic_hidden_size,
                true,
            )?,
        })
    }

    fn encode(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden_states = hidden_states.transpose(1, 2)?;
        let hidden_states = self.project_in.forward(&hidden_states)?;
        let (batch_size, seq_len, hidden_dim) = hidden_states.dims3()?;
        let hidden_states_flat = hidden_states.reshape((batch_size * seq_len, hidden_dim))?;
        let codebook = self.codebook.embeddings();
        let codebook_t = codebook.t()?;
        let scaled_states = hidden_states_flat
            .sqr()?
            .sum_keepdim(candle_core::D::Minus1)?;
        let dot = hidden_states_flat.matmul(&codebook_t)?;
        let codebook_norm = codebook.sqr()?.sum_keepdim(1)?.transpose(0, 1)?;
        let scores = dot.broadcast_mul(&Tensor::new(2.0f32, hidden_states.device())?)?;
        let dist = scores
            .broadcast_sub(&scaled_states)?
            .broadcast_sub(&codebook_norm)?;
        let indices = dist.argmax(candle_core::D::Minus1)?;
        let indices = indices.reshape((batch_size, seq_len))?;
        indices.to_dtype(DType::I64).map_err(Into::into)
    }

    fn decode(&self, indices: &Tensor) -> Result<Tensor> {
        let quantized = self.codebook.forward(indices)?;
        let quantized = self.project_out.forward(&quantized)?;
        quantized.transpose(1, 2).map_err(Into::into)
    }
}

#[allow(clippy::too_many_arguments)]
fn load_conv1d(
    vb: VarBuilder<'_>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    with_bias: bool,
    groups: usize,
) -> Result<Conv1d> {
    let config = Conv1dConfig {
        stride,
        padding,
        dilation,
        groups,
        ..Default::default()
    };
    let weight = vb.get((out_channels, in_channels / groups, kernel_size), "weight")?;
    let bias = if with_bias {
        Some(vb.get(out_channels, "bias")?)
    } else {
        None
    };
    Ok(Conv1d::new(weight, bias, config))
}
