use std::ptr::null_mut;

use bytemuck_derive::{Pod, Zeroable};
use ggml_sys_bleedingedge::{
    ggml_get_rows, ggml_init, ggml_init_params, ggml_new_tensor_1d, ggml_new_tensor_2d,
    ggml_new_tensor_3d, GGML_OBJECT_SIZE, ggml_scratch, ggml_tensor, GGML_TENSOR_SIZE,
    ggml_type, ggml_type_GGML_TYPE_F16, ggml_type_GGML_TYPE_F32, ggml_type_size,
};

use crate::model::ModelContext;

const MB: usize = 1024 * 1024;
const MODEL_MEM_SIZE: usize = 512 * MB;
const MODEL_SCRATCH_SIZE: usize = 1280 * MB;

#[repr(C)]
#[derive(Debug, Copy, Clone, Default, Pod, Zeroable)]
pub struct QwenConfig {
    dtype: ggml_type,
    vocab_size: i32,
    hidden_size: i32,
    num_attention_heads: i32,
    num_kv_heads: i32,
    num_hidden_layers: i32,
    intermediate_size: i32,
    max_length: i32,
    eos_token_id: i32,
    pad_token_id: i32,
    im_start_id: i32,
    im_end_id: i32,
}

#[derive(Default, Debug)]
struct QwenTokenizer {
    eos_token_id: i32,
    im_start_id: i32,
    im_end_id: i32,
}

impl QwenTokenizer {
    fn new(path: &str, config: &QwenConfig) -> Self {
        QwenTokenizer {
            eos_token_id: 0,
            im_start_id: 0,
            im_end_id: 0,
        }
    }

    fn encode(&self, text: &str) -> Vec<i32> {
        todo!()
    }

    fn decode(&self, ids: &Vec<i32>) -> String {
        todo!()
    }
}

struct QwenAttention {
    num_attention_heads: i64,
    num_kv_heads: i64,
    c_attn: Linear,
    c_proj: Linear,
    k_cache: *mut ggml_tensor,
    v_cache: *mut ggml_tensor,
}

impl QwenAttention {
    fn new(
        ctx: &ModelContext,
        hidden_size: i64,
        num_attention_heads: i64,
        num_kv_heads: i64,
        max_length: i64,
    ) -> Self {
        let k_cache = unsafe {
            ggml_new_tensor_3d(
                ctx.ctx_kv,
                ggml_type_GGML_TYPE_F16,
                hidden_size / num_attention_heads,
                max_length,
                num_kv_heads,
            )
        };
        let v_cache = unsafe {
            ggml_new_tensor_3d(
                ctx.ctx_kv,
                ggml_type_GGML_TYPE_F16,
                max_length,
                hidden_size / num_attention_heads,
                num_kv_heads,
            )
        };
        QwenAttention {
            num_attention_heads,
            num_kv_heads,
            c_attn: Linear::new(ctx, hidden_size, 3 * hidden_size, true),
            c_proj: Linear::new(ctx, hidden_size, hidden_size, false),
            k_cache,
            v_cache,
        }
    }
}

struct QwenMLP {
    w1: Linear,
    w2: Linear,
    c_proj: Linear,
}

impl QwenMLP {
    fn new(ctx: &ModelContext, hidden_size: i64, intermediate_size: i64) -> Self {
        QwenMLP {
            w1: Linear::new(ctx, hidden_size, intermediate_size / 2, false),
            w2: Linear::new(ctx, hidden_size, intermediate_size / 2, false),
            c_proj: Linear::new(ctx, intermediate_size / 2, hidden_size, false),
        }
    }
}

struct QwenBlock {
    ln_1: RMSNorm,
    attn: QwenAttention,
    ln_2: RMSNorm,
    mlp: QwenMLP,
}

impl QwenBlock {
    fn new(
        ctx: &ModelContext,
        hidden_size: i64,
        num_attention_heads: i64,
        num_kv_heads: i64,
        intermediate_size: i64,
        max_length: i64,
    ) -> Self {
        QwenBlock {
            ln_1: RMSNorm::new(ctx, hidden_size, false),
            attn: QwenAttention::new(
                ctx,
                hidden_size,
                num_attention_heads,
                num_kv_heads,
                max_length,
            ),
            ln_2: RMSNorm::new(ctx, hidden_size, false),
            mlp: QwenMLP::new(ctx, hidden_size, intermediate_size),
        }
    }
}

pub struct QwenModel {
    wte: Embedding,
    layers: Vec<QwenBlock>,
    ln_f: RMSNorm,
}

impl QwenModel {
    fn new(ctx: &ModelContext, config: &QwenConfig) -> Self {
        let l = config.num_hidden_layers;
        let mut layers: Vec<QwenBlock> = Vec::with_capacity(config.num_hidden_layers as usize);
        for i in 0..config.num_hidden_layers {
            layers.push(QwenBlock::new(
                ctx,
                config.hidden_size as i64,
                config.num_attention_heads as i64,
                config.num_kv_heads as i64,
                config.intermediate_size as i64,
                config.max_length as i64,
            ));
        }
        QwenModel {
            wte: Embedding::new(ctx, config.vocab_size as i64, config.hidden_size as i64),
            layers,
            ln_f: RMSNorm::new(ctx, config.hidden_size as i64, true),
        }
    }
}

pub struct QwenForCausalLM {
    ctx: ModelContext,
    state_dict: Vec<(String, *mut ggml_tensor)>,
    config: QwenConfig,
    transformer: QwenModel,
    lm_head: Linear,
}

impl QwenForCausalLM {
    pub fn new(config: &QwenConfig) -> Self {
        let tensor_ovhd = GGML_TENSOR_SIZE + GGML_OBJECT_SIZE;
        let ctx_w_size = (3 + config.num_hidden_layers as usize * 8) * tensor_ovhd;
        let ctx_w = unsafe {
            ggml_init(ggml_init_params {
                mem_size: ctx_w_size,
                mem_buffer: null_mut(),
                no_alloc: true,
            })
        };

        let ggml_type_size = unsafe { ggml_type_size(ggml_type_GGML_TYPE_F16) };
        let ctx_kv_size = 2
            * config.num_hidden_layers as usize
            * (config.max_length as usize * config.hidden_size as usize
                / config.num_attention_heads as usize
                * config.num_kv_heads as usize
                * ggml_type_size
                + tensor_ovhd);

        let ctx_kv = unsafe {
            ggml_init(ggml_init_params {
                mem_size: ctx_kv_size + 1 * MB,
                mem_buffer: null_mut(),
                no_alloc: false,
            })
        };



        let compute_buffer = vec!['0'; MODEL_MEM_SIZE];
        let mut scratch_buffer = vec!['0'; MODEL_SCRATCH_SIZE];

        let data_ptr: *mut char = scratch_buffer.as_mut_ptr();
        let g_scrath = ggml_scratch {
            offs: 0,
            size: scratch_buffer.len(),
            data: data_ptr as *mut std::os::raw::c_void,
        };
        let context = ModelContext {
            dtype: config.dtype,
            ctx_w,
            ctx_kv,
            ctx_b: None,
            gf: None,
            scratch: g_scrath,
            compute_buffer,
            scratch_buffer,
            work_buffer: vec![],
        };
        let model = QwenModel::new(&context, config);
        let l = Linear::new(&context, config.hidden_size as i64, config.vocab_size as i64, false);

        let mut dict: Vec<(String, *mut ggml_tensor)> =
            Vec::with_capacity(3 + config.num_hidden_layers as usize * 8);

        dict.push(("transformer.wte.weight".to_owned(), model.wte.weight));
        for i in 0..config.num_hidden_layers {
            let layer_prefix = format!("transformer.h.{}.", i);
            let index = i as usize;
            dict.push((format!("{}ln_1.weight", layer_prefix), model.layers[index].ln_1.weight));
            dict.push((format!("{}attn.c_attn.weight", layer_prefix), model.layers[index].attn.c_attn.weight));
            dict.push((format!("{}attn.c_attn.bias", layer_prefix), model.layers[index].attn.c_attn.bias.unwrap_or(null_mut())));
            dict.push((format!("{}attn.c_proj.weight", layer_prefix), model.layers[index].attn.c_proj.weight));
            dict.push((format!("{}ln_2.weight", layer_prefix), model.layers[index].ln_2.weight));
            dict.push((format!("{}mlp.w1.weight", layer_prefix), model.layers[index].mlp.w1.weight));
            dict.push((format!("{}mlp.w2.weight", layer_prefix), model.layers[index].mlp.w2.weight));
            dict.push((format!("{}mlp.c_proj.weight", layer_prefix), model.layers[index].mlp.c_proj.weight));
        }
        dict.push(("transformer.ln_f.weight".to_owned(), model.ln_f.weight));
        dict.push(("lm_head.weight".to_owned(), l.weight));

        QwenForCausalLM {
            ctx: context,
            state_dict: dict,
            config: *config,
            transformer: model,
            lm_head: l,
        }
    }
}

pub struct Linear {
    weight: *mut ggml_tensor,
    bias: Option<*mut ggml_tensor>,
}

impl Linear {
    fn new(ctx: &ModelContext, in_features: i64, out_features: i64, use_bias: bool) -> Self {
        let weight = unsafe { ggml_new_tensor_2d(ctx.ctx_w, ctx.dtype, in_features, out_features) };
        let mut bias = None;
        if use_bias {
            bias = unsafe {
                Some(ggml_new_tensor_1d(
                    ctx.ctx_w,
                    ggml_type_GGML_TYPE_F32,
                    out_features,
                ))
            }
        }
        Linear { weight, bias }
    }

    fn forward(&self) -> *mut ggml_tensor {
        todo!()
    }
}

struct RMSNorm {
    weight: *mut ggml_tensor,
    inplace: bool,
}

impl RMSNorm {
    fn new(ctx: &ModelContext, normalized_shape: i64, inplace: bool) -> Self {
        let w = unsafe { ggml_new_tensor_1d(ctx.ctx_w, ggml_type_GGML_TYPE_F32, normalized_shape) };
        RMSNorm { weight: w, inplace }
    }
}

struct Embedding {
    weight: *mut ggml_tensor,
}

impl Embedding {
    fn new(ctx: &ModelContext, num_embeddings: i64, embedding_dim: i64) -> Self {
        let w = unsafe { ggml_new_tensor_2d(ctx.ctx_w, ctx.dtype, embedding_dim, num_embeddings) };
        Embedding { weight: w }
    }

    fn forward(&self, ctx: &ModelContext, input: *mut ggml_tensor) -> *mut ggml_tensor {
        unsafe { ggml_get_rows(ctx.ctx_b.unwrap(), self.weight, input) }
    }
}
