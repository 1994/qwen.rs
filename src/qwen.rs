use std::collections::HashMap;
use std::ffi::c_int;
use std::fmt::Debug;
use std::fs;
use std::ptr::null_mut;

use base64::engine::general_purpose;
use base64::Engine;
use bytemuck_derive::{Pod, Zeroable};
use ggml_sys_bleedingedge::{
    ggml_add_inplace, ggml_build_forward_expand, ggml_cont, ggml_cpy, ggml_diag_mask_inf_inplace,
    ggml_element_size, ggml_get_rows, ggml_init, ggml_init_params, ggml_mul_inplace, ggml_mul_mat,
    ggml_new_f32, ggml_new_tensor_1d, ggml_new_tensor_2d, ggml_new_tensor_3d, ggml_permute,
    ggml_reshape_2d, ggml_reshape_3d, ggml_rms_norm, ggml_rms_norm_inplace, ggml_rope_inplace,
    ggml_scale_inplace, ggml_scratch, ggml_set_scratch, ggml_silu_inplace, ggml_soft_max_inplace,
    ggml_tensor, ggml_type, ggml_type_GGML_TYPE_F16, ggml_type_GGML_TYPE_F32, ggml_type_size,
    ggml_view_1d, ggml_view_3d, GGML_OBJECT_SIZE, GGML_TENSOR_SIZE,
};
use tiktoken_rs::CoreBPE;

use crate::model::ModelContext;

const MB: usize = 1024 * 1024;
const MODEL_MEM_SIZE: usize = 512 * MB;
const MODEL_SCRATCH_SIZE: usize = 1280 * MB;
const TOKEN_REGEX: &str = r"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

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

#[derive(Debug)]
pub struct QwenTokenizer {
    eos_token_id: i32,
    im_start_id: i32,
    im_end_id: i32,
    bpe: CoreBPE,
}

impl QwenTokenizer {
    pub fn new(path: &str, config: &QwenConfig) -> anyhow::Result<Self> {
        let file = fs::read_to_string(path)?;
        let mut encoder = HashMap::default();
        for line in file.lines() {
            let (left, right) = parse(line)?;
            encoder.insert(left, right);
        }

        let mut special_tokens_s = vec![
            "<|endoftext|>".to_owned(),
            "<|im_start|>".to_owned(),
            "<|im_end|>".to_owned(),
        ];
        for i in 0..205 {
            let v = format!("<|extra_{}|>", i);
            special_tokens_s.push(v);
        }
        let mut special_tokens = HashMap::default();
        let encoder_size = encoder.len();
        for (i, token) in special_tokens_s.iter().enumerate() {
            special_tokens.insert(token.clone(), encoder_size + i);
        }

        let bpe = CoreBPE::new(encoder, special_tokens, TOKEN_REGEX)?;
        Ok(QwenTokenizer {
            eos_token_id: config.eos_token_id,
            im_start_id: config.im_start_id,
            im_end_id: config.im_end_id,
            bpe,
        })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.bpe.encode_with_special_tokens(text)
    }

    pub fn decode(&self, ids: Vec<usize>) -> String {
        let p = ids
            .into_iter()
            .filter(|id| {
                *id != self.im_start_id as usize
                    && *id != self.im_end_id as usize
                    && *id != self.eos_token_id as usize
            })
            .collect();
        self.bpe.decode(p).expect("decode error")
    }
}

#[derive(Debug)]
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

    fn forward(
        &self,
        ctx: &ModelContext,
        hidden_states: *mut ggml_tensor,
        n_past: i64,
    ) -> *mut ggml_tensor {
        let cb = ctx.ctx_b.unwrap();
        let hidden_size = unsafe { (*hidden_states).ne[0] };
        let qlen = unsafe { (*hidden_states).ne[1] };
        let head_size = hidden_size / self.num_attention_heads;
        let rope_dim = head_size;
        let mqa_scale = self.num_attention_heads / self.num_kv_heads;
        let qkv = self.c_attn.forward(ctx, hidden_states);

        let mut query_layer = unsafe {
            let qkv_size = ggml_element_size(qkv);
            ggml_view_3d(
                cb,
                qkv,
                head_size,
                self.num_attention_heads,
                qlen,
                head_size as usize * qkv_size,
                (*qkv).nb[1],
                0,
            )
        };
        query_layer =
            unsafe { ggml_rope_inplace(cb, query_layer, n_past as c_int, rope_dim as c_int, 2, 0) };
        query_layer = unsafe {
            let p = ggml_permute(cb, query_layer, 0, 2, 1, 3);
            ggml_cont(cb, p)
        };
        query_layer = unsafe {
            ggml_reshape_3d(
                cb,
                query_layer,
                head_size,
                mqa_scale * qlen,
                self.num_kv_heads,
            )
        };

        let mut key_layer = unsafe {
            let qkv_size = ggml_element_size(qkv);
            ggml_view_3d(
                cb,
                qkv,
                head_size,
                self.num_kv_heads,
                qlen,
                head_size as usize * qkv_size,
                (*qkv).nb[1],
                hidden_size as usize * qkv_size,
            )
        };
        key_layer =
            unsafe { ggml_rope_inplace(cb, key_layer, n_past as c_int, rope_dim as c_int, 2, 0) };
        key_layer = unsafe { ggml_permute(cb, key_layer, 0, 2, 1, 3) };

        let mut value_layer = unsafe {
            let qkv_size = ggml_element_size(qkv);
            ggml_view_3d(
                cb,
                qkv,
                head_size,
                self.num_kv_heads,
                qlen,
                head_size as usize * qkv_size,
                (*qkv).nb[1],
                (hidden_size as usize + head_size as usize * self.num_kv_heads as usize) * qkv_size,
            )
        };
        value_layer = unsafe { ggml_permute(cb, value_layer, 1, 2, 0, 3) };

        unsafe {
            let k_cache_size = ggml_element_size(self.k_cache);
            let k_cache_view = ggml_view_3d(
                cb,
                self.k_cache,
                head_size,
                qlen,
                self.num_kv_heads,
                (*self.k_cache).nb[1],
                (*self.k_cache).nb[2],
                n_past as usize * head_size as usize * k_cache_size,
            );

            let cpy = ggml_cpy(cb, key_layer, k_cache_view);
            // todo gf init?
            ggml_build_forward_expand(ctx.gf.unwrap(), cpy)
        };

        unsafe {
            let v_cache_size = ggml_element_size(self.v_cache);
            let v_cache_view = ggml_view_3d(
                cb,
                self.v_cache,
                qlen,
                head_size,
                self.num_kv_heads,
                (*self.v_cache).nb[1],
                (*self.v_cache).nb[2],
                n_past as usize * v_cache_size,
            );

            let cpy = ggml_cpy(cb, value_layer, v_cache_view);
            ggml_build_forward_expand(ctx.gf.unwrap(), cpy);
        }

        key_layer = unsafe {
            ggml_view_3d(
                cb,
                self.k_cache,
                head_size,
                n_past + qlen,
                self.num_kv_heads,
                (*self.k_cache).nb[1],
                (*self.k_cache).nb[2],
                0,
            )
        };

        value_layer = unsafe {
            ggml_view_3d(
                cb,
                self.v_cache,
                n_past + qlen,
                head_size,
                self.num_kv_heads,
                (*self.v_cache).nb[1],
                (*self.v_cache).nb[2],
                0,
            )
        };
        let mut atten_scores = unsafe { ggml_mul_mat(cb, key_layer, query_layer) };

        atten_scores = unsafe {
            let v = ggml_new_f32(cb, 1. / (head_size as f32).sqrt());
            ggml_scale_inplace(cb, atten_scores, v)
        };

        if n_past == 0 {
            unsafe {
                atten_scores = ggml_reshape_3d(
                    cb,
                    atten_scores,
                    n_past + qlen,
                    qlen,
                    self.num_attention_heads,
                );
                atten_scores = ggml_diag_mask_inf_inplace(cb, atten_scores, n_past as c_int);
                atten_scores = ggml_reshape_3d(
                    cb,
                    atten_scores,
                    n_past + qlen,
                    mqa_scale * qlen,
                    self.num_kv_heads,
                );
            }
        }
        let attn_probs = unsafe { ggml_soft_max_inplace(cb, atten_scores) };

        let mut context_layer = unsafe { ggml_mul_mat(cb, value_layer, attn_probs) };

        context_layer = unsafe {
            ggml_reshape_3d(cb, context_layer, head_size, qlen, self.num_attention_heads)
        };

        context_layer = unsafe {
            let v = ggml_permute(cb, context_layer, 0, 2, 1, 3);
            ggml_cont(cb, v)
        };
        context_layer = unsafe { ggml_reshape_2d(cb, context_layer, hidden_size, qlen) };
        self.c_proj.forward(ctx, context_layer)
    }
}

#[derive(Debug)]
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
    fn forward(&self, ctx: &ModelContext, hidden_states: *mut ggml_tensor) -> *mut ggml_tensor {
        let cb = ctx.ctx_b.unwrap();
        let mut a2 = self.w2.forward(ctx, hidden_states);
        a2 = unsafe { ggml_silu_inplace(cb, a2) };

        let a1 = self.w1.forward(ctx, hidden_states);
        let output = unsafe { ggml_mul_inplace(cb, a2, a1) };

        self.c_proj.forward(ctx, output)
    }
}

#[derive(Debug)]
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

    fn forward(
        &self,
        ctx: &ModelContext,
        hidden_states: *mut ggml_tensor,
        n_past: i64,
    ) -> *mut ggml_tensor {
        let cb = ctx.ctx_b.unwrap();
        let mut residual = hidden_states;
        let mut temp = hidden_states;
        temp = self.ln_1.forward(ctx, temp, 1e-6);
        temp = self.attn.forward(ctx, temp, n_past);
        temp = unsafe { ggml_add_inplace(cb, hidden_states, residual) };

        residual = temp;
        temp = self.ln_2.forward(ctx, temp, 1e-6);
        temp = self.mlp.forward(ctx, temp);
        temp = unsafe { ggml_add_inplace(cb, temp, residual) };
        temp
    }
}

#[derive(Debug)]
pub struct QwenModel {
    wte: Embedding,
    layers: Vec<QwenBlock>,
    ln_f: RMSNorm,
}

impl QwenModel {
    fn new(ctx: &ModelContext, config: &QwenConfig) -> Self {
        let mut layers: Vec<QwenBlock> = Vec::with_capacity(config.num_hidden_layers as usize);
        for _ in 0..config.num_hidden_layers {
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

    fn forward(
        &self,
        ctx: &ModelContext,
        input_ids: *mut ggml_tensor,
        n_past: i64,
    ) -> *mut ggml_tensor {
        let cb = ctx.ctx_b.unwrap();
        let mut hidden_states = self.wte.forward(ctx, input_ids);
        for layer in &self.layers {
            unsafe {
                ggml_set_scratch(cb, ctx.scratch);
            }
            hidden_states = layer.forward(ctx, hidden_states, n_past);
        }

        unsafe {
            let empty = ggml_scratch {
                offs: 0,
                size: 0,
                data: null_mut(),
            };
            ggml_set_scratch(cb, empty);
        };
        self.ln_f.forward(ctx, hidden_states, 1e-6)
    }
}

#[derive(Debug)]
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
                mem_size: ctx_kv_size + MB,
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
        let l = Linear::new(
            &context,
            config.hidden_size as i64,
            config.vocab_size as i64,
            false,
        );

        let mut dict: Vec<(String, *mut ggml_tensor)> =
            Vec::with_capacity(3 + config.num_hidden_layers as usize * 8);

        dict.push(("transformer.wte.weight".to_owned(), model.wte.weight));
        for i in 0..config.num_hidden_layers {
            let layer_prefix = format!("transformer.h.{}.", i);
            let index = i as usize;
            dict.push((
                format!("{}ln_1.weight", layer_prefix),
                model.layers[index].ln_1.weight,
            ));
            dict.push((
                format!("{}attn.c_attn.weight", layer_prefix),
                model.layers[index].attn.c_attn.weight,
            ));
            dict.push((
                format!("{}attn.c_attn.bias", layer_prefix),
                model.layers[index].attn.c_attn.bias.unwrap_or(null_mut()),
            ));
            dict.push((
                format!("{}attn.c_proj.weight", layer_prefix),
                model.layers[index].attn.c_proj.weight,
            ));
            dict.push((
                format!("{}ln_2.weight", layer_prefix),
                model.layers[index].ln_2.weight,
            ));
            dict.push((
                format!("{}mlp.w1.weight", layer_prefix),
                model.layers[index].mlp.w1.weight,
            ));
            dict.push((
                format!("{}mlp.w2.weight", layer_prefix),
                model.layers[index].mlp.w2.weight,
            ));
            dict.push((
                format!("{}mlp.c_proj.weight", layer_prefix),
                model.layers[index].mlp.c_proj.weight,
            ));
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
    fn forward(
        &self,
        ctx: &ModelContext,
        input_ids: *mut ggml_tensor,
        n_past: i64,
        // n_ctx: i32,
    ) -> *mut ggml_tensor {
        let mut output = self.transformer.forward(ctx, input_ids, n_past);
        unsafe {
            if (*input_ids).ne[0] > 1 {
                let output_size = ggml_element_size(output);
                output = ggml_view_1d(
                    ctx.ctx_b.unwrap(),
                    output,
                    self.config.hidden_size as i64,
                    ((*input_ids).ne[0] - 1) as usize
                        * self.config.hidden_size as usize
                        * output_size,
                );
            }
        }
        self.lm_head.forward(ctx, output)
    }

    fn generate_next_token(&self) {}
}

#[derive(Debug)]
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

    fn forward(&self, ctx: &ModelContext, input: *mut ggml_tensor) -> *mut ggml_tensor {
        let mut output = unsafe { ggml_mul_mat(ctx.ctx_b.unwrap(), self.weight, input) };
        if self.bias.is_some() {
            output = unsafe { ggml_add_inplace(ctx.ctx_b.unwrap(), output, self.bias.unwrap()) };
        }
        output
    }
}

#[derive(Debug)]
struct RMSNorm {
    weight: *mut ggml_tensor,
    inplace: bool,
}

impl RMSNorm {
    fn new(ctx: &ModelContext, normalized_shape: i64, inplace: bool) -> Self {
        let w = unsafe { ggml_new_tensor_1d(ctx.ctx_w, ggml_type_GGML_TYPE_F32, normalized_shape) };
        RMSNorm { weight: w, inplace }
    }

    fn forward(&self, ctx: &ModelContext, input: *mut ggml_tensor, eps: f32) -> *mut ggml_tensor {
        let cb = ctx.ctx_b.unwrap();
        let inplace = self.inplace;
        let output = if inplace {
            unsafe { ggml_rms_norm_inplace(cb, input, eps) }
        } else {
            unsafe { ggml_rms_norm(cb, input, eps) }
        };
        unsafe { ggml_mul_inplace(cb, output, self.weight) }
    }
}

#[derive(Debug)]
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

fn parse(line: &str) -> anyhow::Result<(Vec<u8>, usize)> {
    let v: Vec<&str> = line.split(' ').collect();
    let left = v[0];
    let decode = general_purpose::STANDARD.decode(left)?;
    let value = v[1].parse::<usize>()?;
    Ok((decode, value))
}
