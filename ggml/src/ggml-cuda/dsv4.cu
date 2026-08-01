#include "dsv4.cuh"
#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>

struct dsv4_split_args {
    int32_t n_hc;
    int32_t sinkhorn_iters;
    int64_t n_rows;
    int64_t mix_hc;
    uint64_t nb01;
    uint64_t nb1;
    float eps;
};

struct dsv4_weighted_sum_args {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_x0;
    uint64_t nb_x1;
    uint64_t nb_x2;
    uint64_t nb_w0;
    uint64_t nb_w1;
    uint64_t nb0;
    uint64_t nb1;
};

struct dsv4_expand_args {
    int64_t n_embd;
    int64_t n_hc;
    int64_t n_tokens;
    uint64_t nb_block0;
    uint64_t nb_block1;
    uint64_t nb_res0;
    uint64_t nb_res1;
    uint64_t nb_res2;
    uint64_t nb_post0;
    uint64_t nb_post1;
    uint64_t nb_comb0;
    uint64_t nb_comb1;
    uint64_t nb_comb2;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
};

struct dsv4_fp8_args {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_rot;
};

struct dsv4_rope_tail_args {
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    uint64_t nb0;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    int32_t n_dims;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t inverse;
    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;
    bool src2;
};

static __device__ float dsv4_rope_yarn_ramp(const float low, const float high, const int i0) {
    const float y = (i0 / 2 - low) / fmaxf(0.001f, high - low);
    return 1.0f - fminf(1.0f, fmaxf(0.0f, y));
}

static __device__ void dsv4_rope_yarn(
        const float theta_extrap, const float freq_scale, const float corr0, const float corr1,
        const int i0, const float ext_factor, float mscale, float * cos_theta, float * sin_theta) {
    float theta_interp = freq_scale * theta_extrap;
    float theta = theta_interp;
    if (ext_factor != 0.0f) {
        const float ramp_mix = dsv4_rope_yarn_ramp(corr0, corr1, i0) * ext_factor;
        theta = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }
    *cos_theta = cosf(theta) * mscale;
    *sin_theta = sinf(theta) * mscale;
}

static float dsv4_rope_yarn_corr_dim_host(int n_dims, int n_ctx_orig, float n_rot, float base) {
    constexpr float pi = 3.14159265358979323846f;
    return n_dims * logf(n_ctx_orig / (n_rot * 2.0f * pi)) / (2.0f * logf(base));
}

static void dsv4_rope_yarn_corr_dims_host(
        int n_dims, int n_ctx_orig, float freq_base, float beta_fast, float beta_slow, float dims[2]) {
    dims[0] = floorf(dsv4_rope_yarn_corr_dim_host(n_dims, n_ctx_orig, beta_fast, freq_base));
    dims[1] =  ceilf(dsv4_rope_yarn_corr_dim_host(n_dims, n_ctx_orig, beta_slow, freq_base));
}

static __device__ float dsv4_e4m3fn_value(int i) {
    const int exp  = (i >> 3) & 0x0f;
    const int mant = i & 0x07;
    return exp == 0 ? float(mant) * 0.001953125f : (1.0f + float(mant) * 0.125f) * exp2f(float(exp - 7));
}

static __device__ float dsv4_e4m3fn_dequant(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 448.0f);

    int best = 0;
    float best_diff = ax;
    for (int i = 1; i < 127; ++i) {
        const float val = dsv4_e4m3fn_value(i);
        const float diff = fabsf(ax - val);
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }

    return sign * dsv4_e4m3fn_value(best);
}

static __global__ void dsv4_hc_split_sinkhorn_kernel(
        dsv4_split_args args,
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst) {
    const int64_t r = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (r >= args.n_rows) {
        return;
    }

    constexpr int HC_MAX = 16;
    const int HC = args.n_hc;
    if (HC <= 0 || HC > HC_MAX) {
        return;
    }

    const float * mix = (const float *) ((const char *) mixes + r * args.nb01);
    float * out = (float *) ((char *) dst + r * args.nb1);

    const float epsv = args.eps;
    const float pre_scale = scale[0];
    const float post_scale = scale[1];
    const float comb_scale = scale[2];

    for (int i = 0; i < HC; ++i) {
        const float z = mix[i] * pre_scale + base[i];
        out[i] = 1.0f / (1.0f + expf(-z)) + epsv;
    }

    for (int i = 0; i < HC; ++i) {
        const int off = HC + i;
        const float z = mix[off] * post_scale + base[off];
        out[off] = 2.0f / (1.0f + expf(-z));
    }

    float c[HC_MAX * HC_MAX];

    for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
        float row_max = -INFINITY;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const int off = 2 * HC + idx;
            const float v = mix[off] * comb_scale + base[off];
            c[idx] = v;
            row_max = fmaxf(row_max, v);
        }

        float row_sum = 0.0f;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            const float v = expf(c[idx] - row_max);
            c[idx] = v;
            row_sum += v;
        }

        const float inv_sum = 1.0f / row_sum;
        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            const int idx = src_hc + dst_hc * HC;
            c[idx] = c[idx] * inv_sum + epsv;
        }
    }

    for (int src_hc = 0; src_hc < HC; ++src_hc) {
        float sum = 0.0f;
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            sum += c[src_hc + dst_hc * HC];
        }

        const float inv_denom = 1.0f / (sum + epsv);
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            c[src_hc + dst_hc * HC] *= inv_denom;
        }
    }

    for (int iter = 1; iter < args.sinkhorn_iters; ++iter) {
        for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
            float sum = 0.0f;
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                sum += c[src_hc + dst_hc * HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int src_hc = 0; src_hc < HC; ++src_hc) {
                c[src_hc + dst_hc * HC] *= inv_denom;
            }
        }

        for (int src_hc = 0; src_hc < HC; ++src_hc) {
            float sum = 0.0f;
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                sum += c[src_hc + dst_hc * HC];
            }

            const float inv_denom = 1.0f / (sum + epsv);
            for (int dst_hc = 0; dst_hc < HC; ++dst_hc) {
                c[src_hc + dst_hc * HC] *= inv_denom;
            }
        }
    }

    for (int i = 0; i < HC * HC; ++i) {
        out[2 * HC + i] = c[i];
    }
}

static __global__ void dsv4_hc_weighted_sum_kernel(
        dsv4_weighted_sum_args args,
        const char * x,
        const char * weights,
        char * dst) {
    const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n_elem = args.n_embd * args.n_tokens;
    if (gid >= n_elem) {
        return;
    }

    const int64_t d = gid % args.n_embd;
    const int64_t t = gid / args.n_embd;

    float acc = 0.0f;
    for (int64_t h = 0; h < args.n_hc; ++h) {
        const float xv = *((const float *) (x + d * args.nb_x0 + h * args.nb_x1 + t * args.nb_x2));
        const float wv = *((const float *) (weights + h * args.nb_w0 + t * args.nb_w1));
        acc += xv * wv;
    }

    *((float *) (dst + d * args.nb0 + t * args.nb1)) = acc;
}

static __global__ void dsv4_hc_expand_kernel(
        dsv4_expand_args args,
        const char * block_out,
        const char * residual,
        const char * post,
        const char * comb,
        char * dst) {
    const int64_t gid = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    if (gid >= n_elem) {
        return;
    }

    const int64_t d = gid % args.n_embd;
    const int64_t tmp = gid / args.n_embd;
    const int64_t dst_hc = tmp % args.n_hc;
    const int64_t t = tmp / args.n_hc;

    const float block_v = *((const float *) (block_out + d * args.nb_block0 + t * args.nb_block1));
    const float post_v = *((const float *) (post + dst_hc * args.nb_post0 + t * args.nb_post1));

    float acc = block_v * post_v;
    for (int64_t src_hc = 0; src_hc < args.n_hc; ++src_hc) {
        const float comb_v = *((const float *) (comb + dst_hc * args.nb_comb0 + src_hc * args.nb_comb1 + t * args.nb_comb2));
        const float res_v = *((const float *) (residual + d * args.nb_res0 + src_hc * args.nb_res1 + t * args.nb_res2));
        acc += comb_v * res_v;
    }

    *((float *) (dst + d * args.nb0 + dst_hc * args.nb1 + t * args.nb2)) = acc;
}

static __global__ void dsv4_fp8_kv_quantize_kernel(
        dsv4_fp8_args args,
        const char * src0,
        char * dst) {
    const int64_t row = blockIdx.x;
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if (row >= n_rows) {
        return;
    }

    const int64_t i1 = row % args.ne01;
    const int64_t i2 = (row / args.ne01) % args.ne02;
    const int64_t i3 = row / (args.ne01 * args.ne02);
    const char * src_base = src0 + i1 * args.nb01 + i2 * args.nb02 + i3 * args.nb03;
    char * dst_base = dst + i1 * args.nb1 + i2 * args.nb2 + i3 * args.nb3;

    const int64_t n_nope = args.ne00 - args.n_rot;
    __shared__ float scratch[64];

    for (int64_t off = 0; off < n_nope; off += 64) {
        float v = 0.0f;
        if (threadIdx.x < 64) {
            v = *((const float *) (src_base + (off + threadIdx.x) * args.nb00));
            scratch[threadIdx.x] = fabsf(v);
        }
        __syncthreads();

        for (uint32_t stride = 32; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        const float amax = fmaxf(scratch[0], 1.0e-4f);
        const float scale = exp2f(ceilf(log2f(amax / 448.0f)));
        if (threadIdx.x < 64) {
            const float q = dsv4_e4m3fn_dequant(fminf(fmaxf(v / scale, -448.0f), 448.0f)) * scale;
            *((float *) (dst_base + (off + threadIdx.x) * args.nb0)) = q;
        }
        __syncthreads();
    }

    for (int64_t i = n_nope + threadIdx.x; i < args.ne00; i += 64) {
        *((float *) (dst_base + i * args.nb0)) = *((const float *) (src_base + i * args.nb00));
    }
}

static __global__ void dsv4_rope_tail_kernel(
        dsv4_rope_tail_args args,
        const char * src0,
        const char * src1,
        const char * src2,
        char * dst,
        float corr0,
        float corr1) {
    const int i1 = blockIdx.x;
    const int i2 = blockIdx.y;
    const int i3 = blockIdx.z;

    const int n_nope = args.ne00 - args.n_dims;
    if (n_nope < 0) {
        return;
    }

    const int32_t * pos = (const int32_t *) src1;
    const float theta_base = float(pos[i2]);
    const float inv_ndims = -1.0f / args.n_dims;
    const bool is_neox = args.mode == GGML_ROPE_TYPE_NEOX;

    const char * src_base = src0 + i3 * args.nb03 + i2 * args.nb02 + i1 * args.nb01;
    char * dst_base = dst + i3 * args.nb3 + i2 * args.nb2 + i1 * args.nb1;

    for (int i0 = threadIdx.x; i0 < args.ne00; i0 += blockDim.x) {
        if (i0 < n_nope) {
            *((float *) (dst_base + i0 * args.nb0)) = *((const float *) (src_base + i0 * args.nb00));
            continue;
        }

        const int r = i0 - n_nope;
        if (is_neox) {
            const int n_half = args.n_dims / 2;
            if (r >= n_half) {
                continue;
            }

            const int ic = r;
            const int rel_i0 = 2 * ic;
            const float theta = theta_base * powf(args.freq_base, inv_ndims * rel_i0);
            const float freq_factor = args.src2 ? ((const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            dsv4_rope_yarn(theta / freq_factor, args.freq_scale, corr0, corr1, rel_i0,
                args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + ic;
            const int j1 = n_nope + ic + n_half;
            const float x0 = *((const float *) (src_base + j0 * args.nb00));
            const float x1 = *((const float *) (src_base + j1 * args.nb00));

            *((float *) (dst_base + j0 * args.nb0)) = x0 * cos_theta - x1 * sin_theta;
            *((float *) (dst_base + j1 * args.nb0)) = x0 * sin_theta + x1 * cos_theta;
        } else {
            if ((r & 1) != 0) {
                continue;
            }

            const int ic = r / 2;
            const float theta = theta_base * powf(args.freq_base, inv_ndims * r);
            const float freq_factor = args.src2 ? ((const float *) src2)[ic] : 1.0f;

            float cos_theta;
            float sin_theta;
            dsv4_rope_yarn(theta / freq_factor, args.freq_scale, corr0, corr1, r,
                args.ext_factor, args.attn_factor, &cos_theta, &sin_theta);
            if (args.inverse) {
                sin_theta = -sin_theta;
            }

            const int j0 = n_nope + r;
            const int j1 = j0 + 1;
            const float x0 = *((const float *) (src_base + j0 * args.nb00));
            const float x1 = *((const float *) (src_base + j1 * args.nb00));

            *((float *) (dst_base + j0 * args.nb0)) = x0 * cos_theta - x1 * sin_theta;
            *((float *) (dst_base + j1 * args.nb0)) = x0 * sin_theta + x1 * cos_theta;
        }
    }
}

static int div_up_i64(int64_t n, int d) {
    return int((n + d - 1) / d);
}

bool ggml_cuda_dsv4_hc_split_sinkhorn_supported(const ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];
    const int n_hc = ggml_get_op_params_i32(dst, 0);
    const int sinkhorn_iters = ggml_get_op_params_i32(dst, 1);
    return mixes && scale && base &&
        mixes->type == GGML_TYPE_F32 && scale->type == GGML_TYPE_F32 && base->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32 && mixes->nb[0] == sizeof(float) && scale->nb[0] == sizeof(float) &&
        base->nb[0] == sizeof(float) && dst->nb[0] == sizeof(float) &&
        ggml_is_contiguous(scale) && ggml_is_contiguous(base) && ggml_are_same_shape(mixes, dst) &&
        mixes->ne[2] == 1 && mixes->ne[3] == 1 &&
        ggml_nelements(scale) >= 3 && ggml_nelements(base) >= mixes->ne[0] &&
        n_hc > 0 && n_hc <= 16 && sinkhorn_iters > 0 && mixes->ne[0] == (2 + n_hc) * n_hc;
}

bool ggml_cuda_dsv4_hc_weighted_sum_supported(const ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    return x && weights && x->type == GGML_TYPE_F32 && weights->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        x->ne[0] == dst->ne[0] && x->ne[1] == weights->ne[0] && x->ne[2] == dst->ne[1] &&
        weights->ne[1] == dst->ne[1] && x->ne[3] == 1 && weights->ne[2] == 1 &&
        weights->ne[3] == 1 && dst->ne[2] == 1 && dst->ne[3] == 1;
}

bool ggml_cuda_dsv4_hc_expand_supported(const ggml_tensor * dst) {
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];
    return block_out && residual && post && comb &&
        block_out->type == GGML_TYPE_F32 && residual->type == GGML_TYPE_F32 &&
        post->type == GGML_TYPE_F32 && comb->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        block_out->ne[0] == dst->ne[0] && block_out->ne[1] == dst->ne[2] &&
        residual->ne[0] == dst->ne[0] && residual->ne[1] == dst->ne[1] && residual->ne[2] == dst->ne[2] &&
        post->ne[0] == dst->ne[1] && post->ne[1] == dst->ne[2] &&
        comb->ne[0] == dst->ne[1] && comb->ne[1] == dst->ne[1] && comb->ne[2] == dst->ne[2] &&
        block_out->ne[3] == 1 && residual->ne[3] == 1 && post->ne[2] == 1 && post->ne[3] == 1 &&
        comb->ne[3] == 1 && dst->ne[3] == 1;
}

bool ggml_cuda_dsv4_fp8_kv_quantize_supported(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const int64_t n_rot = ggml_get_op_params_i32(dst, 0);
    if (!src0 || src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 || !ggml_are_same_shape(src0, dst)) {
        return false;
    }
    const int64_t n_nope = src0->ne[0] - n_rot;
    return n_rot >= 0 && n_nope > 0 && n_nope % 64 == 0;
}

bool ggml_cuda_dsv4_rope_tail_supported(const ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * src2 = dst->src[2];
    const int n_dims = ggml_get_op_params_i32(dst, 0);
    const int mode = ggml_get_op_params_i32(dst, 1);
    return src0 && src1 && src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_I32 &&
        (!src2 || (src2->type == GGML_TYPE_F32 && src2->nb[0] == sizeof(float) &&
                   src2->ne[0] >= n_dims / 2)) &&
        dst->type == GGML_TYPE_F32 && ggml_are_same_shape(src0, dst) &&
        src0->nb[0] == sizeof(float) && src1->nb[0] == sizeof(int32_t) && dst->nb[0] == sizeof(float) &&
        ggml_is_vector(src1) && src0->ne[2] == src1->ne[0] &&
        n_dims > 0 && n_dims <= src0->ne[0] && n_dims % 2 == 0 &&
        (mode == GGML_ROPE_TYPE_NORMAL || mode == GGML_ROPE_TYPE_NEOX);
}

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_dsv4_hc_split_sinkhorn_supported(dst));
    const ggml_tensor * mixes = dst->src[0];
    dsv4_split_args args = {
        ggml_get_op_params_i32(dst, 0),
        ggml_get_op_params_i32(dst, 1),
        ggml_nrows(mixes),
        mixes->ne[0],
        (uint64_t) mixes->nb[1],
        (uint64_t) dst->nb[1],
        ggml_get_op_params_f32(dst, 2),
    };
    if (args.n_rows == 0) {
        return;
    }
    constexpr int nth = 256;
    dsv4_hc_split_sinkhorn_kernel<<<div_up_i64(args.n_rows, nth), nth, 0, ctx.stream()>>>(
        args, (const float *) mixes->data, (const float *) dst->src[1]->data,
        (const float *) dst->src[2]->data, (float *) dst->data);
}

void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_dsv4_hc_weighted_sum_supported(dst));
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * weights = dst->src[1];
    dsv4_weighted_sum_args args = {
        dst->ne[0], x->ne[1], dst->ne[1],
        (uint64_t) x->nb[0], (uint64_t) x->nb[1], (uint64_t) x->nb[2],
        (uint64_t) weights->nb[0], (uint64_t) weights->nb[1],
        (uint64_t) dst->nb[0], (uint64_t) dst->nb[1],
    };
    constexpr int nth = 256;
    const int64_t n_elem = args.n_embd * args.n_tokens;
    if (n_elem == 0) {
        return;
    }
    dsv4_hc_weighted_sum_kernel<<<div_up_i64(n_elem, nth), nth, 0, ctx.stream()>>>(
        args, (const char *) x->data, (const char *) weights->data, (char *) dst->data);
}

void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_dsv4_hc_expand_supported(dst));
    const ggml_tensor * block_out = dst->src[0];
    const ggml_tensor * residual  = dst->src[1];
    const ggml_tensor * post      = dst->src[2];
    const ggml_tensor * comb      = dst->src[3];
    dsv4_expand_args args = {
        dst->ne[0], dst->ne[1], dst->ne[2],
        (uint64_t) block_out->nb[0], (uint64_t) block_out->nb[1],
        (uint64_t) residual->nb[0], (uint64_t) residual->nb[1], (uint64_t) residual->nb[2],
        (uint64_t) post->nb[0], (uint64_t) post->nb[1],
        (uint64_t) comb->nb[0], (uint64_t) comb->nb[1], (uint64_t) comb->nb[2],
        (uint64_t) dst->nb[0], (uint64_t) dst->nb[1], (uint64_t) dst->nb[2],
    };
    constexpr int nth = 256;
    const int64_t n_elem = args.n_embd * args.n_hc * args.n_tokens;
    if (n_elem == 0) {
        return;
    }
    dsv4_hc_expand_kernel<<<div_up_i64(n_elem, nth), nth, 0, ctx.stream()>>>(
        args, (const char *) block_out->data, (const char *) residual->data,
        (const char *) post->data, (const char *) comb->data, (char *) dst->data);
}

void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_dsv4_fp8_kv_quantize_supported(dst));
    const ggml_tensor * src0 = dst->src[0];
    dsv4_fp8_args args = {
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        (uint64_t) src0->nb[0], (uint64_t) src0->nb[1], (uint64_t) src0->nb[2], (uint64_t) src0->nb[3],
        (uint64_t) dst->nb[0], (uint64_t) dst->nb[1], (uint64_t) dst->nb[2], (uint64_t) dst->nb[3],
        ggml_get_op_params_i32(dst, 0),
    };
    const int64_t n_rows = args.ne01 * args.ne02 * args.ne03;
    if (n_rows == 0) {
        return;
    }
    dsv4_fp8_kv_quantize_kernel<<<n_rows, 64, 0, ctx.stream()>>>(args, (const char *) src0->data, (char *) dst->data);
}

void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_cuda_dsv4_rope_tail_supported(dst));
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src2 = dst->src[2];
    const int64_t n_elem = ggml_nelements(src0);
    if (n_elem == 0) {
        return;
    }
    dsv4_rope_tail_args args = {
        src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
        (uint64_t) src0->nb[0], (uint64_t) src0->nb[1], (uint64_t) src0->nb[2], (uint64_t) src0->nb[3],
        (uint64_t) dst->nb[0], (uint64_t) dst->nb[1], (uint64_t) dst->nb[2], (uint64_t) dst->nb[3],
        ggml_get_op_params_i32(dst, 0),
        ggml_get_op_params_i32(dst, 1),
        ggml_get_op_params_i32(dst, 2),
        ggml_get_op_params_i32(dst, 3),
        0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        src2 != nullptr,
    };
    memcpy(&args.freq_base,   (const int32_t *) dst->op_params + 4, sizeof(float));
    memcpy(&args.freq_scale,  (const int32_t *) dst->op_params + 5, sizeof(float));
    memcpy(&args.ext_factor,  (const int32_t *) dst->op_params + 6, sizeof(float));
    memcpy(&args.attn_factor, (const int32_t *) dst->op_params + 7, sizeof(float));
    memcpy(&args.beta_fast,   (const int32_t *) dst->op_params + 8, sizeof(float));
    memcpy(&args.beta_slow,   (const int32_t *) dst->op_params + 9, sizeof(float));
    float corr[2];
    dsv4_rope_yarn_corr_dims_host(args.n_dims, args.n_ctx_orig, args.freq_base, args.beta_fast, args.beta_slow, corr);

    const int nth = std::min<int64_t>(256, std::max<int64_t>(1, args.ne00));
    const dim3 grid(src0->ne[1], src0->ne[2], src0->ne[3]);
    dsv4_rope_tail_kernel<<<grid, nth, 0, ctx.stream()>>>(
        args, (const char *) src0->data, (const char *) dst->src[1]->data,
        src2 ? (const char *) src2->data : (const char *) src0->data,
        (char *) dst->data, corr[0], corr[1]);
}

extern "C" bool ggml_cuda_dsv4_supported_for_rpc(const ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_DSV4_HC_SPLIT_SINKHORN:
            return ggml_cuda_dsv4_hc_split_sinkhorn_supported(dst);
        case GGML_OP_DSV4_HC_WEIGHTED_SUM:
            return ggml_cuda_dsv4_hc_weighted_sum_supported(dst);
        case GGML_OP_DSV4_HC_EXPAND:
            return ggml_cuda_dsv4_hc_expand_supported(dst);
        case GGML_OP_DSV4_FP8_KV_QUANTIZE:
            return ggml_cuda_dsv4_fp8_kv_quantize_supported(dst);
        case GGML_OP_DSV4_ROPE_TAIL:
            return ggml_cuda_dsv4_rope_tail_supported(dst);
        default:
            return false;
    }
}
