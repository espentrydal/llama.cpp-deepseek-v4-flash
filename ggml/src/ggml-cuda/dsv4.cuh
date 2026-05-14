#include "common.cuh"

void ggml_cuda_op_dsv4_hc_split_sinkhorn(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_weighted_sum(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_hc_expand(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_fp8_kv_quantize(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dsv4_rope_tail(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

bool ggml_cuda_dsv4_hc_split_sinkhorn_supported(const ggml_tensor * dst);
bool ggml_cuda_dsv4_hc_weighted_sum_supported(const ggml_tensor * dst);
bool ggml_cuda_dsv4_hc_expand_supported(const ggml_tensor * dst);
bool ggml_cuda_dsv4_fp8_kv_quantize_supported(const ggml_tensor * dst);
bool ggml_cuda_dsv4_rope_tail_supported(const ggml_tensor * dst);

extern "C" bool ggml_cuda_dsv4_supported_for_rpc(const ggml_tensor * dst);
