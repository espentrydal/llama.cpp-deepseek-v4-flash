#include "concat.cuh"

// contiguous kernels
static __global__ void concat_f32_dim0(const float * x, const float * y, float * dst, const int ne0, const int ne00) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (nidx < ne00) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne00 +
            blockIdx.z * ne00 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            (nidx - ne00) +
            blockIdx.y * (ne0 - ne00) +
            blockIdx.z * (ne0 - ne00) * gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim1(const float * x, const float * y, float * dst, const int ne0, const int ne01) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.y < (unsigned)ne01) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * ne01;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            (blockIdx.y - ne01) * ne0 +
            blockIdx.z * ne0 * (gridDim.y - ne01);
        dst[offset_dst] = y[offset_src];
    }
}

static __global__ void concat_f32_dim2(const float * x, const float * y, float * dst, const int ne0, const int ne02) {
    int nidx = threadIdx.x + blockIdx.x * blockDim.x;
    if (nidx >= ne0) {
        return;
    }

    int offset_dst =
        nidx +
        blockIdx.y * ne0 +
        blockIdx.z * ne0 * gridDim.y;

    if (blockIdx.z < (unsigned)ne02) { // src0
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            blockIdx.z * ne0 * gridDim.y;
        dst[offset_dst] = x[offset_src];
    } else {
        int offset_src =
            nidx +
            blockIdx.y * ne0 +
            (blockIdx.z - ne02) * ne0 *  gridDim.y;
        dst[offset_dst] = y[offset_src];
    }
}

static void concat_f32_cuda(const float * x, const float * y, float * dst, int ne00, int ne01, int ne02, int ne0, int ne1, int ne2, int dim, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_CONCAT_BLOCK_SIZE - 1) / CUDA_CONCAT_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2);
    if (dim == 0) {
        concat_f32_dim0<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne00);
        return;
    }
    if (dim == 1) {
        concat_f32_dim1<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne01);
        return;
    }
    concat_f32_dim2<<<gridDim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(x, y, dst, ne0, ne02);
}

// non-contiguous kernel (slow)
template <int dim>
static __global__ void __launch_bounds__(CUDA_CONCAT_BLOCK_SIZE)
    concat_f32_non_cont(
        const char * src0,
        const char * src1,
              char * dst,
           int64_t   ne00,
           int64_t   ne01,
           int64_t   ne02,
           int64_t   ne03,
          uint64_t   nb00,
          uint64_t   nb01,
          uint64_t   nb02,
          uint64_t   nb03,
           int64_t /*ne10*/,
           int64_t /*ne11*/,
           int64_t /*ne12*/,
           int64_t /*ne13*/,
          uint64_t   nb10,
          uint64_t   nb11,
          uint64_t   nb12,
          uint64_t   nb13,
           int64_t   ne0,
           int64_t /*ne1*/,
           int64_t /*ne2*/,
           int64_t /*ne3*/,
          uint64_t   nb0,
          uint64_t   nb1,
          uint64_t   nb2,
          uint64_t   nb3){
    static_assert(dim >= 0 && dim <= 3, "dim must be in [0, 3]");

    const int64_t i3 = blockIdx.z;
    const int64_t i2 = blockIdx.y;
    const int64_t i1 = blockIdx.x;

    const float * x;

    for (int64_t i0 = threadIdx.x; i0 < ne0; i0 += blockDim.x) {
        if (i0 < ne00 && i1 < ne01 && i2 < ne02 && i3 < ne03) {
            x = (const float *)(src0 + (i3       )*nb03 + (i2       )*nb02 + (i1       )*nb01 + (i0       )*nb00);
        } else {
            if constexpr (dim == 0) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + i1 * nb11 + (i0 - ne00) * nb10);
            } else if constexpr (dim == 1) {
                x = (const float *) (src1 + i3 * nb13 + i2 * nb12 + (i1 - ne01) * nb11 + i0 * nb10);
            } else if constexpr (dim == 2) {
                x = (const float *) (src1 + i3 * nb13 + (i2 - ne02) * nb12 + i1 * nb11 + i0 * nb10);
            } else if constexpr (dim == 3) {
                x = (const float *) (src1 + (i3 - ne03) * nb13 + i2 * nb12 + i1 * nb11 + i0 * nb10);
            }
        }

        float * y = (float *)(dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

        *y = *x;
    }
}


void ggml_cuda_op_concat(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    const int32_t dim = ((int32_t *) dst->op_params)[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == src0->type);
    GGML_ASSERT(dst->type  == src0->type);

    if (ggml_is_contiguous(src0) && ggml_is_contiguous(src1)) {
        const size_t ts = ggml_type_size(src0->type);
        const char * src0_d = (const char *)src0->data;
        const char * src1_d = (const char *)src1->data;
        char * dst_d = (char *)dst->data;

        // For non-F32, use generic cudaMemcpy-based concat
        if (src0->type != GGML_TYPE_F32) {
            const size_t row_size = dst->ne[0] * ts;
            const size_t src0_row_size = src0->ne[0] * ts;
            const size_t src1_row_size = src1->ne[0] * ts;
            for (int64_t i3 = 0; i3 < dst->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < dst->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < dst->ne[1]; i1++) {
                size_t dst_off = ((i3*dst->ne[2] + i2)*dst->ne[1] + i1) * row_size;
                if (dim == 0) {
                    if (i1 < src0->ne[1] && i2 < src0->ne[2] && i3 < src0->ne[3]) {
                        size_t s0_off = ((i3*src0->ne[2] + i2)*src0->ne[1] + i1) * row_size;
                        CUDA_CHECK(cudaMemcpyAsync(dst_d + dst_off, src0_d + s0_off, src0_row_size, cudaMemcpyDeviceToDevice, stream));
                        CUDA_CHECK(cudaMemcpyAsync(dst_d + dst_off + src0_row_size, src1_d + ((i3*src1->ne[2] + i2)*src1->ne[1] + i1)*src1_row_size, src1_row_size, cudaMemcpyDeviceToDevice, stream));
                    }
                } else if (dim == 1) {
                    if (i1 < src0->ne[1] && i2 < src0->ne[2] && i3 < src0->ne[3])
                        CUDA_CHECK(cudaMemcpyAsync(dst_d + dst_off, src0_d + ((i3*src0->ne[2] + i2)*src0->ne[1] + i1)*row_size, row_size, cudaMemcpyDeviceToDevice, stream));
                    else
                        CUDA_CHECK(cudaMemcpyAsync(dst_d + dst_off, src1_d + ((i3*src1->ne[2] + i2)*src1->ne[1] + (i1-src0->ne[1]))*row_size, row_size, cudaMemcpyDeviceToDevice, stream));
                } else if (dim == 2) {
                    if (i2 < src0->ne[2] && i3 < src0->ne[3])
                        CUDA_CHECK(cudaMemcpyAsync(dst_d + dst_off, src0_d + ((i3*src0->ne[2] + i2)*src0->ne[1] + i1)*row_size, row_size, cudaMemcpyDeviceToDevice, stream));
                    else
                        CUDA_CHECK(cudaMemcpyAsync(dst_d + dst_off, src1_d + ((i3*src1->ne[2] + (i2-src0->ne[2]))*src1->ne[1] + i1)*row_size, row_size, cudaMemcpyDeviceToDevice, stream));
                } else {
                    const size_t src0_plane = src0->ne[0]*src0->ne[1]*ts;
                    const size_t src1_plane = src1->ne[0]*src1->ne[1]*ts;
                    if (i3 < src0->ne[3])
                        CUDA_CHECK(cudaMemcpyAsync(dst_d + dst_off, src0_d + i3*src0_plane + (i2*src0->ne[1]+i1)*row_size, row_size, cudaMemcpyDeviceToDevice, stream));
                    else
                        CUDA_CHECK(cudaMemcpyAsync(dst_d + dst_off, src1_d + (i3-src0->ne[3])*src1_plane + (i2*src1->ne[1]+i1)*row_size, row_size, cudaMemcpyDeviceToDevice, stream));
                }
            }}}
            return;
        }

        const float * src0_d_f = (const float *)src0->data;
        const float * src1_d_f = (const float *)src1->data;
        float * dst_d_f = (float *)dst->data;

        if (dim != 3) {
            for (int i3 = 0; i3 < dst->ne[3]; i3++) {
                concat_f32_cuda(
                        src0_d_f + i3 * (src0->nb[3] / 4),
                        src1_d_f + i3 * (src1->nb[3] / 4),
                        dst_d_f + i3 * ( dst->nb[3] / 4),
                        src0->ne[0], src0->ne[1], src0->ne[2],
                        dst->ne[0],  dst->ne[1],  dst->ne[2], dim, stream);
            }
        } else {
            const size_t size0 = ggml_nbytes(src0);
            const size_t size1 = ggml_nbytes(src1);

            CUDA_CHECK(cudaMemcpyAsync(dst_d_f,         src0_d_f, size0, cudaMemcpyDeviceToDevice, stream));
            CUDA_CHECK(cudaMemcpyAsync(dst_d_f + size0/4, src1_d_f, size1, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        dim3 grid_dim(dst->ne[1], dst->ne[2], dst->ne[3]);
        auto launch_kernel = [&](auto dim) {
            concat_f32_non_cont<dim><<<grid_dim, CUDA_CONCAT_BLOCK_SIZE, 0, stream>>>(
                (const char *) src0->data, (const char *) src1->data, (char *) dst->data,
                src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
                src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
                src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
                src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
                dst->nb[0], dst->nb[1], dst->nb[2], dst->nb[3]);
        };
        switch (dim) {
            case 0:
                launch_kernel(std::integral_constant<int, 0>{});
                break;
            case 1:
                launch_kernel(std::integral_constant<int, 1>{});
                break;
            case 2:
                launch_kernel(std::integral_constant<int, 2>{});
                break;
            case 3:
                launch_kernel(std::integral_constant<int, 3>{});
                break;
            default:
                GGML_ABORT("Invalid dim: %d", dim);
                break;
        }
    }
}
