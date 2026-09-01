// matmul.cu
// HW1 Part 3 — Matrix multiplication in CUDA C, profiled against a CPU baseline.
// DATA 266, SID4 = 1605, SEED = 1605.
//
// C = A * B, all matrices square N x N, row-major float32.
//
// Kernel design (blocks and threads):
//   - The output matrix C is tiled into square blocks of TILE x TILE threads
//     (TILE = 16, so each thread block has 16*16 = 256 threads).
//   - The grid is sized so that grid.x * TILE >= N and grid.y * TILE >= N,
//     i.e. one thread block covers one TILE x TILE tile of the output matrix,
//     and the grid is a 2D array of blocks covering the whole output matrix.
//   - Each individual thread computes exactly one output element C[row][col],
//     where (row, col) is derived from (blockIdx, blockDim, threadIdx):
//         row = blockIdx.y * TILE + threadIdx.y
//         col = blockIdx.x * TILE + threadIdx.x
//   - Within a block, threads cooperatively stage TILE x TILE sub-tiles of A
//     and B into on-chip __shared__ memory (shared across the block, not the
//     whole grid), then each thread accumulates its dot-product contribution
//     from that shared tile before moving to the next tile along the shared
//     K dimension. This is the standard tiled matmul pattern: it turns most
//     global-memory reads into shared-memory reads, cutting DRAM traffic by
//     roughly a factor of TILE.
//   - __syncthreads() barriers ensure every thread in the block has finished
//     writing shared memory before any thread reads it back (load phase),
//     and that every thread is done reading before the tile is overwritten
//     on the next iteration (compute phase).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cuda_runtime.h>

#define TILE 16

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

__global__ void matmulTiledKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;   // output row this thread owns
    int col = blockIdx.x * TILE + threadIdx.x;   // output col this thread owns

    float acc = 0.0f;
    int numTiles = (N + TILE - 1) / TILE;

    for (int t = 0; t < numTiles; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();  // wait until the whole tile is loaded

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();  // wait until everyone is done reading before next load
    }

    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

static void matmulCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < N; ++k) {
                acc += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

static void fillRandom(float *M, int N, unsigned int seed) {
    srand(seed);
    for (int i = 0; i < N * N; ++i) {
        M[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

static double maxAbsDiff(const float *A, const float *B, int N) {
    double m = 0.0;
    for (int i = 0; i < N * N; ++i) {
        double d = fabs((double)A[i] - (double)B[i]);
        if (d > m) m = d;
    }
    return m;
}

int main(int argc, char **argv) {
    const unsigned int SEED = 1605; // SID4 = 1605
    int sizes[] = {256, 1024, 4096};
    int numSizes = 3;
    bool verify = true;

    if (argc > 1) {
        // allow: ./matmul <N>   to run a single size (used for profiler runs)
        sizes[0] = atoi(argv[1]);
        numSizes = 1;
        verify = (argc > 2) ? atoi(argv[2]) : false; // skip slow CPU verify by default for large N when profiling
    }

    printf("size,cpu_ms,gpu_kernel_ms,h2d_ms,d2h_ms,transfer_ms,end_to_end_gpu_ms,speedup\n");

    for (int s = 0; s < numSizes; ++s) {
        int N = sizes[s];
        size_t bytes = (size_t)N * N * sizeof(float);

        float *hA = (float *)malloc(bytes);
        float *hB = (float *)malloc(bytes);
        float *hC_gpu = (float *)malloc(bytes);
        float *hC_cpu = (float *)malloc(bytes);

        fillRandom(hA, N, SEED);
        fillRandom(hB, N, SEED + 1);

        // ---------------- CPU baseline ----------------
        double cpu_ms = -1.0;
        if (verify || N <= 1024) {
            auto cpuStart = std::chrono::high_resolution_clock::now();
            matmulCPU(hA, hB, hC_cpu, N);
            auto cpuEnd = std::chrono::high_resolution_clock::now();
            cpu_ms = std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
        }

        // ---------------- GPU ----------------
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));

        cudaEvent_t h2dStart, h2dStop, kStart, kStop, d2hStart, d2hStop;
        CUDA_CHECK(cudaEventCreate(&h2dStart));
        CUDA_CHECK(cudaEventCreate(&h2dStop));
        CUDA_CHECK(cudaEventCreate(&kStart));
        CUDA_CHECK(cudaEventCreate(&kStop));
        CUDA_CHECK(cudaEventCreate(&d2hStart));
        CUDA_CHECK(cudaEventCreate(&d2hStop));

        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

        // warm-up run (not timed) — first CUDA call pays context/JIT costs
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
        matmulTiledKernel<<<grid, block>>>(dA, dB, dC, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        // ---- timed H2D ----
        CUDA_CHECK(cudaEventRecord(h2dStart));
        CUDA_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(h2dStop));
        CUDA_CHECK(cudaEventSynchronize(h2dStop));

        // ---- timed kernel (repeat and average) ----
        const int REPS = 10;
        CUDA_CHECK(cudaEventRecord(kStart));
        for (int r = 0; r < REPS; ++r) {
            matmulTiledKernel<<<grid, block>>>(dA, dB, dC, N);
        }
        CUDA_CHECK(cudaEventRecord(kStop));
        CUDA_CHECK(cudaEventSynchronize(kStop));
        CUDA_CHECK(cudaGetLastError());

        // ---- timed D2H ----
        CUDA_CHECK(cudaEventRecord(d2hStart));
        CUDA_CHECK(cudaMemcpy(hC_gpu, dC, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(d2hStop));
        CUDA_CHECK(cudaEventSynchronize(d2hStop));

        float h2d_ms, k_ms_total, d2h_ms;
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, h2dStart, h2dStop));
        CUDA_CHECK(cudaEventElapsedTime(&k_ms_total, kStart, kStop));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, d2hStart, d2hStop));
        float k_ms = k_ms_total / REPS;
        float transfer_ms = h2d_ms + d2h_ms;
        float end_to_end_ms = h2d_ms + k_ms + d2h_ms;

        if (verify || N <= 1024) {
            double diff = maxAbsDiff(hC_cpu, hC_gpu, N);
            fprintf(stderr, "N=%d max|CPU-GPU|=%e\n", N, diff);
        }

        double speedup = (cpu_ms > 0) ? (cpu_ms / end_to_end_ms) : -1.0;
        printf("%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
               N, cpu_ms, k_ms, h2d_ms, d2h_ms, transfer_ms, end_to_end_ms, speedup);

        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB)); CUDA_CHECK(cudaFree(dC));
        free(hA); free(hB); free(hC_gpu); free(hC_cpu);
    }

    return 0;
}
