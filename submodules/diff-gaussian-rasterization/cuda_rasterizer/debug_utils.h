#include <cuda.h>
#include "cuda_runtime.h"
#include <iostream>
#include <fstream>

// CUDA_CHECK_ERROR(cudaPeekAtLastError());
// CUDA_CHECK_ERROR(cudaDeviceSynchronize());
#define CUDA_CHECK_ERROR(ans)                 \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

template <typename T>
void saveArray(const T *d_array, const int size, const char *filename)
{
    T *h_array = new T[size];
    cudaMemcpy(h_array, d_array, size * sizeof(T), cudaMemcpyDeviceToHost);
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile)
    {
        std::cerr << "Error opening file for writing!" << std::endl;
        delete[] h_array;
        return;
    }
    outfile.write(reinterpret_cast<char *>(h_array), size * sizeof(T));
    outfile.close();
    std::cout << "save success! " << filename << " size: " << size << std::endl;
    delete[] h_array;
}

void saveFloat32(const float *d_array, const int size, const char *filename)
{
    saveArray(d_array, size, filename);
}

void saveUInt32(const uint32_t *d_array, const int size, const char *filename)
{
    saveArray(d_array, size, filename);
}

void saveInt(const int *d_array, const int size, const char *filename)
{
    saveArray(d_array, size, filename);
}

void saveUintX2(const uint2 *d_array, const int size, const char *filename)
{
    saveArray(reinterpret_cast<const uint32_t*>(d_array), size * 2, filename);
}

void save4Float32(const float4 *d_array, const int size, const char *filename)
{
     saveArray(reinterpret_cast<const float*>(d_array), size * 4, filename);
}
void saveBool(const bool *d_array, const int size, const char *filename)
{
    saveArray(d_array, size, filename);
}