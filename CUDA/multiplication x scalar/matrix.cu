#include <stdlib.h>
#include <cuda.h>
#include <stdio.h>

__host__
void llenar(float *d_a, int tam) {
  int n = 10;
  for (int i = 0; i < tam; i++) {
    d_a[i] = n;
  }
}

void print(float *V, int tam){
  for (int i = 0; i < tam; i++) {
    printf("%.2f ", V[i]);
  }
  printf("\n");
}

__global__

void mult_matKernel(float* h_a, float* h_b , int n) {
  int i = threadIdx.x + blockDim.x *blockIdx.x;
  if (i < n) {
    h_b[i] = h_a[i] * 2 ;
  }
}


int main(int argc, char const *argv[]) {
  int n = 100;

  float *h_a = (float*)malloc(n*sizeof(float));
  float *h_b = (float*)malloc(n*sizeof(float));


  cudaError_t error = cudaSuccess;
  float *d_a, *d_b;

  error = cudaMalloc((void**)&d_a, n*sizeof(float));
  if (error != cudaSuccess) {
    printf("Error al asignar espacio a d_a\n" );
    return 0;
  }

  error = cudaMalloc((void**)&d_b, n*sizeof(float));
  if (error != cudaSuccess) {
    printf("Error al asignar espacio a d_b\n" );
    return 0;
  }

  llenar(h_a, n);
  
 cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
//print(d_a, n);
//print(h_a, n);
 dim3 dimGrid(ceil(n/10.0), 1, 1);
  dim3 dimBlock(10,1,1);

  mult_matKernel<<<dimGrid, dimBlock>>>(d_a, d_b, n);
cudaDeviceSynchronize();
  cudaMemcpy(h_b, d_b, n*sizeof(float), cudaMemcpyDeviceToHost);


  print(h_b, n);

  free(h_a);
  free(h_b);
  cudaFree(d_a);
  cudaFree(d_b);

  return 0;
}
