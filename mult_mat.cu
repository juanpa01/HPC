#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

__host__
void fill_matrix(float *mat, int tam){
	for (int i = 0; i < tam; ++i){
		mat[i] = 2;
	}
}

void print(float *V, int tam){
	for (int i = 0; i < tam; i++) {
    printf("%.2f ", V[i]);
  }
  printf("\n");
  
}

__global__

void matrixKernel(float *d_a, float *d_b, float *d_r, int tam){
	//calculamos los indices de la matrix
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0.0;

	if (col < tam && row < tam)
	{
		for (int i = 0; i < tam; ++i){
			sum = sum + d_a[row*tam+i] * d_b[i*tam+col];
		}
		d_r[row*tam*col] = sum ;
	}
}


int main(int argc, char const *argv[])
{
	int col = 4, row = 4;

	//CPU
	float *A = (float*)malloc(col*row*sizeof(float));
	float *B = (float*)malloc(col*row*sizeof(float));
	float *R = (float*)malloc(col*row*sizeof(float));

	//GPU
	float *d_a, *d_b, *d_r;
	cudaError_t error = cudaSuccess; //manejo de errores

	error = cudaMalloc((void**)&d_a, col*row*sizeof(float));
	if (error != cudaSuccess){
		printf("Error al solicitar espacio de memoria en la gpu para d_a\n");
		return 0;
	}
	error = cudaMalloc((void**)&d_b, col*row*sizeof(float));
	if (error != cudaSuccess){
		printf("Error al solicitar espacio de memoria en la gpu para d_b\n");
		return 0;
	}
	error = cudaMalloc((void**)&d_r, col*row*sizeof(float));
	if (error != cudaSuccess){
		printf("Error al solicitar espacio de memoria en la gpu para d_r\n");
		return 0;
	}

	//llenar las matrices de las variables de cpu
	fill_matrix(A, col*row);
	fill_matrix(B, col*row); 

	//copiar los que tenemos en las variables de cpu a las variables de gpu
	cudaMemcpy(d_a, A, col*row*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, col*row*sizeof(float), cudaMemcpyHostToDevice);


	//ceamos el block y los hilos para pasarle al kernel
	dim3 dimGrid(ceil((col*row)/4.0), 1, 1);
	dim3 dimBlock(4, 1, 1);

	//kernel
	matrixKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_r, col*row);

	//copiamos el resultado desde la variable de la gpa a la variable de la cpu
	cudaMemcpy(R, d_r, col*row*sizeof(float), cudaMemcpyDeviceToHost);

	//imprimimos el resultado
	print(R, col*row);

	printf("\n");

	print(A, col*row);	
	printf("\n");
	print(B, col*row);
	return 0;
}
