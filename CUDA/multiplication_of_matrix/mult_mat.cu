#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>

__host__
void fill_matrix(float *mat, FILE *source, int rows, int cols){
	for (int i = 0; i < rows; i++){
		for (int j = 0; j < cols; j++){
			fscanf(source, "%f,", &mat[i * cols + j]);
		}
	}
	fclose(source);
}

void print(float *V, int tam){
	for (int i = 0; i < tam; i++) {
    printf("%.2f ", V[i]);
  }
  printf("\n");

}

__global__

void matrixKernel(float *d_a, float *d_b, float *d_r, int colA, int rowB, int colB, int rowB){
	//calculamos los indices de la matrix
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int row = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0.0;

	if (row < rowA && col < colB) {
		sum = 0.0;
		for (int i = 0; i < rowb; i++) {
			sum += d_a[row*colA+i] * d_b[i*colB+col];
		}
		d_r[row * colB + col] = sum;
	}


	/*if (col < tam && row < tam)
	{
		sum = 0.0;
		for (int i = 0; i < tam; ++i){
			sum += d_a[row*tam+i] * d_b[i*tam+col];
		}
		d_r[row*tam+col] = sum ;
	}*/
}


int main(int argc, char const** argv)
{
	int colA, rowA, colB, rowB ;

	FILE *file_1, *file_2;

	file_1 = fopen(argv[1], "r");
	file_2 = fopen(argv[2], "r");

	fscanf(file_1, "%d", &colA);
	fscanf(file_1, "%d", &rowA);
	fscanf(file_2, "%d", &colB);
	fscanf(file_2, "%d", &rowB);

	if (colA != rowB)
	{
		printf("Es imposible mutiplicar estas matrices\n");
		return 0;
	}

	//CPU
	float *A = (float*)malloc(colA*rowA*sizeof(float));
	float *B = (float*)malloc(colB*rowB*sizeof(float));
	float *R = (float*)malloc(colB*rowA*sizeof(float));

	//GPU
	float *d_a, *d_b, *d_r;
	cudaError_t error = cudaSuccess; //manejo de errores

	error = cudaMalloc((void**)&d_a, colA*rowA*sizeof(float));
	if (error != cudaSuccess){
		printf("Error al solicitar espacio de memoria en la gpu para d_a\n");
		return 0;
	}
	error = cudaMalloc((void**)&d_b, colB*rowB*sizeof(float));
	if (error != cudaSuccess){
		printf("Error al solicitar espacio de memoria en la gpu para d_b\n");
		return 0;
	}
	error = cudaMalloc((void**)&d_r, colB*rowA*sizeof(float));
	if (error != cudaSuccess){
		printf("Error al solicitar espacio de memoria en la gpu para d_r\n");
		return 0;
	}

	//llenar las matrices de las variables de cpu
	fill_matrix(A, file_1, colA, rowA);
	fill_matrix(B, file_2, colB, rowB);

	//copiar los que tenemos en las variables de cpu a las variables de gpu
	cudaMemcpy(d_a, A, colA*rowA*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, B, colB*rowB*sizeof(float), cudaMemcpyHostToDevice);


	//ceamos el block y los hilos para pasarle al kernel
	dim3 dimGrid(ceil((colB*rowA)/4.0), 1, 1);
	dim3 dimBlock(4, 1, 1);

	//kernel
	matrixKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_r, colA, rowA, colB, rowB);

	//copiamos el resultado desde la variable de la gpa a la variable de la cpu
	cudaMemcpy(R, d_r, colB*rowA*sizeof(float), cudaMemcpyDeviceToHost);

	//imprimimos el resultado

	printf("Matriz A\n");
	print(A, colA*rowA);
	printf("\n");
	printf("MAtriz B\n");
	print(B, colB*rowB);
	printf("\n");
	printf("Matriz Resultado\n");
	print(R, colB*rowA);



	//liberar espacio en memoria
	free(A); free(B); free(R);
	cudaFree(d_a); cudaFree(d_b); cudaFree(d_r);
	return 0;
}
