#include <stdio.h>
#include <stdlib.h>
#include <time.h>

float* array_dinamic(int num) {
  time_t t;
  float *ar;
  srand((unsigned) time(&t));
  srand(time(NULL));
  float r = rand() % 999;
  ar = (float *)malloc(num*sizeof(int));
  for (int i = 0; i < num; i++) {
    ar[i] = r;
    r = rand()% 999 ;
  }
  return ar;
}

void file_vector(float *vec,int size ){
	FILE *file = fopen("result_vector.csv", "a");

	if (file == NULL)
	{
		printf("Error al guardar archivo\n");
	}

	for (int i = 0; i < size; ++i)
	{
		fprintf(file, "%.2f ,",vec[i] );
	}
	fprintf(file, "\n");
  	fclose(file);
}


void suma_vectores(int size) {
  float *vec1, *vec2, *result;
  vec1 = array_dinamic(size);
  vec2 = array_dinamic(size);
  result =(float *)malloc(size*sizeof(int));

  for (int j = 0; j < size; j++) {
    result[j] = vec1[j] + vec2[j];
  }

  file_vector(vec1, size);
  file_vector(vec2, size);
  file_vector(result, size);

  printf("EL resultado de la suma se ha guardado con exito en el archivo.\n");
}


float** matrix_dinamic(int rows, int cols) {
  time_t t;
  float **matrix;
  srand(time(NULL));
  float r = rand() % 999;

  matrix = (float **)malloc(rows * sizeof(float*));
  for (int i = 0; i < rows; i++) {
    matrix[i] = (float *)malloc(cols * sizeof(float));
  }

  
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        matrix[i][j] = r;
        r = rand() % 999;
      }
    }
  

  return matrix;
}


void file_matrix (float **matrix, int row, int col){
	FILE *file = fopen("result_matrix.csv", "a");

	if (file == NULL)
	{
		printf("Error al guardar archivo\n");
	}

	for (int i = 0; i < row; ++i)
	{
		for (int j = 0; j < col; ++j)
		{
			fprintf(file, "%.2f, ",matrix[i][j] );
		}
		fprintf(file, "\n");
	}

	fprintf(file, "\n");
  	fclose(file);
}

void multiplicacion_matrices(int rows1, int cols1, int rows2, int cols2 ) {
  float **mat1, **mat2, **result ;

  if (rows1 != cols2) {
    printf("El numero de filas de la primera matriz debe ser ligual al numero de columnas de la segunda matriz\n" );
  }else{
    mat1 = matrix_dinamic(rows1, cols1);
    mat2 = matrix_dinamic(rows2, cols2);
    result = matrix_dinamic(rows1, cols2);

    for (int i = 0; i < rows1; ++i)
    {
    	for (int j = 0; j < cols2; ++j)
    	{
    		
    		for (int k = 0; k < cols1; ++k)
    		{
    			result[i][j] = result[i][j]+mat1[i][k]*mat2[k][j];
    		}
    	}
    }
    file_matrix(mat1, rows1, cols1);
    file_matrix(mat2, rows2, cols2);
    file_matrix(result, rows1, cols2);

    printf("El resultado de la multiplicacion de las matrices se ha realizado con exito en el archivo.\n");

    /*for (int i = 0; i < rows1; ++i)
    {
    	for (int j = 0; j < cols1; ++j)
    	{
    		printf("%.2f ", mat1[i][j]);
    	}
    	printf("\n");
    }

    printf("\n"); 

    for (int i = 0; i < rows2; ++i)
    {
    	for (int j = 0; j < cols2; ++j)
    	{
    		printf("%.2f ", mat2[i][j]);
    	}
    	printf("\n"); 
    }*/	
  } 
}




int main(int argc, char const *argv[]) {
  int size, rows1, rows2, cols1, cols2;

  printf("Ingrese tamaño de los dos vectores a sumar:\n");
  scanf("%d",&size);

  printf("-----------------------------------------------\n");

  printf("Ingrese el numero de filas de la matriz A\n");
  scanf("%d",&rows1);

  printf("Ingrese el numero de columnas de la matriz A\n");
  scanf("%d",&cols1);

  printf("Ingrese el numero de filas de la matriz B\n");
  scanf("%d",&rows2);

  printf("Ingrese el numero de columnas de la matriz B\n");
  scanf("%d",&cols2);

  suma_vectores(size);
  multiplicacion_matrices(rows1,cols1,rows2,cols2);
  
  return 0;
}
