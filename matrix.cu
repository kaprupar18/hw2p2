#include <stdio.h>

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
	int width;
	int height;
	float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C) {

    printf("Running Matmul\n");

	// Load A and B to device memory
	Matrix d_A;
	d_A.width = A.width;
	d_A.height = A.height;
	size_t size = A.width * A.height * sizeof(float);
	cudaMalloc(&d_A.elements, size);
	cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);

	Matrix d_B;
	d_B.width = B.width;
	d_B.height = B.height;
	size = B.width * B.height * sizeof(float);
	cudaMalloc(&d_B.elements, size);
	cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);

	// Allocate C in device memory
	Matrix d_C;
	d_C.width = C.width;
	d_C.height = C.height;
	size = C.width * C.height * sizeof(float);
	cudaMalloc(&d_C.elements, size);

	// Invoke kernel
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);

    for(int i = 0; i < d_A.width * d_A.width; i++){
        //printf("d_A[%d]=%lf, ", i, d_A.elements[i]);
        //printf("d_B[%d]=%lf, ", i, d_B.elements[i]);
        //printf("d_C[%d]=%lf, ", i, d_C.elements[i]);
    }

	MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

	// Read C from device memory
	cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < C.width; i++){
        for(int j = 0; j < C.width; j++){
            printf("%lf, ", A.elements[i * C.width + j]);
        }
        printf("\n");
    }
    for(int i = 0; i < C.width; i++){
        for(int j = 0; j < C.width; j++){
            printf("%lf, ", B.elements[i * C.width + j]);
        }
        printf("\n");
    }
    for(int i = 0; i < C.width; i++){
        for(int j = 0; j < C.width; j++){
            printf("%lf, ", C.elements[i * C.width + j]);
        }
        printf("\n");
    }

	// Free device memory
	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C) {

	// Each thread computes one element of C
	// by accumulating results into Cvalue
	float Cvalue = 0;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
    printf("row %d \n", row);
	for (int e = 0; e < A.width; ++e){
		Cvalue += A.elements[row * A.width + e] * B.elements[e * B.width + col];
    }
	C.elements[row * C.width + col] = Cvalue;
}

int main( int argc, char **argv )
{
    int mat_size = 16;
	Matrix A;
	A.width = mat_size;
	A.height = mat_size;
    A.elements = (float*)malloc(mat_size * mat_size * sizeof(float));
    for(int i = 0; i < mat_size * mat_size; i++){
        A.elements[i] = (float)i*i/47084659.0;
        //printf("A[%d]=%lf, ", i, A.elements[i]);
    }

    Matrix B;
	B.width = mat_size;
	B.height = mat_size;
    B.elements = (float*)malloc(mat_size * mat_size * sizeof(float));
    for(int i = 0; i < mat_size * mat_size; i++){
        B.elements[i] = (float)i / 22360.0;
        //printf("B[%d]=%lf, ", i, B.elements[i]);
    }

    Matrix C;
    C.width = mat_size;
	C.height = mat_size;
    C.elements = (float*)malloc(mat_size * mat_size * sizeof(float));

    MatMul(A, B, C);

    for(int i = 0; i < mat_size; i++){
        for(int j = 0; j < mat_size; j++){
            //printf("%lf, ", C.elements[i * mat_size + j]);
        }
        //printf("\n");
    }

    
}


























