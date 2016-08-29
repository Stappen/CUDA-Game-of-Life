#include <stdio.h>
#include <iostream>
// CUDA includes
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <device_functions.h>

const int ARRAY_SIZE = 30;

__device__ int getIndex(int x, int y)
{
	return ((y * ARRAY_SIZE) + x);
}

__device__ int mod(int num, int mod)
{
	int ret = a % b;
	if(ret < 0)
		ret+=b;
	return ret;
}

__device__ int isCellAlive(int *d_read)
{
	int up, down, left, right;
	int sm1 = ARRAY_SIZE - 1;		// size-1
	
	up = mod(threadIdx.y - 1, sm1);
	down = mod(threadIdx.y + 1, sm1);
	left = mod(threadIdx.x - 1, sm1);
	right = mod(threadIdx.x + 1, sm1);

	int count = d_read[getIndex(left, up)] + d_read[getIndex(threadIdx.x, up)] + d_read[getIndex(right, up)] + d_read[getIndex(left, threadIdx.y)] + d_read[getIndex(right, threadIdx.y)] + d_read[getIndex(left, down)] + d_read[getIndex(threadIdx.x, down)] + d_read[getIndex(right, down)];

	// check rules of the game
	int rule1,rule2;
	rule1 = ((count == 2) || (count == 3)) && (d_read[getIndex(threadIdx.x, threadIdx.y)]);		// count == 2 or 3 && the current cell == 1
	rule2 = (count == 3) && (d_read[getIndex(threadIdx.x, threadIdx.y)] == 0);					// count == 3 && the current cell == 0

	return (rule1 || rule2);
}

__global__ void simulate(int *d_read, int *d_write)
{
	int i = getIndex(threadIdx.x, threadIdx.y);

	d_write[i] = isCellAlive(d_read);

	// swap values then reset the write buffer
	d_read[i] = d_write[i];
	d_write[i] = 0;
}

void printGrid(int grid[ARRAY_SIZE][ARRAY_SIZE])
{
	for (int y = 0; y < ARRAY_SIZE; y++){
		for (int x = 0; x < ARRAY_SIZE; x++)
		{
			if (grid[x][y] == 1)
			{
				std::cout << " #";
			}
			else
			{
				std::cout << " .";
			}
		}
		std::cout << std::endl;
	}
	std::cout << "\n\n";
}

int main()
{
	// create the host grid used for printing
	int h_read[ARRAY_SIZE][ARRAY_SIZE] = { 0 };

	// glider for testing
	h_read[3][2] = 1;
	h_read[4][3] = 1;
	h_read[2][4] = 1;
	h_read[3][4] = 1;
	h_read[4][4] = 1;

	printGrid(h_read);

	// declate GPU grid pointers
	int *d_read;
	int *d_write;
	// allocate memory in the device's memory space
	cudaMalloc(&d_read, sizeof(int) * (ARRAY_SIZE * ARRAY_SIZE));
	cudaMalloc(&d_write, sizeof(int) * (ARRAY_SIZE * ARRAY_SIZE));

	// copy the input data from the host's memory space to the device's memory space
	cudaMemcpy(d_read, h_read, (sizeof(int) * (ARRAY_SIZE * ARRAY_SIZE)), cudaMemcpyHostToDevice);
	memset(h_read, 0, sizeof(h_read));
	cudaMemcpy(d_write, h_read, (sizeof(int) * (ARRAY_SIZE * ARRAY_SIZE)), cudaMemcpyHostToDevice);

	// run the kernel
	int gen = 10;

	for (int _gen = 0; _gen < gen; _gen++)
	{
		simulate << <1, dim3(ARRAY_SIZE, ARRAY_SIZE) >> >(d_read, d_write);
	}

	// copy the input data from the device's memory space to the host's memory space
	cudaMemcpy(h_read, d_read, (sizeof(int) * (ARRAY_SIZE * ARRAY_SIZE)), cudaMemcpyDeviceToHost);

	printGrid(h_read);

	cudaFree(d_read);
	cudaFree(d_write);
	//std::cin.get();		// stop console from automatically closing (for testing)
	return 0;
}
