//ONLY MODIFY THIS FILE!
//YOU CAN MODIFY EVERYTHING IN THIS FILE!
//This code created by Mohammad H Najafi in June 2023

#include "fft.h"

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

// This kernel performs the bit-reversal permutation necessary for the radix-2 algorithm.
__global__ void radix2reorder(float* x_r_d,float* x_i_d,const unsigned int N, unsigned int M)
{
    int thrds = ( gridDim.x * gridDim.y * bz + gridDim.x * by + bx) * blockDim.x  + tx;
	unsigned int ic = thrds;
	unsigned int temp = 0;
	float R_temp[2];
	float I_temp[2];

	// Perform bit-reversal permutation
    for (int k = 0; k < 32; k++) {
        temp <<= 1;        
        temp |= (ic & 1); 
        ic >>= 1;                
    }
    ic = temp >> (32 - M);
	
	if(thrds < ic)
	{
		// Swap elements in the input signal
		I_temp[0] = x_i_d[ic];
		R_temp[0] = x_r_d[ic];
		I_temp[1] = x_i_d[thrds];
		R_temp[1] = x_r_d[thrds];
		x_i_d[thrds] = I_temp[0];
		x_r_d[thrds] = R_temp[0];
		x_i_d[ic] = I_temp[1];
		x_r_d[ic] = R_temp[1];
	}
}
// This kernel performs the butterfly operations for each stage of the radix-2 FFT.
__global__ void radix2(float* x_r_d, float* x_i_d ,const unsigned int N, unsigned int M) 
{
    int thrds = ( gridDim.x * gridDim.y * bz + gridDim.x * by + bx) * blockDim.x  + tx;
	float R_temp[2],I_temp[2];	
	unsigned int a = thrds + (thrds/M) * M;
	unsigned int b = thrds + (thrds/M) * M + M;

	I_temp[0] = x_i_d[a];
	R_temp[0] = x_r_d[a];
	I_temp[1] = x_i_d[b];
	R_temp[1] = x_r_d[b];	

	// Compute the angle for the butterfly operation
	float angle = - 2 * PI * ( (N/(M * 2)) * thrds  - (N/2) * (thrds/M) ) / N;
	float m = cos(angle);
	float n = sin(angle);
	
	// Perform the butterfly operation
	x_i_d[ a ] = I_temp[0] + R_temp[1] * n + I_temp[1] * m;
	x_r_d[ a ] = R_temp[0] + R_temp[1] * m - I_temp[1] * n;
	x_r_d[ b ] = R_temp[0] - R_temp[1] * m + I_temp[1] * n;
	x_i_d[ b ] = I_temp[0] - R_temp[1] * n - I_temp[1] * m;				



}

// This kernel performs the bit-reversal permutation necessary for the radix-4 algorithm.
__global__ void radix4reorder(float* x_r_d,float* x_i_d,const unsigned int N, unsigned int M)
{
    int thrds = ( gridDim.x * gridDim.y * bz + gridDim.x * by + bx) * blockDim.x  + tx;
	unsigned int ic = thrds;	
	unsigned int temp = 0;
	float R_temp[2];
	float I_temp[2];

	// Perform bit-reversal permutation
    for (int k = 0; k < 32; k += 2) {
        unsigned int bit1 = (ic >> k) & 0x01;
        unsigned int bit2 = (ic >> (k + 1)) & 0x01;
        temp |= (bit2 << (31 - k));
        temp |= (bit1 << (30 - k));
    }
    ic = temp >> (32 - M);
    		
	if(thrds < ic)
	{
		// Swap elements in the input signal
		I_temp[0] = x_i_d[ic];
		R_temp[0] = x_r_d[ic];
		I_temp[1] = x_i_d[thrds];
		R_temp[1] = x_r_d[thrds];
		x_i_d[thrds] = I_temp[0];
		x_r_d[thrds] = R_temp[0];
		x_i_d[ic] = I_temp[1];
		x_r_d[ic] = R_temp[1];
	}
}
// This kernel performs the butterfly operations for each stage of the radix-4 FFT.
__global__ void radix4(float* x_r_d, float* x_i_d, const unsigned int N, const unsigned int M) 
{
    int thrds = ( gridDim.x * gridDim.y * bz + gridDim.x * by + bx) * blockDim.x  + tx;
	float R_temp[4],I_temp[4];	
	float Re[4] , Im[4];
	unsigned int a[4];
	a[ 0 ] = (thrds/M) * 4*M + thrds%M + 0*M;
	a[ 1 ] = (thrds/M) * 4*M + thrds%M + 1*M;
	a[ 2 ] = (thrds/M) * 4*M + thrds%M + 2*M;
	a[ 3 ] = (thrds/M) * 4*M + thrds%M + 3*M;
		
	I_temp[0] = x_i_d[a[0]];
	R_temp[0] = x_r_d[a[0]];
	I_temp[1] = x_i_d[a[1]];
	R_temp[1] = x_r_d[a[1]];
	I_temp[2] = x_i_d[a[2]];
	R_temp[2] = x_r_d[a[2]];
	I_temp[3] = x_i_d[a[3]];	
	R_temp[3] = x_r_d[a[3]];

	// Compute the angle for the butterfly operations
	float angle  = - 2 * PI * ( thrds%M ) / ( 4*M );	
	float m[3] , n[3];
	m[0] = cos(angle);
	m[1] = cos(angle * 2);
	m[2] = cos(angle * 3);
	n[0] = sin(angle);
	n[1] = sin(angle * 2);
	n[2] = sin(angle * 3);

	Im[0] = I_temp[0];
	Re[0] = R_temp[0];
	Im[1] = R_temp[1] * n[0] + I_temp[1] * m[0];
	Re[1] = R_temp[1] * m[0] - I_temp[1] * n[0];
	Im[2] = R_temp[2] * n[1] + I_temp[2] * m[1];
	Re[2] = R_temp[2] * m[1] - I_temp[2] * n[1];
	Im[3] = R_temp[3] * n[2] + I_temp[3] * m[2];
	Re[3] = R_temp[3] * m[2] - I_temp[3] * n[2];
		
	// Perform the butterfly operations
	x_i_d[a[0]] = Im[0] + Im[1] + Im[2] + Im[3];
	x_r_d[a[0]] = Re[0] + Re[1] + Re[2] + Re[3];
	x_i_d[a[1]] = Im[0] - Re[1] - Im[2] + Re[3];
	x_r_d[a[1]] = Re[0] + Im[1] - Re[2] - Im[3];
	x_i_d[a[2]] = Im[0] - Im[1] + Im[2] - Im[3];
	x_r_d[a[2]] = Re[0] - Re[1] + Re[2] - Re[3];
	x_i_d[a[3]] = Im[0] + Re[1] - Im[2] - Re[3];
	x_r_d[a[3]] = Re[0] - Im[1] - Re[2] + Im[3];
	
}



//-----------------------------------------------------------------------------
// This is the main function that performs the FFT on the input signal using the specified radix.
void gpuKernel(float* x_r_d, float* x_i_d, /*float* X_r_d, float* X_i_d,*/ const unsigned int N, const unsigned int M)
{
	// Perform bit-reversal permutation for the radix-2 or radix-4 algorithm
	if((M % 2 == 1))
	{	
		// Perform the radix-2 FFT
		radix2reorder<<< dim3(N / (1024 * 512), 512, 1), 1024 >>>(x_r_d, x_i_d, N, M);	
		for(int k = 1; k < N; k *= 2)
			radix2 <<< dim3(N / (1024 * 512), 32, 32), 256 >>>(x_r_d, x_i_d, N, k);		
			
	}else{
		// Perform the radix-4 FFT	
		radix4reorder<<< dim3(N / (512 * 512), 32, 32), 256 >>>(x_r_d, x_i_d, N, M);
		
		for (int k = 1; k < N; k *= 4)
			radix4 <<< dim3(N / (1024 * 1024), 32, 32), 256 >>>(x_r_d, x_i_d, N, k);			
	}
}
