#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include "cuda_runtime.h"


/* ----------------------------------------------------------------------------------------------- */

void get_device_memory(double* free_db, double* used_db, double* total_db) {

	// gets memory usage in byte
	size_t free_byte;
	size_t total_byte;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
	if (cudaSuccess != cuda_status){
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(EXIT_FAILURE);
	}

	*free_db = (double)free_byte;
	*total_db = (double)total_byte;
	*used_db = *total_db - *free_db;

	return;
}


void print_device_memory(){
	double free_db, used_db, total_db;
	get_device_memory(&free_db, &used_db, &total_db);

	printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",
		used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}
/* ----------------------------------------------------------------------------------------------- */

void exit_on_error(char* info) {
	printf("\nERROR: %s\n", info);
	fflush(stdout);

	//free(info);
	exit(EXIT_FAILURE);
	return;
}

/* ----------------------------------------------------------------------------------------------- */

void exit_on_cuda_error(char* kernel_name) {
	// sync and check to catch errors from previous async operations
	cudaThreadSynchronize();
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		printf("Error after %s: %s\n", kernel_name, cudaGetErrorString(err));

		// releases previous contexts
#if CUDA_VERSION < 4000
		cudaThreadExit();
#else
		cudaDeviceReset();
#endif

		// stops program
		//free(kernel_name);
		exit(EXIT_FAILURE);
	}
}



/* ----------------------------------------------------------------------------------------------- */

// GPU initialization

/* ----------------------------------------------------------------------------------------------- */


void initialize_cuda_device(int* ncuda_devices, int verbose) {

	int device;
	int device_count;
	int myrank = 0;

	/*
	// cuda initialization (needs -lcuda library)
	// note:   cuInit initializes the driver API.
	//             it is needed for any following CUDA driver API function call (format cuFUNCTION(..) )
	//             however, for the CUDA runtime API functions (format cudaFUNCTION(..) )
	//             the initialization is implicit, thus cuInit() here would not be needed...
	CUresult status = cuInit(0);
	if ( CUDA_SUCCESS != status ) exit_on_error("CUDA driver API device initialization failed\n");

	// returns a handle to the first cuda compute device
	CUdevice dev;
	status = cuDeviceGet(&dev, 0);
	if ( CUDA_SUCCESS != status ) exit_on_error("CUDA device not found\n");

	// gets device properties
	int major,minor;
	status = cuDeviceComputeCapability(&major,&minor,dev);
	if ( CUDA_SUCCESS != status ) exit_on_error("CUDA device information not found\n");

	// make sure that the device has compute capability >= 1.3
	if (major < 1){
	fprintf(stderr,"Compute capability major number should be at least 1, got: %d \nexiting...\n",major);
	exit_on_error("CUDA Compute capability major number should be at least 1\n");
	}
	if (major == 1 && minor < 3){
	fprintf(stderr,"Compute capability should be at least 1.3, got: %d.%d \nexiting...\n",major,minor);
	exit_on_error("CUDA Compute capability major number should be at least 1.3\n");
	}
	*/

	// note: from here on we use the runtime API  ...

	// Gets number of GPU devices
	device_count = 0;
	cudaGetDeviceCount(&device_count);
	exit_on_cuda_error("CUDA runtime error: cudaGetDeviceCount failed\ncheck if driver and runtime libraries work together\nexiting...\n");

	// returns device count to fortran
	if (device_count == 0) exit_on_error("CUDA runtime error: there is no device supporting CUDA\n");
	*ncuda_devices = device_count;

	// Sets the active device
	if (device_count >= 1) {
		// generalized for more GPUs per node
		// note: without previous context release, cudaSetDevice will complain with the cuda error
		//         "setting the device when a process is active is not allowed"

		// releases previous contexts
		#if CUDA_VERSION < 4000
				cudaThreadExit();
		#else
				cudaDeviceReset();
		#endif

		// sets active device
		device = myrank % device_count;
		cudaSetDevice(device);
		exit_on_cuda_error("cudaSetDevice has invalid device");

		// double check that device was  properly selected
		cudaGetDevice(&device);
		if (device != (myrank % device_count)){
			printf("error rank: %d devices: %d \n", myrank, device_count);
			printf("  cudaSetDevice()=%d\n  cudaGetDevice()=%d\n", myrank%device_count, device);
			exit_on_error("CUDA set/get device error: device id conflict \n");
		}
	}

	// returns a handle to the active device
	cudaGetDevice(&device);

	// get device properties
	struct cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, device);

	// exit if the machine has no CUDA-enabled device
	if (deviceProp.major == 9999 && deviceProp.minor == 9999){
		printf("No CUDA-enabled device found, exiting...\n\n");
		exit_on_error("CUDA runtime error: there is no CUDA-enabled device found\n");
	}

	// outputs device infos to file
	if (verbose>1){
		//printf("GPU device for rank: %d\n\n", myrank);
		printf("\n");
		// display device properties
		printf("Device Name = %s\n", deviceProp.name);
		printf("multiProcessorCount: %d\n", deviceProp.multiProcessorCount);
		printf("totalGlobalMem (in MB): %f\n", (unsigned long)deviceProp.totalGlobalMem / (1024.f * 1024.f));
		printf("totalGlobalMem (in GB): %f\n", (unsigned long)deviceProp.totalGlobalMem / (1024.f * 1024.f * 1024.f));
		printf("sharedMemPerBlock (in bytes): %lu\n", (unsigned long)deviceProp.sharedMemPerBlock);
		printf("Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
		printf("Maximum size of each dimension of a block: %d x %d x %d\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		printf("Maximum sizes of each dimension of a grid: %d x %d x %d\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf("Compute capability of the device = %d.%d\n", deviceProp.major, deviceProp.minor);
		if (deviceProp.canMapHostMemory){
			printf("canMapHostMemory: TRUE\n");
		}
		else{
			printf("canMapHostMemory: FALSE\n");
		}
		if (deviceProp.deviceOverlap){
			printf("deviceOverlap: TRUE\n");
		}
		else{
			printf("deviceOverlap: FALSE\n");
		}

		// outputs initial memory infos via cudaMemGetInfo()
		print_device_memory();
		printf("\n");
	}

	// make sure that the device has compute capability >= 1.3
	if (deviceProp.major < 1){
		printf("Compute capability major number should be at least 1, exiting...\n\n");
		exit_on_error("CUDA Compute capability major number should be at least 1\n");
	}
	if (deviceProp.major == 1 && deviceProp.minor < 3){
		printf("Compute capability should be at least 1.3, exiting...\n");
		exit_on_error("CUDA Compute capability major number should be at least 1.3\n");
	}
	// we use pinned memory for asynchronous copy
	if (!deviceProp.canMapHostMemory){
		printf("Device capability should allow to map host memory, exiting...\n");
		exit_on_error("CUDA Device capability canMapHostMemory should be TRUE\n");
	}
}