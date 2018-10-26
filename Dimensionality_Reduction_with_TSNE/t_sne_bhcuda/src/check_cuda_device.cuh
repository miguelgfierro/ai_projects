#ifndef CHECK_CUDA_DEVICE_CUH_
#define CHECK_CUDA_DEVICE_CUH_

void initialize_cuda_device(int* ncuda_devices, int verbose);
void exit_on_cuda_error(char* kernel_name);
void exit_on_error(char* info);
void get_device_memory(double* free_db, double* used_db, double* total_db);
void print_device_memory();

#endif