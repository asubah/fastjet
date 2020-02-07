// fastjet stuff
#include "fastjet/ClusterSequence.hh"
#include "fastjet/GPUPlugin.hh"
#include "cuda/cudaCheck.h"

// other stuff
#include <vector>
#include <cuda_runtime.h>


using namespace std;

void initialise() {
  cudaSetDevice(0);
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "Running on CUDA device " << prop.name << std::endl;
  int value;
  cudaDeviceGetAttribute(&value, cudaDevAttrMaxSharedMemoryPerBlock, 0);
  std::cout << "  - maximum shared memory per block: " << value / 1024 << " kB" << std::endl;

  cudaCheck(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 10 * 1024 * 1024));
  size_t size;
  cudaCheck(cudaDeviceGetLimit(&size, cudaLimitPrintfFifoSize));
  std::cout << "  - kernel printf buffer size:       " << size / 1024 << " kB" << std::endl;
}

GPUPlugin::GPUPlugin(fastjet::ClusterSequence & cs){
  std::cout << "GPU!!!\n";
  std::cout << cs.jets().size() << std::endl;
  
  // initialise the GPU
  initialise();
}

//void GPUPlugin::run_clustering(ClusterSequence & cs) const {
//}
