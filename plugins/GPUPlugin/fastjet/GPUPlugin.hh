#ifndef cluster_hh
#define cluster_hh
#include "fastjet/ClusterSequence.hh"
// #include "fastjet/ClusterSequence.hh"
#include<iostream>


class GPUPlugin {
public:
  GPUPlugin(fastjet::ClusterSequence & cs);
  // void gpuClustring(fastjet::ClusterSequence & cs){
  //   std::cout << "GPU!!!\n";
  //   std::cout << cs.jets().size() << std::endl;
  // }

};

#endif  // cluster_hh
