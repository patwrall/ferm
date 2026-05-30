add_executable(lob_bench lob_bench.cpp)
target_link_libraries(lob_bench PRIVATE ferm::ferm)

add_executable(gpu_bench gpu_bench.cu)
target_link_libraries(gpu_bench PRIVATE ferm::ferm_cuda)
set_target_properties(gpu_bench PROPERTIES CUDA_ARCHITECTURES "86;89")
