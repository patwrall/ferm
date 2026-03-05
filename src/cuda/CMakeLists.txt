find_package(CUDAToolkit REQUIRED)

add_library(ferm_cuda gpu_match.cu gpu_hawkes.cu)
target_include_directories(ferm_cuda PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ferm_cuda PRIVATE ferm::ferm_options CUDA::cudart CUDA::curand)
set_target_properties(ferm_cuda PROPERTIES CUDA_ARCHITECTURES "86;89;90")

add_library(ferm::ferm_cuda ALIAS ferm_cuda)
