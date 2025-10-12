{ mkShell
, pkgs
, ... }:

mkShell {
  packages = with pkgs; [
    gcc
    clang
    llvmPackages_19.bintools
    gdb
    cmake
    ninja

    cudaPackages.cuda_cudart.static
    cudaPackages.cudatoolkit
    cudaPackages.cuda_nvcc
    cudaPackages.libcurand

    doxygen
    ccache
    cppcheck
    include-what-you-use
  ];

  shellHook = ''
    echo "Entered FERM Devshell"

    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
    export PATH=${pkgs.cudaPackages.cuda_nvcc}/bin:$PATH

    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:${pkgs.cudaPackages.cuda_cudart.static}/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:${pkgs.cudaPackages.cuda_cudart.static}/lib:$LIBRARY_PATH

    export CMAKE_CUDA_ARCHITECTURES="75;86;89"
  '';
}
