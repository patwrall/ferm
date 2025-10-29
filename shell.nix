{
  mkShell,
  pkgs,
  ...
}:

mkShell {
  packages = with pkgs; [
    bintools
    clang
    llvmPackages.llvm

    gdb
    cmake
    ninja

    cudaPackages.cudatoolkit

    doxygen
    ccache
    cppcheck
    include-what-you-use
  ];

  buildInputs = with pkgs; [
    llvmPackages.libcxx

    cudaPackages.cuda_cudart.static
    cudaPackages.libcurand
  ];

  shellHook = ''
    echo "Entered FERM Devshell"

    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
    export PATH=${pkgs.cudaPackages.cuda_nvcc}/bin:${pkgs.llvmPackages.llvm}/bin:$PATH

    export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:${pkgs.cudaPackages.cuda_cudart.static}/lib:$LD_LIBRARY_PATH
    export LIBRARY_PATH=$CUDA_PATH/lib64:$CUDA_PATH/lib:${pkgs.cudaPackages.cuda_cudart.static}/lib:$LIBRARY_PATH

    export CMAKE_CUDA_ARCHITECTURES="75;86;89"

    export CMAKE_AR="${pkgs.llvmPackages.llvm}/bin/llvm-ar"
    export CMAKE_RANLIB="${pkgs.llvmPackages.llvm}/bin/llvm-ranlib"
  '';
}
