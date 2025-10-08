{
  mkShell,
  pkgs,
  ...
}:
mkShell {
  packages = with pkgs; [
    gcc
    clang
    llvmPackages_19.bintools
    gdb
    cmake
    ninja
    cudatoolkit
    doxygen
    ccache
    cppcheck
    include-what-you-use
  ];

  shellHook = ''
    echo "Entered FERM Devshell"
  '';
}
