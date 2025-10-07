{
  mkShell,
  pkgs,
  ...
}:
mkShell {
  packages = with pkgs; [
    gcc
    clang
    gdb
    cmake
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
