include(cmake/SystemLink.cmake)
include(cmake/LibFuzzer.cmake)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)


include(CheckCXXSourceCompiles)


macro(ferm_supports_sanitizers)
  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND NOT WIN32)

    message(STATUS "Sanity checking UndefinedBehaviorSanitizer, it should be supported on this platform")
    set(TEST_PROGRAM "int main() { return 0; }")

    # Check if UndefinedBehaviorSanitizer works at link time
    set(CMAKE_REQUIRED_FLAGS "-fsanitize=undefined")
    set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=undefined")
    check_cxx_source_compiles("${TEST_PROGRAM}" HAS_UBSAN_LINK_SUPPORT)

    if(HAS_UBSAN_LINK_SUPPORT)
      message(STATUS "UndefinedBehaviorSanitizer is supported at both compile and link time.")
      set(SUPPORTS_UBSAN ON)
    else()
      message(WARNING "UndefinedBehaviorSanitizer is NOT supported at link time.")
      set(SUPPORTS_UBSAN OFF)
    endif()
  else()
    set(SUPPORTS_UBSAN OFF)
  endif()

  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND WIN32)
    set(SUPPORTS_ASAN OFF)
  else()
    if (NOT WIN32)
      message(STATUS "Sanity checking AddressSanitizer, it should be supported on this platform")
      set(TEST_PROGRAM "int main() { return 0; }")

      # Check if AddressSanitizer works at link time
      set(CMAKE_REQUIRED_FLAGS "-fsanitize=address")
      set(CMAKE_REQUIRED_LINK_OPTIONS "-fsanitize=address")
      check_cxx_source_compiles("${TEST_PROGRAM}" HAS_ASAN_LINK_SUPPORT)

      if(HAS_ASAN_LINK_SUPPORT)
        message(STATUS "AddressSanitizer is supported at both compile and link time.")
        set(SUPPORTS_ASAN ON)
      else()
        message(WARNING "AddressSanitizer is NOT supported at link time.")
        set(SUPPORTS_ASAN OFF)
      endif()
    else()
      set(SUPPORTS_ASAN ON)
    endif()
  endif()
endmacro()

macro(ferm_setup_options)
  option(ferm_ENABLE_HARDENING "Enable hardening" ON)
  option(ferm_ENABLE_COVERAGE "Enable coverage reporting" OFF)
  cmake_dependent_option(
    ferm_ENABLE_GLOBAL_HARDENING
    "Attempt to push hardening options to built dependencies"
    ON
    ferm_ENABLE_HARDENING
    OFF)

  ferm_supports_sanitizers()

  if(NOT PROJECT_IS_TOP_LEVEL OR ferm_PACKAGING_MAINTAINER_MODE)
    option(ferm_ENABLE_IPO "Enable IPO/LTO" OFF)
    option(ferm_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
    option(ferm_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
    option(ferm_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
    option(ferm_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(ferm_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
    option(ferm_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(ferm_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(ferm_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(ferm_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(ferm_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
    option(ferm_ENABLE_PCH "Enable precompiled headers" OFF)
    option(ferm_ENABLE_CACHE "Enable ccache" OFF)
  else()
    option(ferm_ENABLE_IPO "Enable IPO/LTO" ON)
    option(ferm_WARNINGS_AS_ERRORS "Treat Warnings As Errors" ON)
    option(ferm_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
    option(ferm_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" ${SUPPORTS_ASAN})
    option(ferm_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(ferm_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" ${SUPPORTS_UBSAN})
    option(ferm_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(ferm_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(ferm_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(ferm_ENABLE_CLANG_TIDY "Enable clang-tidy" ON)
    option(ferm_ENABLE_CPPCHECK "Enable cpp-check analysis" ON)
    option(ferm_ENABLE_PCH "Enable precompiled headers" OFF)
    option(ferm_ENABLE_CACHE "Enable ccache" ON)
  endif()

  if(NOT PROJECT_IS_TOP_LEVEL)
    mark_as_advanced(
      ferm_ENABLE_IPO
      ferm_WARNINGS_AS_ERRORS
      ferm_ENABLE_USER_LINKER
      ferm_ENABLE_SANITIZER_ADDRESS
      ferm_ENABLE_SANITIZER_LEAK
      ferm_ENABLE_SANITIZER_UNDEFINED
      ferm_ENABLE_SANITIZER_THREAD
      ferm_ENABLE_SANITIZER_MEMORY
      ferm_ENABLE_UNITY_BUILD
      ferm_ENABLE_CLANG_TIDY
      ferm_ENABLE_CPPCHECK
      ferm_ENABLE_COVERAGE
      ferm_ENABLE_PCH
      ferm_ENABLE_CACHE)
  endif()

  ferm_check_libfuzzer_support(LIBFUZZER_SUPPORTED)
  if(LIBFUZZER_SUPPORTED AND (ferm_ENABLE_SANITIZER_ADDRESS OR ferm_ENABLE_SANITIZER_THREAD OR ferm_ENABLE_SANITIZER_UNDEFINED))
    set(DEFAULT_FUZZER ON)
  else()
    set(DEFAULT_FUZZER OFF)
  endif()

  option(ferm_BUILD_FUZZ_TESTS "Enable fuzz testing executable" ${DEFAULT_FUZZER})

endmacro()

macro(ferm_global_options)
  if(ferm_ENABLE_IPO)
    include(cmake/InterproceduralOptimization.cmake)
    ferm_enable_ipo()
  endif()

  ferm_supports_sanitizers()

  if(ferm_ENABLE_HARDENING AND ferm_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN 
       OR ferm_ENABLE_SANITIZER_UNDEFINED
       OR ferm_ENABLE_SANITIZER_ADDRESS
       OR ferm_ENABLE_SANITIZER_THREAD
       OR ferm_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    message("${ferm_ENABLE_HARDENING} ${ENABLE_UBSAN_MINIMAL_RUNTIME} ${ferm_ENABLE_SANITIZER_UNDEFINED}")
    ferm_enable_hardening(ferm_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()
endmacro()

macro(ferm_local_options)
  if(PROJECT_IS_TOP_LEVEL)
    include(cmake/StandardProjectSettings.cmake)
  endif()

  add_library(ferm_warnings INTERFACE)
  add_library(ferm_options INTERFACE)

  include(cmake/CompilerWarnings.cmake)
  ferm_set_project_warnings(
    ferm_warnings
    ${ferm_WARNINGS_AS_ERRORS}
    ""
    ""
    ""
    "")

  if(ferm_ENABLE_USER_LINKER)
    include(cmake/Linker.cmake)
    ferm_configure_linker(ferm_options)
  endif()

  include(cmake/Sanitizers.cmake)
  ferm_enable_sanitizers(
    ferm_options
    ${ferm_ENABLE_SANITIZER_ADDRESS}
    ${ferm_ENABLE_SANITIZER_LEAK}
    ${ferm_ENABLE_SANITIZER_UNDEFINED}
    ${ferm_ENABLE_SANITIZER_THREAD}
    ${ferm_ENABLE_SANITIZER_MEMORY})

  set_target_properties(ferm_options PROPERTIES UNITY_BUILD ${ferm_ENABLE_UNITY_BUILD})

  if(ferm_ENABLE_PCH)
    target_precompile_headers(
      ferm_options
      INTERFACE
      <vector>
      <string>
      <utility>)
  endif()

  if(ferm_ENABLE_CACHE)
    include(cmake/Cache.cmake)
    ferm_enable_cache()
  endif()

  include(cmake/StaticAnalyzers.cmake)
  if(ferm_ENABLE_CLANG_TIDY)
    ferm_enable_clang_tidy(ferm_options ${ferm_WARNINGS_AS_ERRORS})
  endif()

  if(ferm_ENABLE_CPPCHECK)
    ferm_enable_cppcheck(${ferm_WARNINGS_AS_ERRORS} "" # override cppcheck options
    )
  endif()

  if(ferm_ENABLE_COVERAGE)
    include(cmake/Tests.cmake)
    ferm_enable_coverage(ferm_options)
  endif()

  if(ferm_WARNINGS_AS_ERRORS)
    check_cxx_compiler_flag("-Wl,--fatal-warnings" LINKER_FATAL_WARNINGS)
    if(LINKER_FATAL_WARNINGS)
      # This is not working consistently, so disabling for now
      # target_link_options(ferm_options INTERFACE -Wl,--fatal-warnings)
    endif()
  endif()

  if(ferm_ENABLE_HARDENING AND NOT ferm_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN 
       OR ferm_ENABLE_SANITIZER_UNDEFINED
       OR ferm_ENABLE_SANITIZER_ADDRESS
       OR ferm_ENABLE_SANITIZER_THREAD
       OR ferm_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    ferm_enable_hardening(ferm_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()

endmacro()
