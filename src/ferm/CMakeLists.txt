include(GenerateExportHeader)

add_library(ferm OrderBook.cpp)

add_library(ferm::ferm ALIAS ferm)

ferm_target_link_cuda(ferm)

target_link_libraries(ferm PRIVATE ferm_options ferm_warnings)

target_include_directories(ferm ${WARNING_GUARD} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                                                        $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>)

target_compile_features(ferm PUBLIC cxx_std_20)

set_target_properties(
  ferm
  PROPERTIES VERSION ${PROJECT_VERSION}
             CXX_VISIBILITY_PRESET hidden
             VISIBILITY_INLINES_HIDDEN YES)

generate_export_header(ferm EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/include/ferm/ferm_export.hpp)

if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(ferm PUBLIC ferm_STATIC_DEFINE)
endif()
