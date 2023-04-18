cmake_minimum_required(VERSION 3.10)

project(libspatialize)

add_executable(main main.cpp)
target_link_libraries(main PRIVATE spatialize)
target_compile_features(main PRIVATE cxx_std_14)
target_compile_options(main PRIVATE -O3)

add_subdirectory(contrib)

add_library(spatialize INTERFACE)
target_include_directories(spatialize INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/spatialize/include>
)

target_link_libraries(spatialize
  INTERFACE eigen doctest m
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_PREFIX}/spatialize/include)

install(TARGETS spatialize
  EXPORT spatialize-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/spatialize/lib
)

add_custom_target(copy_test_data ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/test/testdata
          test/testdata
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Copying test files..."
  VERBATIM
)
