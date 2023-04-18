add_executable(tests tests.cpp)

target_compile_features(tests PRIVATE cxx_std_17)
target_compile_options(spatialize PRIVATE -Wall -Wextra -O3)
target_link_libraries(tests PRIVATE spatialize)

add_custom_target(copy_test_data ALL
  COMMAND ${CMAKE_COMMAND} -E copy_directory
          ${CMAKE_CURRENT_SOURCE_DIR}/testdata
          $<TARGET_FILE_DIR:tests>/testdata
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Copying test files..."
  VERBATIM
)
