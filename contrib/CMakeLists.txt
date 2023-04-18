add_library(eigen INTERFACE)
# Mark library folder as system to suppress a ton of gcc warnings
target_include_directories(eigen SYSTEM INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/contrib/eigen/include>
)

add_library(doctest INTERFACE)
target_include_directories(doctest SYSTEM INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/contrib/doctest>
)
