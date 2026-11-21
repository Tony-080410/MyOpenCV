cmake_minimum_required(VERSION 3.16)
project(MyOpenCV VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 不指定 -DCMAKE_BUILD_TYPE 时按 Release 编，否则卷积慢到没法用
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)   # clangd 用

add_library(mycv STATIC
  src/core/mat.cpp
  src/core/memory.cpp
  src/imgproc/color.cpp
  src/imgproc/geom.cpp
  src/imgproc/thresh.cpp
  src/imgproc/filter.cpp
  src/imgproc/morph.cpp
  src/imgproc/edge.cpp
  src/imgproc/contours.cpp
  src/imgproc/draw.cpp
  src/io/image.cpp
  src/io/video.cpp
)
target_include_directories(mycv PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(mycv PRIVATE -Wall -Wextra)

# JPEG：系统有 libjpeg 就开，没有就只支持 PGM/PPM。
find_package(JPEG QUIET)
if(JPEG_FOUND)
  target_link_libraries(mycv PUBLIC JPEG::JPEG)
  target_compile_definitions(mycv PRIVATE MYCV_HAVE_JPEG=1)
else()
  message(STATUS "JPEG: 没找到 libjpeg")
endif()

include(CTest)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
