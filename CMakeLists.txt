cmake_minimum_required (VERSION 3.10)
project (cudaperf)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDA REQUIRED)

set(COMMON_FILES
	${PROJECT_SOURCE_DIR}/src/common/timelogger.cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
