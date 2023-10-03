# cmake_minimum_required(VERSION 2.6)

# project(yolox)

# add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE Debug)

find_package(CUDA REQUIRED)

find_package(OpenMP REQUIRED)

if (OPENMP_FOUND)
	set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-11.4/targets/aarch64-linux/include)
link_directories(/usr/local/cuda-11.4/targets/aarch64-linux/lib)
# # cudnn
# include_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/include)
# link_directories(/data/cuda/cuda-10.2/cudnn/v8.0.4/lib64)
# # tensorrt
# include_directories(/data/cuda/cuda-10.2/TensorRT/v7.2.1.6/include)
# link_directories(/data/cuda/cuda-10.2/TensorRT/v7.2.1.6/lib)

list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "cudart")
list(APPEND PLUGINS "nvparsers")

foreach(libName ${PLUGINS})
	find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
	list(APPEND PLUGIN_LIBS "${${libName}_lib}")
endforeach()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

set(NMS_OPTIONS
	"0.45"
	"0.65")

set(CONF_THRESH
	"0.3"
	"0.001")

foreach(NMS_VAL ${NMS_OPTIONS})
	foreach(CONF_VAL ${CONF_THRESH})
		# Get digits after decimal point
		string(REGEX MATCH "[0-9]+$" NMS_VAL_DIGITS ${NMS_VAL})
		string(REGEX MATCH "[0-9]+$" CONF_VAL_DIGITS ${CONF_VAL})
		# Set output name
		set(OUTPUT_NAME yolox_${NMS_VAL_DIGITS}_${CONF_VAL_DIGITS})
		add_executable(${OUTPUT_NAME}
			yolox.cpp
			${COMMON_FILES}
		)
		target_compile_definitions(${OUTPUT_NAME} PUBLIC BBOX_CONF_THRESH=${CONF_VAL})
		target_compile_definitions(${OUTPUT_NAME} PUBLIC NMS_THRESH=${NMS_VAL})
		target_link_libraries(${OUTPUT_NAME}
			${CUDA_LIBRARIES}
			${CUDNN_LIBRARIES}
			${PLUGIN_LIBRARIES}
			${OpenCV_LIBRARIES}
			nvinfer
			cudart
			)
		target_include_directories(${OUTPUT_NAME}
			PUBLIC
			${CUDA_INCLUDE_DIRS}
			${CUDNN_INCLUDE_DIRS}
			${CMAKE_CURRENT_SOURCE_DIR}/..
			)
		add_definitions(-O2 -pthread)
	endforeach()
endforeach()
