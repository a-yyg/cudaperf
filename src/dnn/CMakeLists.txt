list(APPEND PLUGINS "nvinfer")
list(APPEND PLUGINS "nvonnxparser")
# list(APPEND PLUGINS "nvparsers")

foreach(libName ${PLUGINS})
    find_library(${libName}_lib NAMES ${libName} "/usr" PATH_SUFFIXES lib)
    list(APPEND PLUGIN_LIBS "${${libName}_lib}")
endforeach()

if (NOT PLUGIN_LIBS)
	message(WARNING "No TensorRT plugins found. Not building DNN samples.")
	return()
endif()

find_package(OpenCV REQUIRED)

add_executable(make_engine
	make_engine.cpp
	/usr/src/tensorrt/samples/common/logger.cpp
	${COMMON_FILES})

target_link_libraries(make_engine
	${CUDA_LIBRARIES}
	${CUDNN_LIBRARIES}
	${PLUGIN_LIBS}
	${OpenCV_LIBS}
	)

target_include_directories(make_engine
	PUBLIC
	${CUDA_INCLUDE_DIRS}
	${CUDNN_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}/..
	/usr/src/tensorrt/samples/common
	${OpenCV_INCLUDE_DIRS}
	)

add_executable(run_engine
	run_engine.cpp
	${COMMON_FILES})

target_link_libraries(run_engine
	${CUDA_LIBRARIES}
	${CUDNN_LIBRARIES}
	${OpenCV_LIBS}
	${PLUGIN_LIBS})

target_include_directories(run_engine
	PUBLIC
	${CUDA_INCLUDE_DIRS}
	${CUDNN_INCLUDE_DIRS}
	${OpenCV_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}/..
	)
