cuda_add_executable(graph_b
	graph_b.cu
	${COMMON_FILES})

target_link_libraries(graph_b
	m
	${CUDA_LIBRARIES})

target_include_directories(graph_b
	PUBLIC
	${CUDA_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}/..
	)
