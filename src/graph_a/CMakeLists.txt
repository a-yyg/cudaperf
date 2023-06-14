cuda_add_executable(graph_a
	graph_a.cu
	${COMMON_FILES})

target_link_libraries(graph_a
	m
	${CUDA_LIBRARIES})

target_include_directories(graph_a
	PUBLIC
	${CUDA_INCLUDE_DIRS}
	${CMAKE_CURRENT_SOURCE_DIR}/..
	)
