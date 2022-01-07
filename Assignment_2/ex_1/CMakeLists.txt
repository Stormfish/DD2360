cmake_minimum_required(VERSION 3.18)

project(DD2360-A2E1 VERSION 1.0
                        DESCRIPTION "DD2350-lab3"
                        LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER g++)


#add_library(matrix SHARED 
#   c/matrix_mkl.c)

add_executable(printkernel
    excercise_1.cu)

#set output target
set_target_properties(printkernel
    PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY 	"lib/"
    LIBRARY_OUTPUT_DIRECTORY 	"lib/"
    RUNTIME_OUTPUT_DIRECTORY 	"bin/"
)

#target_link_directories(matrix PUBLIC ${MKL_LIBS})
#target_link_libraries(matrix -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -ldl)

#target_include_directories(matrix 
#	PRIVATE ${MKL_INCLUDES}
#	PUBLIC c)

# set to use C11 features
target_compile_features(matrix PRIVATE c_std_11)

# set compiler options
target_compile_options(matrix PRIVATE -Wall -O2 )



