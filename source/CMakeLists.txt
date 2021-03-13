cmake_minimum_required(VERSION 3.10)

# set the project name
project(MA)

#set directories
set(SOURCE_DIR "X://MA//source")
set(INCLUDE_DIR ${SOURCE_DIR}/include)

set(CMAKE_PREFIX_PATH "C://Devel//armadillo-9.900.1")

#include headers
set(INCLUDE 
	${INCLUDE_DIR}/read.h 
	${INCLUDE_DIR}/write.h 
	${INCLUDE_DIR}/sample.h 
	${INCLUDE_DIR}/matrixgen.h 
	${INCLUDE_DIR}/mathfns.h
	${INCLUDE_DIR}/util.h
	${INCLUDE_DIR}/mac0.h
)

include_directories(${INCLUDE_DIR})

#set sources for executable
set(SOURCES 
	${SOURCE_DIR}/main.cc
	${SOURCE_DIR}/read.cc
	${SOURCE_DIR}/write.cc
	${SOURCE_DIR}/sample.cc
	${SOURCE_DIR}/matrixgen.cc
	${SOURCE_DIR}/mathfns.cc
	${SOURCE_DIR}/util.cc
	# Include
	${INCLUDE}
)

set(ARMADILLO_LIBRARY "C://Devel//armadillo-9.900.1//build//Debug//armadillo.lib")
#set(ARMADILLO_INCLUDE_DIRS "C://Devel//armadillo-9.900.1//include")

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_executable(MA ${SOURCES})

target_link_libraries(MA ${ARMADILLO_LIBRARIES})
