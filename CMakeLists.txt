cmake_minimum_required(VERSION 3.10)
project(libelnet)

# use c++ 11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add subdirectories
add_subdirectory(lib/googletest)

# header files together with cpp files
include_directories(src)

# header files for Eigen
include_directories(lib/eigen)

# find source files via wildcard
file(GLOB SOURCES "src/*.cpp")

# add library out of source files to be found by test files
set(MAIN_LIBRARY ${CMAKE_PROJECT_NAME}_lib)
add_library(${MAIN_LIBRARY} STATIC ${SOURCES})

# testing via GoogleTest
enable_testing()

# executable for the tests
file(GLOB TEST_SOURCES "tests_cpp/*.cpp")
add_executable(test_libelnet ${TEST_SOURCES})

target_link_libraries(test_libelnet ${MAIN_LIBRARY} gtest_main)

include(GoogleTest)
gtest_discover_tests(test_libelnet)
