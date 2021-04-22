cmake_minimum_required(VERSION 3.10)  # my cmake version
project(libelnet)  # name of the project

# use c++ 11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add google subdirectory to compile with
add_subdirectory(lib/googletest)

# folder with header files
include_directories(src)

# header files for Eigen as Eigen does not need to compile
# because of being a header only library
include_directories(lib/eigen)

# find source files via wildcard

# add library out of source files to be found by test files
set(MAIN_LIBRARY ${CMAKE_PROJECT_NAME}_lib)
file(GLOB SOURCES "src/*.cpp")  # this does glob search and saves file names into SOURCES
add_library(${MAIN_LIBRARY} STATIC ${SOURCES})

# enable testing for GoogleTest
enable_testing()

# add executable for the tests
file(GLOB TEST_SOURCES "tests_cpp/*.cpp")
add_executable(test_libelnet ${TEST_SOURCES})

# link main source code and googletest to test executable
# for the test files to find the main code and GoogleTest
target_link_libraries(test_libelnet ${MAIN_LIBRARY} gtest_main)

# include GoogleTest library
include(GoogleTest)
# let GoogleTest find out the tests
gtest_discover_tests(test_libelnet)
