cmake_minimum_required(VERSION 3.10)
project(libelnet)

# use c++ 11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# add subdirectories
add_subdirectory(lib/googletest)
add_subdirectory(lib/eigen)
add_subdirectory(src)
add_subdirectory(tests_cpp)

# header files together with cpp files
include_directories(src)

# add source files via wildcard
file(GLOB SOURCES "src/*.cpp")

# testing via GoogleTest
enable_testing()

# executable for the tests
add_executable(test_libelnet tests_cpp/main.cpp)

target_link_libraries(test_libelnet gtest_main)

include(GoogleTest)
gtest_discover_tests(test_libelnet)
