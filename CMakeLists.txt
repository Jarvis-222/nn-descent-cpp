cmake_minimum_required(VERSION 3.14)
project(nn_descent_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O2")

include_directories(include)

add_executable(nn_descent
    src/main.cpp
    src/initializer.cpp
    src/nn_descent.cpp
    src/recall.cpp
)
