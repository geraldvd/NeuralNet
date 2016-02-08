project(NN)
cmake_minimum_required(VERSION 2.8)
add_definitions(-std=c++11)

set(CMAKE_BUILD_TYPE Debug)
add_executable(NN main.cpp net.cpp neuron.cpp connection.cpp)

