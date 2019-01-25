cmake_minimum_required(VERSION 3.8)
project(LinearModel)

set(CMAKE_CXX_STANDARD 14)

add_executable(LinearModel main.cpp function.cpp l2r_lr_fun.cpp)