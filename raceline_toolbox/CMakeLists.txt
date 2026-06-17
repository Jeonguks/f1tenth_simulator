cmake_minimum_required(VERSION 3.10)
project(raceline_core)

find_package(pybind11 REQUIRED)

pybind11_add_module(raceline_core raceline_core.cpp)