cmake_minimum_required(VERSION 3.22.1)
project(BrailleWordGeometry)
set(CMAKE_CXX_STANDARD 20)

add_library(BrailleWordGeometry SHARED src/main/cpp/cpp-adapter.cpp)
include(${CMAKE_CURRENT_SOURCE_DIR}/../nitrogen/generated/android/BrailleWordGeometry+autolinking.cmake)
find_package(react-native-vision-camera REQUIRED)
find_package(react-native-nitro-image REQUIRED)
target_link_libraries(BrailleWordGeometry react-native-vision-camera::VisionCamera react-native-nitro-image::NitroImage android)
