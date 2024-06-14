
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# Check if the current CMake version is greater than a specified version
if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.0)
    message(STATUS "CMake version is within the allowed range.")
else()
    message(FATAL_ERROR "CMake version is not supported. Minimum version required: 2.6")
endif()

# Specify the minimum required version
if(CMAKE_VERSION VERSION_LESS 3.10)
    message(FATAL_ERROR "Minimum required version of CMake is 3.10")
endif()

# Specify the maximum allowed version
if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
    message(FATAL_ERROR "Maximum allowed version of CMake is 4.0")
endif()

project(sphorb)

find_package(OpenCV 2.4 REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem)

include_directories(${OpenCV_INCLUDE_DIRS}
                    include)

add_executable (example1 example1.cpp
                         pfm.cpp
                         utility.cpp
                         read_file.cpp
                         detector.cpp
                         nonmax.cpp
                         SPHORB.cpp)

target_link_libraries (example1 ${OpenCV_LIBRARIES})

add_executable (example2 example2.cpp
                         pfm.cpp
                         utility.cpp
                         detector.cpp
                         nonmax.cpp
                         SPHORB.cpp)

target_link_libraries (example2 ${OpenCV_LIBRARIES})

add_executable (read_from_file read_from_file.cpp
                         pfm.cpp
                         utility.cpp
                         read_file.cpp
                         detector.cpp
                         nonmax.cpp
                         SPHORB.cpp)

target_link_libraries (read_from_file ${OpenCV_LIBRARIES})

add_executable (process_directory process_directory.cpp
                         pfm.cpp
                         utility.cpp
                         detector.cpp
                         nonmax.cpp
                         SPHORB.cpp)

target_link_libraries (process_directory ${OpenCV_LIBRARIES} Boost::filesystem)