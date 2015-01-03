cmake_minimum_required(VERSION 2.8)
project(object_recognition_transparent_objects)

find_package(PCL REQUIRED)
find_package(catkin QUIET)

if (catkin_FOUND)
    find_package(catkin REQUIRED ecto object_recognition_core opencv_candidate)
    catkin_package(INCLUDE_DIRS include)

    #install targets for all things python
    catkin_python_setup()

    include_directories(${catkin_INCLUDE_DIRS})
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

find_package(Boost REQUIRED program_options signals)
find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

add_subdirectory(src)

if (${catkin_FOUND})
    add_subdirectory(cells)
endif()

# deal with the tests
option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()  

# add docs
ecto_sphinx(${CMAKE_CURRENT_SOURCE_DIR}/doc/source doc)
