cmake_minimum_required(VERSION 2.8)
project(object_recognition_transparent_objects)

find_package(PCL REQUIRED)
find_package(catkin QUIET)
if (catkin_FOUND)
    if (${catkin_VERSION} VERSION_GREATER "0.5.28")
    find_package(catkin REQUIRED ecto ecto_opencv object_recognition_core opencv_candidate)
        catkin_package(INCLUDE_DIRS include)
    else()
        find_package(catkin REQUIRED opencv_candidate)
        find_package(ecto REQUIRED)
        find_package(ecto_opencv REQUIRED)
        find_package(object_recognition_core REQUIRED)
        catkin_stack()
        catkin_project(object_recognition_transparent_objects
                   INCLUDE_DIRS include
        )
        list(APPEND catkin_INCLUDE_DIRS ${ecto_INCLUDE_DIRS} ${object_recognition_core_INCLUDE_DIRS}
                                        ${opencv_candidate_INCLUDE_DIRS}
        )
        list(APPEND catkin_LIBRARIES ${ecto_LIBRARIES} ${object_recognition_core_LIBRARIES} ${opencv_candidate_LIBRARIES})
    endif()

    #install targets for all things python
    catkin_python_setup()

    find_package(ecto REQUIRED)
    find_package(ecto_opencv REQUIRED)
    find_package(object_recognition_core REQUIRED)
else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endif()

find_package(OpenCV REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${object_recognition_core_INCLUDE_DIRS})

add_subdirectory(src)

if (${ecto_FOUND} AND ${object_recognition_core_FOUND})
    add_subdirectory(cells)
endif()

# deal with the tests
option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()  
