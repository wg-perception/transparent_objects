project(object_recognition_transparent_objects)
find_package(catkin QUIET)
if (catkin_FOUND)
    catkin_stack()
    catkin_project(object_recognition_transparent_objects
                   DEPENDS Eigen Boost OpenCV
                   INCLUDE_DIRS include
    )
    #install targets for all things python
    catkin_python_setup()

    find_package(ecto REQUIRED)
    find_package(ecto_opencv REQUIRED)
    find_pcl_package()
    find_package(object_recognition_core REQUIRED)
else()
    find_package(PCL)
endif()

#this needs a stanza to prevent overwrite.
if(NOT CMAKE_BUILD_TYPE)#Only do this the on the first run, if the build type hasn't been set prior
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
  FORCE)
endif()

find_package(OpenCV REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${object_recognition_core_INCLUDE_DIRS})

add_subdirectory(src)

if (${ecto_FOUND} AND ${object_recognition_core_FOUND})    
    add_subdirectory(cells)
    add_subdirectory(python)
endif()

# deal with the tests
option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()  
