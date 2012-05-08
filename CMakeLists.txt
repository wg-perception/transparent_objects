project(object_recognition_transparent_objects)
#this needs a stanza to prevent overwrite.
if(NOT CMAKE_BUILD_TYPE)#Only do this the on the first run, if the build type hasn't been set prior
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
  FORCE)
endif()

find_package(object_recognition_core QUIET)
find_package(ecto QUIET)
if (${ecto_FOUND} AND ${object_recognition_core_FOUND})
    set(DO_BUILD_ECTO_CELLS True)
endif()

find_package(OpenCV REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(SYSTEM ${object_recognition_core_INCLUDE_DIRS})

add_subdirectory(src)

# Make sure we can build transparent_objects without ecto or the core
if (DO_BUILD_ECTO_CELLS)
    find_package(catkin REQUIRED)

    catkin_stack()
    catkin_project(object_recognition_transparent_objects
                   DEPENDS Eigen Boost OpenCV
                   INCLUDE_DIRS include
    )

    #install targets for all things python
    catkin_python_setup()

    find_package(ecto REQUIRED)

    add_subdirectory(cells)
    add_subdirectory(python)
endif()

# deal with the tests
option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()  
