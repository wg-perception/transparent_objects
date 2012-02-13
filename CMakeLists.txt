project(transparent_objects)
#this needs a stanza to prevent overwrite.
if(NOT CMAKE_BUILD_TYPE)#Only do this the on the first run, if the build type hasn't been set prior
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
    "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
  FORCE)
endif()

find_package(ecto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(object_recognition REQUIRED)

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${object_recognition_INCLUDE_DIRS}
                    )

add_subdirectory(src)
add_subdirectory(python)

ecto_python_env_gen(${CMAKE_BINARY_DIR}/lib ${CMAKE_CURRENT_SOURCE_DIR}/python)
add_subdirectory(cells)

option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()  

