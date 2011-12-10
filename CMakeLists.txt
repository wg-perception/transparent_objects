project(transparent_objects)

find_package(ecto REQUIRED)
find_package(OpenCV REQUIRED)
find_package(object_recognition REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${object_recognition_INCLUDE_DIRS}
                    )

add_subdirectory(src)

ecto_python_env_gen(${CMAKE_BINARY_DIR}/lib ${CMAKE_CURRENT_SOURCE_DIR}/python)
add_subdirectory(cells)

