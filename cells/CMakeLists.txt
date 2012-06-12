set_ecto_install_package_name(${PROJECT_NAME})

ectomodule(transparent_objects_cells
    module.cpp
    train.cpp
    detect.cpp
    db_transparent_objects.cpp
    modelFiller.cpp
    )

find_pcl_package()
find_package(object_recognition_core REQUIRED)

include_directories(SYSTEM
                    ${object_recognition_core_INCLUDE_DIRS}
                    ${PCL_INCLUDE_DIRS}
)

link_ecto(transparent_objects_cells
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    ${object_recognition_core_LIBRARIES}
    edges_pose_refiner
)

install_ecto_module(transparent_objects_cells)

set(ADD_COLLISION_OBJECTS True)
if (ADD_COLLISION_OBJECTS)
    include_directories(/opt/ros/fuerte/stacks/arm_navigation/arm_navigation_msgs/msg_gen/cpp/include/)
    add_definitions(-DADD_COLLISION_OBJECTS)
    find_package(ROS COMPONENTS roscpp)
    link_ecto(transparent_objects_cells ${roscpp_LIBRARIES})
endif()
