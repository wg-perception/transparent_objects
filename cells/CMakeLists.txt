ectomodule(transparent_objects_cells DESTINATION ${PROJECT_NAME}
                                     INSTALL
    module.cpp
    train.cpp
    detect.cpp
    db_transparent_objects.cpp
    modelFiller.cpp
    )

include_directories(SYSTEM ${catkin_INCLUDE_DIRS}
                           ${PCL_INCLUDE_DIRS}
)

link_ecto(transparent_objects_cells
    ${PCL_LIBRARIES}
    ${Boost_LIBRARIES}
    ${catkin_LIBRARIES}
    edges_pose_refiner
)

set(ADD_COLLISION_OBJECTS False)
if (ADD_COLLISION_OBJECTS)
    include_directories(/opt/ros/fuerte/stacks/arm_navigation/arm_navigation_msgs/msg_gen/cpp/include/
                        /opt/ros/fuerte/stacks/arm_navigation/arm_navigation_msgs/srv_gen/cpp/include/
                        /opt/ros/fuerte/stacks/arm_navigation/planning_environment/include/
                        /opt/ros/fuerte/stacks/arm_navigation/planning_models/include/
                        /opt/ros/fuerte/stacks/arm_navigation/geometric_shapes/include/
                        /opt/ros/fuerte/stacks/robot_model/urdf/include/
                        /opt/ros/fuerte/stacks/robot_model/urdf_parser/include/
                        /opt/ros/fuerte/stacks/robot_model/urdf_interface/include/
                        /opt/ros/fuerte/stacks/robot_model/collada_parser/include/
                        /opt/ros/fuerte/stacks/geometry/tf/include/
                        /opt/ros/fuerte/stacks/geometry_experimental/tf2/include/
                        /opt/ros/fuerte/stacks/arm_navigation/collision_space/include/
                        /opt/ros/fuerte/stacks/physics_ode/opende/opende/include/
    )
    add_definitions(-DADD_COLLISION_OBJECTS)
    find_package(ROS COMPONENTS roscpp)
    link_ecto(transparent_objects_cells ${roscpp_LIBRARIES})
endif()
