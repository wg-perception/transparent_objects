#TDS: dont glob together sources.
set(src 
    edgeModel.cpp
    glassSegmentator.cpp
    localPoseRefiner.cpp
    pclProcessing.cpp
    pinholeCamera.cpp
    poseError.cpp
    poseEstimator.cpp
    poseRT.cpp
    silhouette.cpp
    TODBaseImporter.cpp
    detector.cpp
    utils.cpp
    nonMaximumSuppression.cpp
)

include_directories(SYSTEM
                    ${PCL_INCLUDE_DIRS}
)

#EAR: is there a reason to list headers in this target?
add_library(edges_pose_refiner SHARED
    ${src}
    )

set_target_properties(edges_pose_refiner PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")

# Thanks a lot, vtk.  Why are we dependent on vtk?
add_definitions(-Wno-deprecated)

target_link_libraries(edges_pose_refiner
                      ${PCL_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${OpenCV_LIBRARIES}
)

install(TARGETS edges_pose_refiner
  DESTINATION lib
  )

add_executable(sample ../sample/sample.cpp)
# not sure why but we need Python on Lucid
find_package(PythonLibs)
target_link_libraries(sample edges_pose_refiner
                             ${PYTHON_LIBRARIES}
)

set(USE_ON_ROBOT False)
if (USE_ON_ROBOT)
    add_definitions(-DdDOUBLE)
    #TODO: use macros
    include_directories(/opt/ros/fuerte/stacks/pr2_object_manipulation/perception/tabletop_object_detector/srv_gen/cpp/include/
                        /opt/ros/fuerte/stacks/pr2_object_manipulation/perception/tabletop_object_detector/msg_gen/cpp/include/
                        /opt/ros/fuerte/stacks/arm_navigation/arm_navigation_msgs/msg_gen/cpp/include/
                        /opt/ros/fuerte/stacks/arm_navigation/arm_navigation_msgs/srv_gen/cpp/include/
                        /opt/ros/fuerte/stacks/object_manipulation/household_objects_database_msgs/msg_gen/cpp/include/
                        /opt/ros/fuerte/stacks/object_manipulation/object_manipulation_msgs/msg_gen/cpp/include/
                        /opt/ros/fuerte/stacks/pr2_object_manipulation/perception/tabletop_collision_map_processing/srv_gen/cpp/include/

                        #dependencies for moving arm to a side
                        /opt/ros/fuerte/stacks/object_manipulation/object_manipulator/include/
                        /opt/ros/fuerte/stacks/geometry/tf/include/
                        /opt/ros/fuerte/stacks/geometry/tf/msg_gen/cpp/include/
                        /opt/ros/fuerte/stacks/bullet/include/
                        /opt/ros/fuerte/stacks/geometry/tf/srv_gen/cpp/include/
                        /opt/ros/fuerte/stacks/geometry/eigen_conversions/include/
                        /opt/ros/fuerte/stacks/arm_navigation/kinematics_msgs/srv_gen/cpp/include/
                        /opt/ros/fuerte/stacks/arm_navigation/kinematics_msgs/msg_gen/cpp/include/
                        /opt/ros/fuerte/stacks/arm_navigation/arm_navigation_msgs/include/
                        /opt/ros/fuerte/stacks/pr2_controllers/pr2_controllers_msgs/msg_gen/cpp/include/
                        /opt/ros/fuerte/stacks/arm_navigation_experimental/interpolated_ik_motion_planner/srv_gen/cpp/include/
                        /opt/ros/fuerte/stacks/arm_navigation/planning_environment/include/
                        /opt/ros/fuerte/stacks/arm_navigation/planning_models/include/
                        /opt/ros/fuerte/stacks/arm_navigation/geometric_shapes/include/
                        /opt/ros/fuerte/stacks/robot_model/urdf/include/
                        /opt/ros/fuerte/stacks/robot_model/urdf_parser/include/
                        /opt/ros/fuerte/stacks/robot_model/urdf_interface/include/
                        /opt/ros/fuerte/stacks/robot_model/collada_parser/include/
                        /opt/ros/fuerte/stacks/arm_navigation/collision_space/include/
                        /opt/ros/fuerte/stacks/physics_ode/opende/opende/include/
                        /opt/ros/fuerte/stacks/object_manipulation/object_manipulation_msgs/srv_gen/cpp/include/
                        /opt/ros/fuerte/stacks/pr2_mechanism/pr2_mechanism_msgs/srv_gen/cpp/include/
                        /opt/ros/fuerte/stacks/arm_navigation/arm_navigation_msgs/msg_gen/cpp/include/
    )

    add_executable(pickAndPlace apps/pickAndPlace.cpp)

    find_package(ROS COMPONENTS roscpp actionlib)
    find_package(Boost REQUIRED)

    message("libs: ${object_manipulator_LIBRARIES}")
    target_link_libraries(pickAndPlace ${roscpp_LIBRARIES}
                                       ${Boost_LIBRARIES}
                                       ${actionlib_LIBRARIES}
                                       /opt/ros/fuerte/stacks/arm_navigation/planning_environment/lib/libplanning_environment.so
                                       /opt/ros/fuerte/stacks/geometry/tf/lib/libtf.so
                                       /opt/ros/fuerte/stacks/object_manipulation/object_manipulator/lib/libobject_manipulator_tools.so
                                       /opt/ros/fuerte/stacks/robot_model/colladadom/lib/libminizip.so
                         )
endif()

#add_executable(transparentExperiments transparent_experiments/transparentExperiments.cpp)
#set_target_properties(transparentExperiments
#  PROPERTIES
#  COMPILE_FLAGS "-std=c++0x"
#  )
#target_link_libraries(transparentExperiments edges_pose_refiner)

#add_executable(createObjectMask apps/createObjectMask.cpp)
#target_link_libraries(createObjectMask edges_pose_refiner)

#add_executable(kinfuSegmentation apps/kinfuSegmentation.cpp)
#target_link_libraries(kinfuSegmentation edges_pose_refiner)

#add_executable(poseEstimationDemo apps/poseEstimationDemo.cpp)
#set_target_properties(poseEstimationDemo
#  PROPERTIES
#  COMPILE_FLAGS "-std=c++0x"
#  )
#target_link_libraries(poseEstimationDemo edges_pose_refiner)
