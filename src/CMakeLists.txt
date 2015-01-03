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
    chamfer_matching/chamfer_matching.cpp
    tableSegmentation.cpp
    pcl.cpp
    pclVisualization.cpp
)

include_directories(SYSTEM
                    ${PCL_INCLUDE_DIRS}
                    ${ecto_opencv_INCLUDE_DIRS}
)

#EAR: is there a reason to list headers in this target?
add_library(edges_pose_refiner SHARED
    ${src}
    )

set_target_properties(edges_pose_refiner PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")

# Thanks a lot, vtk.  Why are we dependent on vtk?
add_definitions(-Wno-deprecated)


set(src_fdcm
    fdcm/image/DistanceTransform.cpp
    fdcm/fitline/LFLineFitter.cpp
    fdcm/fitline/LFLineSegment.cpp
    fdcm/fdcm/LMDistanceImage.cpp
    fdcm/fdcm/LMDirectionalIntegralDistanceImage.cpp
    fdcm/fdcm/EIEdgeImage.cpp
    fdcm/fdcm.cpp
)

add_library(fdcm SHARED
    ${src_fdcm}
    )
target_link_libraries(fdcm
                      ${PCL_LIBRARIES}
                      ${OpenCV_LIBRARIES}
)

set(MODEL_CAPTURE False)
if(MODEL_CAPTURE)
    set(src_mrf
        mrf/BP-S.cpp
        mrf/GCoptimization.cpp
        mrf/ICM.cpp
        mrf/maxflow.cpp
        mrf/mrf.cpp
        mrf/TRW-S.cpp
        mrf/graph.cpp
        mrf/LinkedBlockList.cpp
        mrf/MaxProdBP.cpp
        mrf/regions-maxprod.cpp
    )

    add_library(mrf SHARED
        ${src_mrf}
        )
    set_target_properties(mrf PROPERTIES COMPILE_FLAGS "-W -Wall -DUSE_64_BIT_PTR_CAST")



    set(src_model_capture
        modelCapture/modelCapturer.cpp
    )
    add_library(model_capture SHARED
        ${src_model_capture}
        )
    set_target_properties(model_capture PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")
    target_link_libraries(model_capture
                          edges_pose_refiner
                          mrf
    )



    add_executable(captureModel apps/captureModel.cpp)
    set_target_properties(captureModel PROPERTIES
      COMPILE_FLAGS "-fopenmp"
      LINK_FLAGS "-fopenmp"
      )
    target_link_libraries(captureModel edges_pose_refiner
                                       model_capture
                                       mrf
                                       ${PYTHON_LIBRARIES}
    )
endif()

target_link_libraries(edges_pose_refiner
                      fdcm
                      ${PCL_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      ${opencv_candidate_LIBRARIES}
)

install(TARGETS edges_pose_refiner fdcm
  DESTINATION lib
  )



add_executable(sample ../sample/sample.cpp)
# not sure why but we need Python on Lucid
find_package(PythonLibs)
target_link_libraries(sample edges_pose_refiner
                             ${PYTHON_LIBRARIES}
                             ${PCL_LIBRARIES}
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
#target_link_libraries(transparentExperiments edges_pose_refiner
#                                             ${PYTHON_LIBRARIES}
#)

#add_executable(createObjectMask apps/createObjectMask.cpp)
#target_link_libraries(createObjectMask edges_pose_refiner
#                                       ${PYTHON_LIBRARIES}
#)

#add_executable(kinfuSegmentation apps/kinfuSegmentation.cpp)
#target_link_libraries(kinfuSegmentation edges_pose_refiner)

#add_executable(poseEstimationDemo apps/poseEstimationDemo.cpp)
#set_target_properties(poseEstimationDemo
#  PROPERTIES
#  COMPILE_FLAGS "-std=c++0x"
#  )
#target_link_libraries(poseEstimationDemo edges_pose_refiner
#                                         ${PYTHON_LIBRARIES}
#)

#add_executable(evaluateLine2D line2D/evaluation.cpp line2D/line2D.cpp)
#set_target_properties(evaluateLine2D
#  PROPERTIES
#  COMPILE_FLAGS "-std=c++0x"
#  )
#target_link_libraries(evaluateLine2D edges_pose_refiner
#                                     ${PYTHON_LIBRARIES}
#)

#add_executable(labelDataset apps/labelDataset.cpp)
#set_target_properties(labelDataset
#  PROPERTIES
#  COMPILE_FLAGS "-std=c++0x"
#  )
#target_link_libraries(labelDataset edges_pose_refiner
#                                   ${PYTHON_LIBRARIES}#
#)

#add_executable(labelSegmentation apps/labelSegmentation.cpp)
#target_link_libraries(labelSegmentation edges_pose_refiner
#                                        ${PYTHON_LIBRARIES}
#)

#add_executable(showTable apps/showTable.cpp)
#set_target_properties(showTable PROPERTIES
#    COMPILE_FLAGS "-fopenmp"
#    LINK_FLAGS "-fopenmp"
#)
#target_link_libraries(showTable edges_pose_refiner
#                                ${PYTHON_LIBRARIES}
#)

#add_executable(reconstructRotationalModel_singleImage apps/reconstructRotationalModel_singleImage.cpp)
#target_link_libraries(reconstructRotationalModel_singleImage edges_pose_refiner
#                                                 ${OpenCV_LIBRARIES}
#)

add_executable(runDetectionOnDataset apps/runDetectionOnDataset.cpp)
set_target_properties(runDetectionOnDataset PROPERTIES
    COMPILE_FLAGS "-fopenmp"
    LINK_FLAGS "-fopenmp"
)
target_link_libraries(runDetectionOnDataset edges_pose_refiner)

#add_executable(writeVideo apps/writeVideo.cpp)
#target_link_libraries(writeVideo ${OpenCV_LIBRARIES})

#if (${catkin_FOUND})
#    add_executable(dumpBagFile apps/bags/dumpBagFile.cpp)

#    target_link_libraries(dumpBagFile ${Boost_LIBRARIES}
#                                      ${catkin_LIBRARIES}
#                                      ${OpenCV_LIBRARIES}
#                                      ${PYTHON_LIBRARIES}
#    )
#endif()
