find_ros_package(pcl)

file(GLOB src "*.cpp" "chamfer_matching/*.cpp")
file(GLOB inc "../include/edges_pose_refiner/*.hpp")

add_library(edges_pose_refiner SHARED
    ${src}
    ${inc}
    )

set_target_properties(edges_pose_refiner
 PROPERTIES
 COMPILE_FLAGS "-std=c++0x -Wno-deprecated"
 )

target_link_libraries(edges_pose_refiner
  ${pcl_LIBRARIES}
  ${OpenCV_LIBS}
  )

install(TARGETS edges_pose_refiner
  DESTINATION lib
  )

#add_executable(transparentExperiments src/transparent_experiments/transparentExperiments.cpp src/transparent_experiments/TODBaseImporter.cpp src/poseError.cpp)
#rosbuild_add_compile_flags(transparentExperiments -std=c++0x)
#target_link_libraries(transparentExperiments ${PROJECT_NAME})
#target_link_libraries(transparentExperiments pcl_processing)
#rosbuild_add_openmp_flags(transparentExperiments)


#rosbuild_add_executable(stereoCalib2TOD src/stereoCalib2TOD.cpp)
#rosbuild_add_executable(testSolvePnP src/testSolvePnP.cpp)

#rosbuild_add_executable(kinectCalibration src/kinect_calibration/kinectCalibration.cpp)
#target_link_libraries(kinectCalibration edge_model)
#rosbuild_add_openmp_flags(kinectCalibration)

#rosbuild_add_executable(centralizeModel src/centralizeModel.cpp)
#target_link_libraries(centralizeModel edge_model)

#rosbuild_add_executable(kinfuSegmentation src/kinfuSegmentation.cpp)
#target_link_libraries(kinfuSegmentation edge_model)
#target_link_libraries(kinfuSegmentation pcl_processing)


