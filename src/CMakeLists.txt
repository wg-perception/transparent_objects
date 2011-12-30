find_ros_package(pcl)

message("TDS FIXME: don't glob together your source.")
file(GLOB src "*.cpp" "chamfer_matching/*.cpp")
file(GLOB inc "../include/edges_pose_refiner/*.hpp")

add_library(edges_pose_refiner SHARED
    ${src}
    ${inc}
    )

#
# Hey!  if you're going to do this, you better test it, thoroughly.  In this case
# it appears the first person to attempt to build on natty was a user.  I'm guessing
# the code worked on oneiric.  Or something.
#
# set_target_properties(edges_pose_refiner
#  PROPERTIES
#  COMPILE_FLAGS "-std=c++0x -Wno-deprecated"
#  )

# Thanks a lot, vtk.  Why are we dependent on vtk?
add_definitions(-Wno-deprecated)

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


