find_pcl_package()

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
                      #FIXME: the dirty line due to problems with linking PCL under Ubuntu 12.04
                      /usr/lib/libboost_python.so;/usr/lib/libboost_thread-mt.so;/usr/lib/libboost_system-mt.so;/usr/lib/libboost_filesystem-mt.so;/usr/lib/libboost_date_time-mt.so;/usr/lib/libboost_iostreams-mt.so;/opt/ros/fuerte/lib/libpcl_common.so;/opt/ros/fuerte/lib/libflann_cpp_s.a;/opt/ros/fuerte/lib/libflann_cpp_s-gd.a;/opt/ros/fuerte/lib/libpcl_kdtree.so;/opt/ros/fuerte/lib/libpcl_octree.so;/opt/ros/fuerte/lib/libpcl_search.so;/opt/ros/fuerte/lib/libpcl_sample_consensus.so;/opt/ros/fuerte/lib/libpcl_filters.so;/opt/ros/fuerte/lib/libpcl_tracking.so;/usr/lib/libOpenNI.so;vtkCommon;vtkRendering;vtkHybrid;/opt/ros/fuerte/lib/libpcl_io.so;/opt/ros/fuerte/lib/libpcl_features.so;/opt/ros/fuerte/lib/libpcl_registration.so;/opt/ros/fuerte/lib/libpcl_segmentation.so;/usr/lib/libqhull.so;/opt/ros/fuerte/lib/libpcl_surface.so;/opt/ros/fuerte/lib/libpcl_visualization.so;/opt/ros/fuerte/lib/libpcl_keypoints.so
#                      ${PCL_LIBRARIES}
                      ${OpenCV_LIBS}
                      ${OpenCV_LIBRARIES}
)

install(TARGETS edges_pose_refiner
  DESTINATION lib
  )

add_executable(sample ../sample/sample.cpp)
target_link_libraries(sample edges_pose_refiner)

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
