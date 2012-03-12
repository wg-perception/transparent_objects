find_ros_package(pcl)

#TDS: dont glob together sources.
set(src 
    edgeModel.cpp
    glassDetector.cpp
    kPartiteGraph.cpp
    localPoseRefiner.cpp
    pclProcessing.cpp
    pinholeCamera.cpp
    poseError.cpp
    poseEstimator.cpp
    poseRT.cpp
    registerPointClouds.cpp
    silhouette.cpp
    tangentVectorsEstimator.cpp
    TODBaseImporter.cpp
    transparentDetector.cpp
    utils.cpp
    chamfer_matching/chamfer_matching.cpp
    glass_segmentation/geodesicActiveContour.cpp
    glass_segmentation/enhancedGlassSegmenter.cpp
    glass_segmentation/region.cpp
    glass_segmentation/segmentedImage.cpp
)

#EAR: is there a reason to list headers in this target?
add_library(edges_pose_refiner SHARED
    ${src}
    )

set_target_properties(edges_pose_refiner PROPERTIES COMPILE_FLAGS "-fopenmp" LINK_FLAGS "-fopenmp")

#
# Hey!  if you're going to do this, you better test it, thoroughly.  In this case
# it appears the first person to attempt to build on natty was a user.  I'm guessing
# the code worked on lucid.  Or something.
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
  ${ITK_LIBRARIES}
  )

install(TARGETS edges_pose_refiner
  DESTINATION lib
  )

#add_executable(transparentExperiments transparent_experiments/transparentExperiments.cpp transparent_experiments/TODBaseImporter.cpp)
#set_target_properties(transparentExperiments
#  PROPERTIES
#  COMPILE_FLAGS "-std=c++0x"
#  )
#target_link_libraries(transparentExperiments edges_pose_refiner)

add_executable(enhancedGlassSegmenter glass_segmentation/demo.cpp)
target_link_libraries(enhancedGlassSegmenter edges_pose_refiner)

add_executable(createSegmentedImage glass_segmentation/createSegmentedImage.cpp)
target_link_libraries(createSegmentedImage edges_pose_refiner)

add_executable(createSegmentation glass_segmentation/createSegmentation.cpp)
target_link_libraries(createSegmentation edges_pose_refiner)

add_executable(createObjectMask apps/createObjectMask.cpp)
target_link_libraries(createObjectMask edges_pose_refiner)
