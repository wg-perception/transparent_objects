find_package(object_recognition_core QUIET)
if (object_recognition_core_FOUND)
    # pipeline tests
    object_recognition_core_config_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/detection.ros.ork)
    #object_recognition_core_config_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/training.ros.ork)
endif()
