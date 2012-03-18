find_package(object_recognition_core QUIET)
if (object_recognition_core_FOUND)
    # pipeline tests
    object_recognition_core_detection_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_detection.transparent_objects)
    object_recognition_core_training_test(${CMAKE_CURRENT_SOURCE_DIR}/../conf/config_training.transparent_objects)
endif()
