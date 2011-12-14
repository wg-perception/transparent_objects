#install targets for all things python
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/transparent_objects
  DESTINATION ${PYTHON_PACKAGES_PATH}
  COMPONENT main
  FILES_MATCHING PATTERN "*.py"
)

