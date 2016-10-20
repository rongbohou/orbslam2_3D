CMAKE_MINIMUM_REQUIRED( VERSION 2.8 ) #设定版本
MESSAGE(STATUS "USING BUNDLED FindOpenNI.cmake ...")
FIND_PATH(OPENNI_INCLUDE_DIR NAMES XnCppWrapper.h
    PATHS
    /usr/include/ni 
  )

FIND_LIBRARY(OPENNI_LIBRARY_DIR NAMES OpenNI
     PATHS
     /usr/lib
   )

