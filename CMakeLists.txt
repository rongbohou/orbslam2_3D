cmake_minimum_required(VERSION 2.8)
project(ORB_SLAM2_3D)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")

# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
   add_definitions(-DCOMPILEDWITHC11)
   message(STATUS "Using flag -std=c++11.")
elseif(COMPILER_SUPPORTS_CXX0X)
   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
   add_definitions(-DCOMPILEDWITHC0X)
   message(STATUS "Using flag -std=c++0x.")
else()
   message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(OpenCV 2.4.3 REQUIRED)
#find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
# 增加PCL库的依赖
FIND_PACKAGE( PCL REQUIRED COMPONENTS common io visualization filters)
find_package(OpenNI2 REQUIRED)
find_package(OpenNI REQUIRED)

include_directories(
${PROJECT_SOURCE_DIR}
${PROJECT_SOURCE_DIR}/include
#${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
${PCL_INCLUDE_DIRS}
"/usr/include/openni2/"
${OPENNI_INCLUDE_DIR}
)
#设定可执行二进制文件的目录
SET( EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

#并且把该目录设为连接目录
LINK_DIRECTORIES( ${PROJECT_SOURCE_DIR}/lib)
LINK_LIBRARIES( ${OpenNI2_LIBRARY} )

add_executable(rgbd_tum src/rgbd_tum.cc)
target_link_libraries(rgbd_tum 
ORB_SLAM2
${OpenCV_LIBS}
#${EIGEN3_LIBS}
${Pangolin_LIBRARIES}
#${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
#${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)

add_executable(rgbd_my src/rgbd_my.cc)
target_link_libraries(rgbd_my
ORB_SLAM2
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${PCL_LIBRARIES}
)

add_executable(rgbd_kinect1 src/rgbd_kinect1.cpp)
target_link_libraries(rgbd_kinect1
ORB_SLAM2
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${PCL_LIBRARIES}
${OPENNI_LIBRARY_DIR}
)


add_executable(rgbd_xtion src/rgbd_xtion_cc.cpp)
target_link_libraries(rgbd_xtion
ORB_SLAM2
${OpenCV_LIBS}
${Pangolin_LIBRARIES}
${PCL_LIBRARIES}
${OpenNI2_LIBRARY}
)


