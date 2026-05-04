cmake_minimum_required(VERSION 3.10)
project(FaceRecognition)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(face_recognition 
    main.cpp 
    dataset.cpp 
    model.cpp 
    metrics.cpp
)

target_link_libraries(face_recognition ${OpenCV_LIBS} pthread)
