# 对拍：同一张图跑 mycv 和 cv::，比 max abs diff。需要真 OpenCV。
find_package(OpenCV REQUIRED)
message(STATUS "对拍用 OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

# 测试里用 dataPath() 读 data/，数据目录编进二进制，任意目录都能跑
add_executable(test_mat test_mat.cpp)
target_link_libraries(test_mat PRIVATE mycv)
add_test(NAME test_mat COMMAND test_mat)

add_executable(test_compare_opencv test_compare_opencv.cpp)
target_link_libraries(test_compare_opencv PRIVATE mycv ${OpenCV_LIBS})
target_compile_definitions(test_compare_opencv PRIVATE MYCV_DATA_DIR="${PROJECT_SOURCE_DIR}/data")
add_test(NAME test_compare_opencv COMMAND test_compare_opencv)

set_tests_properties(test_mat test_compare_opencv PROPERTIES WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
