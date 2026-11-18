# 对拍：同一张图跑 mycv 和 cv::，比 max abs diff。需要真 OpenCV。
find_package(OpenCV REQUIRED)
message(STATUS "对拍用 OpenCV ${OpenCV_VERSION} from ${OpenCV_DIR}")

# 每个测试写完 main() 后取消对应三行注释
# add_executable(test_mat test_mat.cpp)
# target_link_libraries(test_mat PRIVATE mycv)
# add_test(NAME test_mat COMMAND test_mat)

# add_executable(test_vs_opencv test_vs_opencv.cpp)
# target_link_libraries(test_vs_opencv PRIVATE mycv ${OpenCV_LIBS})
# add_test(NAME test_vs_opencv COMMAND test_vs_opencv)
