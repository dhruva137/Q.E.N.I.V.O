cmake_minimum_required(VERSION 3.20)
project(qnv_compat LANGUAGES C CXX)

# Standalone build of the QNV interface and its test program, linked statically with the
# QENIVO C ABI and the native simplex core. The Python-backed checks need QENIVO_PYTHON_DLL,
# QENIVO_SRC, PYTHONHOME and PYTHONPATH set (capi/compat/build_qnv.py sets them).
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CAPI "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_library(qnv_compat STATIC
    qnv_compat.c
    "${CAPI}/qenivo_c.cpp"
    "${CAPI}/python_bridge.cpp"
    "${CAPI}/../src/qenivo/native/simplex_core.cpp")
target_compile_definitions(qnv_compat PUBLIC QENIVO_BUILD)
target_include_directories(qnv_compat PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CAPI}")
if(UNIX)
    target_link_libraries(qnv_compat PUBLIC ${CMAKE_DL_LIBS} pthread)
endif()

add_executable(test_qnv_compat test_qnv_compat.c)
target_link_libraries(test_qnv_compat PRIVATE qnv_compat)
if(MINGW)
    target_link_options(test_qnv_compat PRIVATE -static)
endif()

enable_testing()
add_test(NAME qnv_compat COMMAND test_qnv_compat)
