cmake_minimum_required(VERSION 3.20)
project(qenivo_capi LANGUAGES C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(QENIVO_SIMPLEX_CORE "${CMAKE_CURRENT_SOURCE_DIR}/../src/qenivo/native/simplex_core.cpp"
    CACHE FILEPATH "Existing native simplex. The C API compiles it and does not modify it.")

add_library(qenivo_c SHARED
    qenivo_c.cpp
    python_bridge.cpp
    "${QENIVO_SIMPLEX_CORE}")
target_compile_definitions(qenivo_c PRIVATE QENIVO_BUILD)
target_include_directories(qenivo_c PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(MSVC)
    target_compile_options(qenivo_c PRIVATE /W4)
    set_source_files_properties(qenivo_c.cpp python_bridge.cpp PROPERTIES COMPILE_OPTIONS "/WX")
else()
    target_compile_options(qenivo_c PRIVATE -Wall -Wextra -fPIC)
    set_source_files_properties(qenivo_c.cpp python_bridge.cpp PROPERTIES COMPILE_OPTIONS "-Wall;-Wextra;-Werror")
endif()

if(MINGW)
    target_link_options(qenivo_c PRIVATE -static)
endif()
if(UNIX)
    target_link_libraries(qenivo_c PRIVATE ${CMAKE_DL_LIBS})
endif()

function(qenivo_c_program target source)
    add_executable(${target} ${source})
    target_link_libraries(${target} PRIVATE qenivo_c)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /WX)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Werror)
    endif()
endfunction()

qenivo_c_program(qenivo_capi_test test_harness.c)
qenivo_c_program(example_min_neg_x examples/min_neg_x.c)
qenivo_c_program(example_two_binary examples/two_binary.c)

enable_testing()
add_test(NAME capi_lp COMMAND qenivo_capi_test "${CMAKE_CURRENT_SOURCE_DIR}/../src")
add_test(NAME example_lp COMMAND example_min_neg_x)
add_test(NAME example_milp COMMAND example_two_binary "${CMAKE_CURRENT_SOURCE_DIR}/../src")
