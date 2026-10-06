cmake_minimum_required(VERSION 3.20)
project(qenivo_exact LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(QENIVO_EXACT_SANITIZE "Build the harness with ASan and UBSan" OFF)

add_library(qenivo_exact STATIC
    bigint.cpp
    rational.cpp
    lu.cpp
    refine.cpp)
target_include_directories(qenivo_exact PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(exact_harness test_harness.cpp)
target_link_libraries(exact_harness PRIVATE qenivo_exact)

if(MSVC)
    target_compile_options(qenivo_exact PRIVATE /W4 /WX)
    target_compile_options(exact_harness PRIVATE /W4 /WX)
else()
    target_compile_options(qenivo_exact PRIVATE -Wall -Wextra -Werror)
    target_compile_options(exact_harness PRIVATE -Wall -Wextra -Werror)
endif()

if(QENIVO_EXACT_SANITIZE)
    if(MSVC)
        message(FATAL_ERROR "QENIVO_EXACT_SANITIZE is supported with gcc or clang")
    endif()
    foreach(tgt qenivo_exact exact_harness)
        target_compile_options(${tgt} PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer -g)
        target_link_options(${tgt} PRIVATE -fsanitize=address,undefined)
    endforeach()
endif()

enable_testing()
add_test(NAME exact_harness COMMAND exact_harness)
