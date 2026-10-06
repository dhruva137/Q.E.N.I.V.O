cmake_minimum_required(VERSION 3.20)
project(qenivo_native_ipm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/W4 /WX)
else()
  add_compile_options(-Wall -Wextra -Werror)
endif()

option(QENIVO_SANITIZE "Address and undefined-behavior sanitizers" OFF)
if(QENIVO_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer -O1 -g)
  add_link_options(-fsanitize=address,undefined)
endif()

find_package(Threads REQUIRED)

set(QENIVO_IPM_SRC
  amd.cpp
  dissect.cpp
  symbolic.cpp
  factor.cpp
  api.cpp
)

add_library(qenivo_ipm SHARED ${QENIVO_IPM_SRC})
target_link_libraries(qenivo_ipm PUBLIC Threads::Threads)
if(NOT WIN32)
  target_compile_options(qenivo_ipm PRIVATE -fPIC)
endif()

add_executable(nr_ipm_test test_factor.cpp ${QENIVO_IPM_SRC})
target_link_libraries(nr_ipm_test PRIVATE Threads::Threads)

enable_testing()
add_test(NAME nr_ipm_test COMMAND nr_ipm_test)
