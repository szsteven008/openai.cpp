cmake_minimum_required(VERSION 3.10)

project(openai.cpp)

set(CMAKE_CXX_STANDARD 11)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    include_directories(/opt/homebrew/include /usr/local/include)
    link_directories(/opt/homebrew/lib /usr/local/lib)
endif()

add_subdirectory(test)