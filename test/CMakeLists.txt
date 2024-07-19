find_package(Boost REQUIRED COMPONENTS program_options)

add_executable(main main.cpp)
target_link_libraries(main ${Boost_LIBRARIES} crypto ssl)
