cmake_minimum_required(VERSION 3.15)
project(RogueEngine)
set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
set(CMAKE_CXX_STANDARD 17)

# Ищем SFML (добавляем Audio)
find_package(SFML 3 COMPONENTS Graphics Window System Audio REQUIRED)
find_package(Box2D REQUIRED)

set(SOURCE_FILES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

if(EXISTS ${SOURCE_FILES})
    message(STATUS "Found source file: ${SOURCE_FILES}")
    add_executable(RogueEngine ${SOURCE_FILES})
else()
    message(FATAL_ERROR "Source file not found: ${SOURCE_FILES}")
endif()

# Линкуем SFML (добавляем SFML::Audio)
target_link_libraries(RogueEngine
    PRIVATE
    SFML::Graphics
    SFML::Window
    SFML::System
    SFML::Audio 
    box2d::box2d
)

# Копируем ассеты (если нужно)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/assets DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/Debug)