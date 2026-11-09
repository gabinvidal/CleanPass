# CleanPass - Nuke NDK plugin
#   cmake -B build/nuke17 -G "Visual Studio 17 2022" -A x64 ^
#         -DNUKE_ROOT="C:/Program Files/Nuke17.0v3" -DOIDN_ROOT="<oidn-2.5.1 sdk>"
#   cmake --build build/nuke17 --config Release
# build.bat does all of this for every installed Nuke version.

cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(CleanPass VERSION 1.0.0 LANGUAGES CXX)
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW) # NUKE_ROOT is also a search root for find_package(Nuke)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(NOT NUKE_ROOT)
  message(FATAL_ERROR "Set NUKE_ROOT to your Nuke install, e.g. -DNUKE_ROOT=\"C:/Program Files/Nuke17.0v3\"")
endif()
set(Nuke_DIR "${NUKE_ROOT}/cmake")
find_package(Nuke REQUIRED)
find_package(OpenImageDenoise REQUIRED)

add_nuke_plugin(CleanPass
  src/CleanPass.cpp
  src/cp_core.cpp
  src/cp_deflicker.cpp
  src/cp_pipeline.cpp
  src/cp_oidn.cpp
)
target_compile_definitions(CleanPass PRIVATE FN_PLUGIN)
target_link_libraries(CleanPass PRIVATE OpenImageDenoise::OpenImageDenoise)
if(MSVC)
  # /fp:precise keeps std::isfinite reliable (NaN / inf repair)
  target_compile_options(CleanPass PRIVATE /O2 /W3 /fp:precise /EHsc /MP /Zi
    /wd4996) # C4996: deprecation warnings coming from the Nuke 17 headers themselves
  # keep a .pdb next to the DLL so crash reports can be traced back to a line
  target_link_options(CleanPass PRIVATE /DEBUG /OPT:REF /OPT:ICF)
endif()
set_target_properties(CleanPass PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
