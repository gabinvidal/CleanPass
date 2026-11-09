# Finds an Open Image Denoise SDK (headers + import library).
# Set OIDN_ROOT to the extracted release folder (oidn-2.5.1.x64.windows).
# build.ps1 gives its runtime private names so that it cannot clash with Nuke's TBB.

set(_oidn_hints "${OIDN_ROOT}" "$ENV{OIDN_ROOT}" "${CMAKE_CURRENT_LIST_DIR}/../deps/oidn-2.5.1.x64.windows")

find_path(OpenImageDenoise_INCLUDE_DIR NAMES OpenImageDenoise/oidn.hpp
  HINTS ${_oidn_hints} PATH_SUFFIXES include NO_DEFAULT_PATH)
find_library(OpenImageDenoise_LIBRARY NAMES OpenImageDenoise
  HINTS ${_oidn_hints} PATH_SUFFIXES lib NO_DEFAULT_PATH)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenImageDenoise
  REQUIRED_VARS OpenImageDenoise_INCLUDE_DIR OpenImageDenoise_LIBRARY)

if(OpenImageDenoise_FOUND AND NOT TARGET OpenImageDenoise::OpenImageDenoise)
  add_library(OpenImageDenoise::OpenImageDenoise UNKNOWN IMPORTED)
  set_target_properties(OpenImageDenoise::OpenImageDenoise PROPERTIES
    IMPORTED_LOCATION "${OpenImageDenoise_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OpenImageDenoise_INCLUDE_DIR}")
endif()
mark_as_advanced(OpenImageDenoise_INCLUDE_DIR OpenImageDenoise_LIBRARY)
