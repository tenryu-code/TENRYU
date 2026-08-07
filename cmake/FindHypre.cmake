# Find Hypre with CUDA support.
#
# Provides:
#   Hypre_FOUND
#   Hypre::Hypre
#   HYPRE_INCLUDE_DIR
#   HYPRE_LIBRARY

find_path(HYPRE_INCLUDE_DIR
  NAMES HYPRE.h
  HINTS
    ${HYPRE_DIR}
    $ENV{HYPRE_DIR}
  PATH_SUFFIXES include
)

find_library(HYPRE_LIBRARY
  NAMES HYPRE libHYPRE
  HINTS
    ${HYPRE_DIR}
    $ENV{HYPRE_DIR}
  PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Hypre
  REQUIRED_VARS HYPRE_INCLUDE_DIR HYPRE_LIBRARY
)

if(Hypre_FOUND)
  if(EXISTS "${HYPRE_INCLUDE_DIR}/HYPRE_config.h")
    file(READ "${HYPRE_INCLUDE_DIR}/HYPRE_config.h" HYPRE_CONFIG_TEXT)
    string(FIND "${HYPRE_CONFIG_TEXT}" "#define HYPRE_USING_CUDA 1" HYPRE_CUDA_POS)
    if(HYPRE_CUDA_POS EQUAL -1)
      message(FATAL_ERROR "Hypre was found but HYPRE_USING_CUDA is not enabled")
    endif()
  endif()

  if(CMAKE_CUDA_ARCHITECTURES)
    set(_max_arch 0)
    foreach(_arch IN LISTS CMAKE_CUDA_ARCHITECTURES)
      string(REGEX REPLACE "[^0-9]" "" _arch_num "${_arch}")
      if(_arch_num)
        if(_arch_num GREATER _max_arch)
          set(_max_arch ${_arch_num})
        endif()
      endif()
    endforeach()
    if(_max_arch LESS 70)
      message(FATAL_ERROR "Hypre GPU backend requires CUDA architecture >= 70")
    endif()
  endif()

  if(NOT TARGET Hypre::Hypre)
    add_library(Hypre::Hypre UNKNOWN IMPORTED)
    set_target_properties(Hypre::Hypre PROPERTIES
      IMPORTED_LOCATION "${HYPRE_LIBRARY}"
      INTERFACE_INCLUDE_DIRECTORIES "${HYPRE_INCLUDE_DIR}"
    )
  endif()
endif()

