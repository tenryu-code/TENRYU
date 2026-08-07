set(TENRYU_CONFIG_STATE_ABI_HEADERS
  "${PROJECT_SOURCE_DIR}/src/core/config.hpp"
  "${PROJECT_SOURCE_DIR}/src/core/state.hpp"
)

function(tenryu_attach_config_state_abi_deps target_name)
  if(NOT TARGET "${target_name}")
    message(FATAL_ERROR "Unknown TENRYU target: ${target_name}")
  endif()

  get_target_property(_tenryu_abi_sources "${target_name}" SOURCES)
  if(NOT _tenryu_abi_sources)
    return()
  endif()

  foreach(_tenryu_abi_source IN LISTS _tenryu_abi_sources)
    if(_tenryu_abi_source MATCHES "\\.(c|cc|cpp|cxx|cu)$")
      set_property(SOURCE "${_tenryu_abi_source}" APPEND PROPERTY
        OBJECT_DEPENDS ${TENRYU_CONFIG_STATE_ABI_HEADERS}
      )
    endif()
  endforeach()
endfunction()
