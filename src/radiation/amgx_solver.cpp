#include "radiation/amgx_solver.hpp"

#include <cstdlib>
#include <mutex>
#include <string>

#include "core/error.hpp"

#if TENRYU_HAVE_AMGX && TENRYU_ENABLE_AMGX
#include <amgx_c.h>
#endif

namespace tenryu::radiation {
namespace {

#if TENRYU_HAVE_AMGX && TENRYU_ENABLE_AMGX

void amgx_check(const AMGX_RC rc, const char* context) {
  if (rc == AMGX_RC_OK) {
    return;
  }
  char buffer[4096] = {};
  AMGX_get_error_string(rc, buffer, static_cast<int>(sizeof(buffer)));
  TENRYU_ASSERT(false, std::string(context) + ": " + buffer);
}

void initialize_amgx_once() {
  amgx_check(AMGX_initialize(), "AMGX_initialize failed");
  amgx_check(AMGX_initialize_plugins(), "AMGX_initialize_plugins failed");
  std::atexit([]() {
    AMGX_finalize_plugins();
    AMGX_finalize();
  });
}

const char* amgx_config_path(const std::string& preset) {
  TENRYU_ASSERT(preset == "AGGREGATION_JACOBI",
                "Unsupported FLD2D AmgX preset");
  if (const char* env_path = std::getenv("TENRYU_AMGX_FLD_CONFIG");
      env_path != nullptr && env_path[0] != '\0') {
    return env_path;
  }
  return "resources/amgx_fld_config.json";
}

#endif

}  // namespace

bool amgx_solver_available() {
#if TENRYU_HAVE_AMGX && TENRYU_ENABLE_AMGX
  return true;
#else
  return false;
#endif
}

void solve_with_amgx(const AmgxSolveRequest& request) {
#if TENRYU_HAVE_AMGX && TENRYU_ENABLE_AMGX
  TENRYU_ASSERT(request.n_rows > 0, "AmgX solve requires positive row count");
  TENRYU_ASSERT(request.nnz > 0, "AmgX solve requires positive nnz");
  TENRYU_ASSERT(request.row_offsets != nullptr, "AmgX solve missing row offsets");
  TENRYU_ASSERT(request.col_indices != nullptr, "AmgX solve missing column indices");
  TENRYU_ASSERT(request.values != nullptr, "AmgX solve missing matrix values");
  TENRYU_ASSERT(request.rhs != nullptr, "AmgX solve missing rhs");
  TENRYU_ASSERT(request.solution != nullptr, "AmgX solve missing solution");

  static std::once_flag init_flag;
  std::call_once(init_flag, initialize_amgx_once);

  AMGX_config_handle cfg = nullptr;
  AMGX_resources_handle resources = nullptr;
  AMGX_matrix_handle matrix = nullptr;
  AMGX_vector_handle rhs = nullptr;
  AMGX_vector_handle solution = nullptr;
  AMGX_solver_handle solver = nullptr;
  const AMGX_Mode mode = AMGX_mode_dDDI;

  amgx_check(AMGX_config_create_from_file(&cfg, amgx_config_path(request.preset)),
             "AMGX_config_create_from_file failed");
  amgx_check(AMGX_resources_create_simple(&resources, cfg),
             "AMGX_resources_create_simple failed");
  amgx_check(AMGX_matrix_create(&matrix, resources, mode),
             "AMGX_matrix_create failed");
  amgx_check(AMGX_vector_create(&rhs, resources, mode),
             "AMGX_vector_create rhs failed");
  amgx_check(AMGX_vector_create(&solution, resources, mode),
             "AMGX_vector_create solution failed");
  amgx_check(AMGX_solver_create(&solver, resources, mode, cfg),
             "AMGX_solver_create failed");

  amgx_check(AMGX_matrix_upload_all(matrix,
                                    request.n_rows,
                                    request.nnz,
                                    1,
                                    1,
                                    request.row_offsets,
                                    request.col_indices,
                                    request.values,
                                    nullptr),
             "AMGX_matrix_upload_all failed");
  amgx_check(AMGX_vector_upload(rhs, request.n_rows, 1, request.rhs),
             "AMGX_vector_upload rhs failed");
  amgx_check(AMGX_vector_upload(solution, request.n_rows, 1, request.solution),
             "AMGX_vector_upload solution failed");
  amgx_check(AMGX_solver_setup(solver, matrix), "AMGX_solver_setup failed");
  amgx_check(AMGX_solver_solve(solver, rhs, solution), "AMGX_solver_solve failed");

  AMGX_SOLVE_STATUS status = AMGX_SOLVE_FAILED;
  amgx_check(AMGX_solver_get_status(solver, &status),
             "AMGX_solver_get_status failed");
  TENRYU_ASSERT(status == AMGX_SOLVE_SUCCESS,
                "AmgX did not report solve success for FLD2D");
  amgx_check(AMGX_vector_download(solution, request.solution),
             "AMGX_vector_download solution failed");

  AMGX_solver_destroy(solver);
  AMGX_vector_destroy(solution);
  AMGX_vector_destroy(rhs);
  AMGX_matrix_destroy(matrix);
  AMGX_resources_destroy(resources);
  AMGX_config_destroy(cfg);
#else
  (void)request;
  TENRYU_ASSERT(false,
                "AmgX solve wrapper is not linked in this build; use "
                "linear_solver_2d=\"cusparse_cg_jacobi\" or rebuild with AmgX");
#endif
}

}  // namespace tenryu::radiation
