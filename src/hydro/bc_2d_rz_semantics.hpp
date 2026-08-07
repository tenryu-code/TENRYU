#pragma once

#include <string>

namespace tenryu::hydro {

enum class BCMaterialNormalKind {
    Slip,            // u_n = 0; no normal mass flux
    StateSupply,     // ghost state imposed at the boundary
    Reflect,         // u_n flipped at boundary (closed wall)
    Free,            // zero gradient
    Pressure,        // imposed pressure
};

enum class BCMaterialTangentialKind {
    Free,            // no constraint on tangential velocity
    NoSlip,          // u_t = 0
};

enum class BCMeshNormalKind {
    ClampedAtBoundary,  // mesh node position fixed at boundary coordinate
    Lagrangian,         // mesh node moves with material
};

enum class BCMeshTangentialKind {
    Lagrangian,         // node moves with material (default ALE)
    ReferenceTarget,    // node target tangential position from reference mesh
    LagrangianFreeze,   // freeze tangential position (no motion)
};

struct BC2DRZAxis {
    BCMaterialNormalKind material_normal;
    BCMaterialTangentialKind material_tangential;
    BCMeshNormalKind mesh_normal;
    BCMeshTangentialKind mesh_tangential;
    // For state_supply, ghost state values
    double supply_rho = 0.0;
    double supply_u_z = 0.0;
    double supply_T = 0.0;
    // Remap flux contract: enabled when PR B is on and mesh_normal=ClampedAtBoundary
    //                     and material_normal=StateSupply, computes rho*(u_n - w_n)*A*dt
    bool open_flow_remap_eligible = false;
};

struct BC2DRZConfig {
    BC2DRZAxis r_inner;
    BC2DRZAxis r_outer;
    BC2DRZAxis z_bottom;
    BC2DRZAxis z_top;
    std::string state_supply_donor_mode = "interior_per_i";
};

}  // namespace tenryu::hydro
