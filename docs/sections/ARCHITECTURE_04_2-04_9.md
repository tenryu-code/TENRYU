<!-- 分割元: docs/ARCHITECTURE.md | このファイルは参照用です。原本（docs/ARCHITECTURE.md）が権威です。 -->
### 4.2 mesh/
**責務**：計算格子（ALE/Lagrangian）と幾何演算、近傍構造

- `Mesh::Topology`：論理構造格子のインデックスとフラグ管理
- `Mesh::Geometry`：体積、面積、法線、中心、RZ回転体積（2πr）など
- `Mesh::Search`：退役（光子粒子のセル同定。§4.2.3）
- `Mesh::Remap`：rezoning後の保存的remap（v1.0で必須、NUMERICS §3.3.4）

`Mesh::VoronoiRZ` emits a global `node_r`/`node_z` table and per-cell CCW `node_indices`, while retaining compatibility coordinate cycles populated from that table. Interior generator triples, boundary generator-pair/segment tuples, and input domain-corner vertex indices are canonical keys whose coordinates are computed once (domain corners are copied verbatim), so shared topology is bitwise deterministic and reciprocity and valence collapse operate on node ids. The v1 cocircular policy accepts roundoff-scale sliver edges from distinct generator triples, merges only bitwise zero-length edges before collapse, and retains unreferenced nodes without compaction.

- `Mesh::PolygonMesh`（`src/mesh/polygon_mesh.{hpp,cpp}`、ALE P0A F1）：トポロジ可変
  多角形 mesh の canonical host 状態 — 次数上限付き CSR（`kMaxPolygonDegree`=8）、
  stable cell/node ID + 単調 allocator、AMR-ready lineage（parent id／lineage epoch）、
  reference epoch と別立ての topology epoch、FNV-1a 表現契約 hash。検証は
  result-struct イディオム（`PolygonMeshValidation`、`MeshGeometryResult` と同型）+
  構築時 fail-loud。structured quad import が現行 mesh を特殊例として再現。P0A では
  mutation API・device mirror・幾何モーメントを持たない（幾何は F4 が権威）。
- `Mesh::PentagonBeltShell`（`src/mesh/pentagon_belt_shell.cu`）：五角形
  belt 遷移リング付き polar-shell の生産 builder。ring-major 節点（θ は最細 ladder の
  サブサンプリング＝2:1 入れ子 bitwise・南半球は鏡映）、block-major セル（2K+1 block、
  role PENTAGON_BELT、orientation −1）、stride-8 CSR＋nverts-aware bilateral 検証、
  解析殻体積ゲート、内縁 node flag（NODE_INNER_PHYSICAL_BOUNDARY）。幾何は multiblock
  CSR kernel（容量 8）＋汎用 ring Svec tangent balance（外環・内環）。checkpoint は
  topology v3＋v4（corner_stride・cell_nverts）。離散化契約は NUMERICS §3.2.y。
- `Mesh::Moments`（`src/mesh/rz_moments.cuh`、ALE P0A F4）：厳密 RZ 重み付き幾何
  モーメント庫（単一幾何権威）— 三角形／多角形の閉形式 signed r-モーメント
  （area, ∫r, ∫z, ∫r², ∫rz、λ 重み一次）、fan／star 三角形分割、線形再構成転送
  モーメント、回転面測度、決定論的 orient 述語、overlay partition gate。全演算は
  contraction-immune 2 層規約（関数内明示 fma + 公開関数の最終積 fma(a,b,0.0)）で
  host≡device bitwise（sm_89 実測、memcmp ctest 常設）。既存
  `hydro::ale::detail::rz_signed_quad_volume` の置換統合は後続 wave。
- `Mesh::ReferenceFlatPath` (`src/mesh/reference_flat_path.{hpp,cpp}`):
  construction-coordinate corner typing and host exact predicates shared by
  the Hydro G1 predictor/corrector path guard and max-min repair. It owns the
  adaptive exact reference orientation certificate, exact-dyadic quadratic
  interval/equality-order tests, and continuous nonincident-edge collision
  root isolation for cells containing structural flat corners. CUDA receives
  only the resulting immutable flat-corner byte mask; regular-corner strict
  evaluation remains in `Hydro::CornerJacobianQuality`.

#### 4.2.1 Mesh::Topology — 論理構造格子

v1.0の既定メッシュは **論理構造格子**（structured quad mesh）である。
ALE rezone 後も論理トポロジ（i,j）は変わらず、物理座標のみ変化する。
S1の `topology_scheme="multiblock_cart_core_polar_shell"` は default-off の
mesh topology extension で、single-block では CSR 等の非構造格子データ構造を
確保しない。

```cpp
struct MeshTopology {
    int nr, nz;                 // single_block dimensions; multiblock stores shell Nr and 4*Nc
    int n_cells;                // single_block: nr * nz; multiblock: block-total cell count
    int n_nodes;                // single_block structured count; multiblock: shared-node total

    // linear indexing: cell(i,j) = i * nz + j （row-major, i=r方向, j=z方向）
    // node(i,j) = i * (nz+1) + j

    // 境界フラグ（device配列、初期化後は read-only）
    uint8_t* node_flags;        // OWNED [n_nodes] bit flags: BOUNDARY, AXIS, CENTER, POLE_AXIS

    optional<MultiBlockTopology> multiblock; // nullopt for single_block
};

struct MultiBlockTopology {
    vector<int> cell_block_id;          // [n_cells_total], dense block id
    vector<int> cell_id_stable;         // [n_cells_total], block-major stable ids
    vector<uint8_t> cell_nverts;        // [n_cells_total], active corners (3 for cap triangles, 4 otherwise)
    vector<int> cell_orientation_sign;  // [n_cells_total], +1/-1 canonical geometry sign
    vector<int> cell_node_csr_offsets;  // [n_cells_total+1]
    vector<int> cell_node_csr_indices;  // [total_corners]
    vector<int> face_adj_csr_offsets;   // [n_cells_total+1]
    vector<int> face_adj_csr_indices;   // [total_faces]
    vector<int> face_bc_tags;           // [total_faces], BoundaryKind as int
    int block_count;                    // 3 for gamma MVP, 5 for half-butterfly
    vector<int> block_role;             // semantic role enum per block
    vector<int> block_cell_counts;      // [block_count]
    vector<int> block_node_counts;      // [block_count], owner-written nodes
    int n_cells_core, n_cells_bridge, n_cells_shell;
    int n_nodes_core, n_nodes_bridge_interior, n_nodes_shell;
};

struct ReverseCellNodeCSR {
    vector<int> node_offsets; // [n_nodes+1]
    vector<int> node_cells;   // incident cell ids, sorted by (cell_id_stable, corner)
    vector<int> node_corners; // incident corner slot in {0,1,2,3}
};

struct Mesh {
    DeviceArray<int> multiblock_cell_node_csr_offsets; // device mirror, multiblock only
    DeviceArray<int> multiblock_cell_node_csr_indices; // device mirror, multiblock only
    DeviceArray<int> multiblock_reverse_csr_node_offsets; // [n_nodes+1], multiblock only
    DeviceArray<int> multiblock_reverse_csr_node_cells;   // incident cells, multiblock only
    DeviceArray<int> multiblock_reverse_csr_node_corners; // incident corners, multiblock only
};

// フラグ定数
enum NodeFlag : uint8_t {
    NODE_NONE      = 0,
    NODE_BOUNDARY  = 1 << 0,    // 外側境界上
    NODE_AXIS      = 1 << 1,    // R=0 軸上（2D_RZ）
    NODE_CENTER    = 1 << 2,    // r=0 center (1D_SPH or tri_fan origin row)
    NODE_POLE_AXIS = 1 << 3,    // spherical-polar theta=0/pi pole axis
};
```

**single-block の面（Face）はデータ構造を持たない**：構造格子では面は暗黙的に定義される。
セル `(i,j)` の4面は隣接セル `(i±1,j)`, `(i,j±1)` との間の面であり、
面積・法線はジオメトリ計算時にon-the-flyで算出する。
Multiblock では `MultiBlockTopology` の CSR vectors が cell-to-node と
face-adjacency の source of truth になる。`cell_block_id` と
`cell_id_stable` は block-major order の stable cell identity を保持し、
`block_count` と per-block counts は allocation、serialization、diagnostics の
shape contract になる。

**Runtime dispatch (single_block vs multiblock):**

When `mesh.topology_scheme = "single_block"` (default), the hydro stack uses
structured `c = i*nz + j` cell indexing and the structured `(nr+1)*(nz+1)`
node grid throughout. This path is byte-identical to TENRYU's pre-S2 behavior.
Tri_fan center treatment is similarly unchanged for
`polar_center_treatment = "tri_fan"` decks.

When `mesh.topology_scheme` selects a multiblock mode, hydro
consumers branch on `mesh_topo_is_multiblock(cfg)` and dispatch CSR-aware
kernel variants. These read connectivity from
`state.mesh.multiblock_cell_node_csr_offsets/indices` (cell-to-node CSR) and
`state.mesh.multiblock_reverse_csr_node_*` (reverse CSR for node-loop
force/mass accumulation). Determinism on GPU is guaranteed via fixed
incident-corner sort order `(cell_id_stable ASC, corner_index ASC)` and no
`atomicAdd`. The numerical equations and units are documented in NUMERICS
§3.2; the corresponding namelist keys and planned S2 feature gate are in
SPECIFICATION §6.4.2.
The S5 `rounded_half_butterfly` transition scheme is a geometry generator
variant inside the same three-block multiblock architecture: it changes only
bridge interior node coordinates via rounded-superellipse-cap TFI and optional
mesh-local elliptic smoothing.  Core/shell seam nodes, block counts, CSR
connectivity, reverse CSR, face adjacency, seam tags, HDF5 layout, and remap
dispatch remain identical to the legacy `hermite_bridge` topology.
`topology_scheme="multiblock_half_butterfly_5block"` uses the same multiblock
CSR containers with five block table entries: central core, north fan, east
fan, south fan, and polar shell. Shared central-fan, fan-fan diagonal, and
fan-shell seams resolve to one owner-written node id, so all incident blocks
read one coordinate value. The shell keeps the existing polar-shell node
ordering. The finite-valence multiblock vertices at the half-plane axis and
fan diagonals replace the former smooth square-core diagonal corner; no
single logical cell corner is required to carry both edge tangents of a rounded
seam. `cell_orientation_sign` records the canonical signed-volume convention
(central core `+1`, fans and shell `-1`) so geometry kernels can use one
positive-volume predicate across the mixed orientation. B-S1 validated this
mesh/schema topology; B-S2 makes it hydro-runnable by retargeting ALE/remap,
scaled-reference orientation, polar-shell pressure BC, path-admissibility
diagnostics, radial/Fourier diagnostics, committed mesh-quality observation,
and GCL audit code to block-role metadata, CSR connectivity, and
`cell_orientation_sign` instead of three-block structured offsets. The B-S2
smoke acceptance is intentionally gentle: a closed, uniform, at-rest 2T
Lagrangian plus ALE/remap run. B-S3 still owns seam flux under gradients and
B-S4 still owns the decisive compression gate. Single-block and three-block
multiblock paths are unchanged and remain byte-identical in their default/off
configurations.
`topology_scheme="multiblock_half_butterfly_trifan_cap_5block"` replaces the
central core block with a pinned tri-fan cap. The cap owns one apex
\(O=(R,Z)=(0,0)\) flagged `NODE_CENTER|NODE_AXIS|NODE_BOUNDARY`, \(4N_c\)
first-ring triangular cells with `cell_nverts=3`, and outer cap quad rings.
The cap outer ring reuses the fan inner-seam node ids, so cap/fan exchange is
ordinary bilateral CSR over `unique_internal_faces`; there is no special
one-sided cap seam. The fan and shell blocks keep the half-butterfly ownership,
counts, and CSR containers.
The active-slot topology contract is centralized in
`mesh.hpp::mesh_topo_cell_active_nverts` and
`mesh.hpp::mesh_topo_active_local_face_corners`: storage remains four slots for
every cell, but a cap triangle uses active vertex slots `{0,1,2}` and active
local faces `{1,2,3}`. Storage slot 3 and local face 0 are inactive and must
not contribute geometry, flux, mass, velocity projection, or quality samples.
Boundary dispatch follows the topology, not `logical_mesh_2d`: multiblock
`apply_boundary_2d` always uses the node-flag boundary path and never the
structured `i*stride+j` path, even when the deck uses the polar logical mesh
setting for gamma-MVP construction.

S3 verifies this runtime dispatch at seam-conservation level with four gates:
constant-state seam GCL, uniform-pressure force balance, homothetic three-ring
symmetry, and dynamic A4 spherical smoke convergence. These gates add no new
runtime architecture or namelist keys; they constrain the existing CSR-aware
multiblock path and its boundary-projection ordering. Thresholds and empirical
γ MVP floors are documented in NUMERICS §3.2.x.

**Multiblock ALE dispatch (S4-T1-next)**: When `mesh_topo_is_multiblock` AND
`numerics.ale.enabled`, `apply_ale` (`src/hydro/ale_driver.cu`) enters the CSR
production path. Under `mesh.motion="ale"`, the production driver reaches this
path from the scheduled post-step ALE block so `apply_multiblock_csr_ale_step`
controls the per-step cadence. The default
`numerics.ale.multiblock_cross_seam_rezone_enabled=false` runs T5a per-block
CSR Winslow smoothing and T3/T4 CSR conservative remap; setting the flag true
uses the T5b cross-seam smoother before the same CSR remap. Reference-barrier
ALE uses CSR admissibility and the CSR remap path for multiblock states. When
`numerics.ale.multiblock_scaled_reference_enabled=true`, multiblock reference
targets are rebuilt through `hydro/ale_scaled_reference.cuh`: the driver/remap
path computes the current outer-ring scale \(\alpha(t)\), installs
\(\alpha X^0\) plus \(\alpha^3 V^0\) as the conservative target, and the
reference-barrier target builder uses the same scaled coordinates. The target
volume orientation comes from `cell_orientation_sign`, so the five-block
central/fan/shell winding no longer depends on `cell < n_cells_core`
inference. With the flag false, the static IC reference arrays are unchanged.
When `Numerics.ale.multiblock_differential_reference_enabled=true`, the CSR
remap path calls
`prepare_multiblock_differential_reference_if_enabled(state, cfg)` before the
scaled γ-MVP installer. A true return means
`install_multiblock_differential_reference` has installed the Lagrangian-close
reference and recomputed `State.cell_vol_initial`; a false return falls back to
`prepare_scaled_gamma_mvp_reference_if_enabled`. The differential path is
therefore an opt-in gate-replacement for multiblock 2D_RZ, while γ-MVP remains
the legacy default. Its helper functions in `src/hydro/ale_tracking_reference.cuh`
are `build_multiblock_reference_xi_initial`,
`build_multiblock_differential_band_scales`,
`install_multiblock_differential_reference`, and
`prepare_multiblock_differential_reference_if_enabled`. These build the fixed
radial ξ/director fields, compute ξ-band median/smoothed/cap-limited/monotone
corrections, run the orientation-aware CSR reference line search, write
`State.x_r_reference/x_z_reference`, and recompute reference volumes from the
accepted coordinates. `src/hydro/ale_reference_diagnostics.cuh` provides
pre-remap sampling for this experiment and does not affect remap state.
`Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled` is a
second replacement reference mode, mutually exclusive with both scaled and
differential multiblock reference modes. The CSR ALE driver calls the
center/quality patch builder, uploads `node_rezone_active` as a masked CSR
Winslow seed mask plus `node_patch_boundary` for frozen patch edges, then uses
the ALE motion trigger to run the local Phi-barrier optimizer only when the seed
patch corner-J quality crosses the center-patch on-demand threshold. The seed or
barrier output then goes through the existing CSR admissibility line search.
Bulk, patch-boundary, axis, pole-axis, center, and cap-apex nodes stay at the
post-hydro Lagrangian coordinates. The driver installs the accepted reference
and exact RZ reference volumes for remap, then restores the Lagrangian donor
coordinates before CSR conservative remap. The remap installer path skips the
differential/scaled reference installers under this flag so they cannot clobber
the installed center-patch target. The host-only patch/mask builder lives in
`src/hydro/multiblock_center_patch_reference.cuh`; the device-side local
barrier optimizer lives in `src/hydro/center_patch_barrier_optimizer.cuh`. The
builder constructs the permanent and quality-seeded cell patch, dilates it
through multiblock face adjacency, and returns host `cell_in_patch`,
`node_rezone_active`, and `node_patch_boundary` masks while persisting the
runtime-only `State.center_patch_latch` hysteresis bitmask.
With `TENRYU_I1B_DIFFREF_DIAG`, CSR remap fills
`AleRemap2DRZResult::eta_contact_step` and emits `[eta_contact_diag]`; the ALE
driver forwards it through `AleStepResult::eta_contact_step` and
`eta_contact_cumulative`. The center-patch branch also exposes the CSR
line-search `sigma_accepted` as `AleStepResult::center_patch_sigma_accepted`.
The CSR conservative remap has two hydro-energy branches. With
`Numerics.hydro.total_energy_remap_2d_rz=false`, it uses the legacy separate
electron/ion internal-energy remap and legacy cell-to-node velocity projection.
With `total_energy_remap_2d_rz=true` on
`topology_scheme="multiblock_half_butterfly_trifan_cap_5block"`, it builds and
remaps the extensive material total energy, remaps \(mY_e^{int}\), uses the
corrected swept-volume convention, applies the CSR hydro mass-positivity
face-flux limiter, projects cell velocity to nodes with
RZ corner-mass weights, applies the KE-realizability nodal velocity limiter, and
recovers \(e_e/e_i\) from remapped total energy minus the actual post-limiter
corner kinetic energy. Floor energy from the total-energy recovery is reported
through `AleRemap2DRZResult::E_floor_injected`, then
`AleStepResult::E_floor_injected`, and finally the driver energy budget; it is
distinct from the CSR mass-floor closure term `E_redistribution_unresolved`.
`Numerics.ale.swept_volume_sign_fixed` is corrected-only since epoch 2
(2026-08-05); the legacy convention has been removed.
The default/off path is kept byte-identical. Stage 4b adds a default-off CSR
Option B corner-velocity remap component in
`hydro::ale::csr_optionb_corner_velocity_remap_component`
(`src/hydro/ale_remap_2d_rz.hpp`, `src/hydro/ale_remap_2d_rz.cu`).  It is
enabled only by the process environment flag
`TENRYU_OPTIONB_CSR_CORNER_VELOCITY_REMAP=1` and writes to
`CsrOptionBCornerVelocityRemapBuffers`, not to production `State::v_r/v_z`.
The component builds deterministic colors for `unique_internal_faces`, launches
Option B FCT corner-momentum packets color by color, applies the Option B
affine-orthogonal hourglass filter, and scatters through
`multiblock_reverse_csr_node_*`.  It shares the existing CSR swept-volume and
mass-flux-scale kernels with scalar remap.
Stage 4c wires that component into `conservative_remap_csr` behind the separate
default-off process environment flag `TENRYU_I1B_OPTIONB_VELREMAP=1`.  When the
flag is unset, the driver does not allocate Option B buffers or launch Option B
kernels.  When set, the scalar path's effective swept-volume convention includes
the env flag, the driver completes scalar CSR remap first, passes the finished
`d_mass_new` to the Option B component so the corner-mass cell sums match the
scalar mass state, copies the reverse-CSR scattered velocity to `State::v_r/v_z`,
and skips both `csr_project_cell_velocity_to_nodes*` kernels. Boundary
conditions are applied after the copy. If total-energy remap is also enabled,
the subsequent KE-realizability scale and `E-K` recovery consume that Option B
nodal velocity.
The 5-block half-butterfly axis ALE path is an opt-in, target-only extension:
`mesh::build_full_axis_node_chain` derives the full shared-node physical
\(R=0\) chain and lumped incident corner mass. The production driver sanitizes
that mass for the gated path by zeroing void/inactive-cell corners, preserving
strict finite non-negative checks for active-cell corners, and compacting the
PAVA input to positive-mass axis nodes only. All-dormant zero-mass axis nodes
are not active degrees of freedom. `hydro/axis_ale_rezone.{cuh,cu}` computes
the weighted lower-bound PAVA target \(Z^*\) and first off-axis ring
diagnostics. `src/hydro/ale_driver.cu` checks
`numerics.ale.axis_rezone_enabled` after the Lagrangian update inside the
multiblock CSR ALE path, gates it to
`mesh.topology_scheme="multiblock_half_butterfly_5block"` or
`mesh.topology_scheme="multiblock_half_butterfly_trifan_cap_5block"` with five
blocks,
applies the edge/altitude trigger at most once per hydro step, installs the
axis target into the existing conservative-reference target, and immediately
uses the existing CSR conservative remap. The module does not directly write
hydro state, force, velocity, pressure, or energy arrays; state changes come
only from the remap transaction. The feature is default-off and inert for
single-block and legacy three-block topologies.

The active-vertex contract is consumed throughout the cap runtime path:
CSR remap/GCL, reverse cell-node CSR rebuilds, Hydro2D corner mass, node mass,
\(\dot V\), pressure force and compatible force-work, CSW edge AV, subzonal
pressure/mass, CFL, ALE reference barriers, axis rezone, and the full-patch
driver all read `cell_nverts` rather than assuming four live corners. Triangle
anti-hourglass is intentionally a no-op because a true triangle has no
bilinear keystone/hourglass mode, while its three subzones still participate in
compatible bookkeeping. The cap apex \(O\) is fixed by PAVA and full-patch
target construction, and velocity/position projection pins it at
\((R,Z)=(0,0)\). HDF5 `/mesh/topology/v3` dispatch carries the mixed active
vertex counts additively; default/off decks retain legacy shapes and behavior.

`numerics.ale.force_rezone_every_n_steps>0` is a driver-only diagnostic policy
that passes `force_rezone=true` into the same 2D ALE dispatch on matching steps.
When `numerics.ale.multiblock_path_admissibility_enabled=true`, Hydro2D calls
`mesh/path_admissibility.cuh` after the Lagrangian corrector commit and before
geometry refresh.  The call is controlled by the path-admissibility toggle and
multiblock topology, not by `mesh.motion`.  A path failure returns a
`ReduceDtOnly` soft failure to the
coupling driver, whose existing full-step snapshot/restore path recomputes the
entire split step at the smaller dt.
`src/hydro/pole_angular_coarsen.{cuh,cu}` provides the default-off I1-B Q2
pole pilot overlay.  Hydro2D builds it under
`TENRYU_I1B_POLE_COARSEN_PILOT=1` for the topology-only path quotient pilot,
or under `TENRYU_I1B_POLE_MOTION_PILOT=1` for the coherent mesh-position
velocity pilot; ALE rezone callers use the default null overlay.  The helper
describes a POLAR_SHELL radial band and dyadic angular quotient macros.
`mesh/path_admissibility.cuh` validates candidate macro loops at old and trial
positions, escalates simple-loop failures to the next dyadic span, builds a
separate accepted-macro fine-cell mask, and evaluates the accepted macro
boundaries on host.  The coarsen pilot merges that mask with the existing
inactive-member mask for the CUDA cell scan; the motion pilot leaves fine-cell
paths enabled and uses the accepted macros only to reconstruct \(w_r,w_z\) on
the q-band plus the configured inward smoothstep transition rows.  Material
velocity, CCH force/work, `State` masks, and remap ownership are not modified
by the motion pilot.
Structured local repair entry points in `src/hydro/local_rezone.cu` retain an
early multiblock guard.

For `logical_mesh_2d="spherical_polar_halfplane"` with
`polar_center_treatment="tri_fan"`, TENRYU overlays a derived triangle-fan
center topology without changing the structured storage. The full
`(nr+1)*(nz+1)` node array remains allocated, `node_index(i,j)=i*(nz+1)+j`
is unchanged, HDF5/restart shapes are unchanged, and no new HDF5 datasets are
introduced. The `i=0` row is initialized at the origin and marked
`NODE_CENTER`. `Mesh::cell_nverts` is derived at mesh creation: center cells
use three active slots `{0,1,2}` and all other cells use four. This derived
vector is runtime-only. Stage 1 wires it into geometry, candidate
admissibility, and ALE post-rezone quality predicates. Stage 2 also threads
optional `cell_nverts` into hydro corner mass, node-mass gather,
ALE mass/kinetic/projection accounting, output node-mass recompute, and energy
accounting. Stage 3 threads optional `cell_nverts` and `node_flags` into the
2D_RZ conservative reference remap for tri-aware centroids, swept volumes, and
post-remap `NODE_CENTER` velocity pinning. Null device pointers keep
rectangular and non-polar quad paths on the legacy code path.

Spherical-polar half-plane meshes also mark theta-pole boundary nodes with
`NODE_POLE_AXIS` for `i>0` (excluding the tri_fan `NODE_CENTER` origin row). Hydro and
ALE velocity-projection kernels upload `node_flags` when either `NODE_CENTER`
or `NODE_POLE_AXIS` is present. At `NODE_POLE_AXIS` nodes, reflect/state-supply
pole constraints zero only the cylindrical radial component; fixed pole
constraints zero both components. `NODE_CENTER` remains the final constraint
and pins both velocity components and the origin position.

Hydro receives `MeshTopology::node_flags` as an optional device pointer when
`NODE_CENTER` or `NODE_POLE_AXIS` nodes exist. Stage 2 pins center nodes in the acceleration,
velocity, and position-update kernels: acceleration is zero, predictor and
corrector velocity writes are zero, and predictor/corrector position commits
write exactly `(R,Z)=(0,0)`. The force and dVdt kernels still consume `Svec`;
tri_fan slot 3 is zero by the geometry contract, while active apex slot 0 is
nonzero and requires the pin.

Config validation rejects tri_fan with anti-hourglass, HLLC z-flux, precise
RZ geometric CFL, or `total_energy_remap_2d_rz=true` until their center-cell
designs are implemented. Stage 3 supports first-order conservative reference
remap for tri_fan; center-aware rezone policy remains Stage 4.

#### 4.2.2 Mesh 構造体

`Mesh` は1D_SPHと2D_RZを統一的に扱う計算格子データ構造であり、
`MeshTopology`（§4.2.1）とジオメトリデータ（座標・体積・面積）を統合する。

```cpp
// 計算格子（1D_SPH / 2D_RZ 共通）
// NUMERICS §3.1（1D）、§3.2（2D RZ）準拠
struct Mesh {
    MeshTopology topo;              // 論理トポロジ（§4.2.1）

    // --- ノード座標（deviceメモリ、BORROWED from State.x_r/x_z）---
    // 1D_SPH：node_r[i]（i = 0..nr）、node_z 未使用
    // 2D_RZ ：row-major node_r[i*(nz+1)+j], node_z[i*(nz+1)+j]
    double* node_r;                 // BORROWED [n_nodes] R座標 [cm]（所有権: State.x_r）
    double* node_z;                 // BORROWED [n_nodes] Z座標 [cm]（所有権: State.x_z、1Dでは nullptr）

    // --- ノード速度（deviceメモリ、BORROWED from State.v_r/v_z）---
    double* node_vr;                // BORROWED [n_nodes] R方向速度 [cm/s]（所有権: State.v_r）
    double* node_vz;                // BORROWED [n_nodes] Z方向速度 [cm/s]（所有権: State.v_z、1Dでは nullptr）

    // --- セル量（deviceメモリ、OWNED: Mesh が確保・解放）---
    double* cell_vol;               // OWNED [n_cells] セル体積 [cm³]
    double* cell_centroid_r;        // OWNED [n_cells] セル重心R座標 [cm]
    DeviceArray<double> cell_centroid_r_device; // device mirror for hydro predicates
    double* cell_centroid_z;        // OWNED [n_cells] セル重心Z座標 [cm]（1Dでは nullptr）

    // --- セル↔ノード接続（構造格子のため暗黙的）---
    // cell(i,j) の4頂点：node(i,j), node(i+1,j), node(i+1,j+1), node(i,j+1)
    // 1D_SPH：cell(i) の2端点：node(i), node(i+1)
    // spherical_polar tri_fan: cell_nverts[c]==3 for i=0 center cells and
    // slots {0,1,2} are active; slot 3 remains allocated but inactive.
    std::vector<uint8_t> cell_nverts; // derived runtime topology, not persisted

    // --- ジオメトリ次元 ---
    int dim;                        // 1 = 1D_SPH, 2 = 2D_RZ

    // --- ジオメトリ再計算 ---
    // Lagrangian移動またはALE rezone後に体積・重心・面積を再計算
    // cell_vol, cell_centroid_r, cell_centroid_r_device, cell_centroid_z を更新する
    // recompute_geometry_checked() は host-side MeshGeometryResult を返し、
    // 既存 recompute_geometry() は同じ検査を HardAssert policy で呼ぶ互換wrapperである。
    // **同期契約**：recompute_geometry() 完了直後に State.vol = Mesh.cell_vol の
    //   コピーを実行すること（Coupling::step 内で実施）。State.vol は物理カーネル
    //   （source_injection, EOS, conduction等）が参照する正規の体積フィールドである。
    //   Mesh.cell_vol はジオメトリ計算の一次ソースとして扱い、State.vol との
    //   同期は Coupling レイヤーの責務とする。
    void recompute_geometry(cudaStream_t stream);  // NUMERICS §3.2.2, §3.2.3

    // 1D_SPH hydro fast path: State.vol の device buffer へ直接体積を書き、
    // Mesh の host geometry cache は更新しない。host consumer の前には
    // sync_device_geometry_to_host() または recompute_geometry() による同期を行う。
    void recompute_geometry_device_only(double* state_vol);
    void sync_device_geometry_to_host(const double* state_vol);

    // --- セル体積計算（device function）---
    // 1D_SPH: V = (4π/3)(r_{i+1}³ - r_i³)（NUMERICS §3.1.2）
    // 2D_RZ : V = 2π × 四辺形面積 × R_centroid（回転体積）（NUMERICS §3.2.2）
    // 戻り値：セル体積 [cm³]
    __device__ static double compute_cell_volume(
        int dim, int i, int j,
        const double* __restrict__ node_r,
        const double* __restrict__ node_z,
        int nz_plus1
    );
};
```

> **1D/2Dの統一**：`dim` フラグで分岐する。1D_SPHでは `node_z`, `node_vz`, `cell_centroid_z`
> は `nullptr` であり、Z方向の処理はスキップされる。
> 構造格子のため接続情報は `(i,j)` インデックス演算で暗黙的に解決され、
> 明示的な接続配列は不要である。

> **Mesh/State エイリアス**：`Mesh.node_r`/`node_z` と `State.x_r`/`x_z` は同一デバイスメモリのエイリアス。所有権は State。Mesh は借用ポインタ（non-owning）。

#### 4.2.3 Mesh::Search — ハッシュグリッドセル探索

退役：モンテカルロ輻射の光子粒子のセル同定（NUMERICS §9）のためのハッシュグリッド。粒子の再同定（`particle_reid.cu`）とともに
2026-09-29 に退役し、記述を `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した。

#### 4.2.4 Mesh::Remap — ALE rezone後の保存的転写

```cpp
// ALE rezone後の保存的remap（NUMERICS §3.3.4）
void remap_conservative_fields(
    const Mesh& old_mesh,           // rezone前のメッシュ
    const Mesh& new_mesh,           // rezone後のメッシュ
    CellField* fields,              // remap対象フィールド配列（§5.1 CellField エイリアス）
    int n_fields,                   // フィールド数
    cudaStream_t stream
);
```

Shock-frame 2D_RZ reference remap is exposed separately as
`hydro::ale::ale_remap_2d_rz` (`src/hydro/ale_remap_2d_rz.hpp`,
`src/hydro/ale_remap_2d_rz.cu`).  It runs immediately after a Lagrangian
Hydro2D update when `Numerics.ale.conservative_remap_enabled=true`, maps from
the post-Lagrange mesh to `State.x_*_reference`, and then replaces
`State.x_r/x_z` and `State.vol` with the configured reference geometry. For
multiblock states, `Numerics.ale.multiblock_scaled_reference_enabled=true`
refreshes that geometry to the scaled γ-MVP target immediately before CSR
remap. `Numerics.ale.multiblock_lagrangian_bulk_center_patch_reference_enabled=true`
takes precedence over both remap-path installers: the scheduled CSR ALE driver
has already installed a Lagrangian-bulk center/quality-patch target, and the
remap path preserves it. For this opt-in path the driver forces the protected
CSR swept-volume convention in the temporary remap config, activating the CSR
outgoing-mass scale without changing default legacy remap decks. Otherwise
`Numerics.ale.multiblock_differential_reference_enabled=true` takes precedence
over the scaled γ-MVP installer for multiblock 2D_RZ and refreshes the target
to the Lagrangian-close differential reference instead. Otherwise the scaled or
IC reference geometry is used. This path is
default-off and bypasses the scheduled Winslow ALE block for that step.  Its
state-supply z-face boundary flux helper uses the adjacent interior cell
velocity for the signed face speed and selects supply vs interior donor by
upwind direction before the first- or second-order remap kernel consumes the
per-column boundary flux arrays.

For `polar_center_treatment="tri_fan"`, this remap uploads optional
`cell_nverts` for tri cells and uploads `node_flags` whenever center or
pole-axis velocity constraints are present. The uploaded topology
selects three-node center-cell centroids, applies the polar signed-volume
orientation to swept volumes, falls back to first-order donor flux on faces
whose donor pair includes a tri cell, re-pins `NODE_CENTER` node velocities
to zero after projection, and applies `NODE_POLE_AXIS` radial-only pole
projection constraints. `total_energy_remap_2d_rz=true` remains guarded off
for the structured `polar_center_treatment="tri_fan"` single-block path; the
supported CSR total-energy branch is the multiblock
`multiblock_half_butterfly_trifan_cap_5block` path described in §4.2.1 above.

The experimental I1 z-HLLC path is exposed as
`hydro::apply_hllc_z_flux_2d_rz` (`src/hydro/hllc_z_flux_2d_rz.hpp`,
`src/hydro/hllc_z_flux_2d_rz.cu`).  It is selected only by
`Numerics.hydro.hllc_z_flux_2d_rz=true` and requires
`total_energy_remap_2d_rz=true`.  In that mode the Hydro2D step uses a fixed
reference mesh Eulerian z-face HLLC update for quasi-1D shocks and the driver
skips the conservative reference remap for that hydro substep, preventing double
z transport.  The path stores authoritative cell-centered z momentum in
`State.hllc_mom_z_cell`; projected nodal `v_z` is compatibility/output state and
is not used to reconstruct the next HLLC primitive state.
The I1 closure evidence and scope caveats for this experimental path are in
`docs/validation/2d_rz/I1/closure_summary.md`.

**remap対象フィールド**（保存量 = 密度 × 体積）：
- `ρ`（質量密度）→ 質量 `m = ρV`
- `ρu_r`, `ρu_z`（運動量密度）
- `ρe_e`, `ρe_i`（内部エネルギー密度）
- `volFrac[mat]`（材料体積分率）— 保存的 remap: `f_new[c] = Σ_{c'∈overlap} f_old[c'] × V_overlap(c,c') / V_new[c]`。
  Σ_mat f_new[c,mat] = 1 の正規化を remap 後に強制し、丸め誤差を修正する。
  単一材料セル（f=1.0）は remap 不要（最適化対象）

**remap後の reclosure**（CUDA_KERNELS §9 Phase 5 準拠）：
remap は保存量（ρe_e, ρe_i, ρu_r, ρu_z 等）を転写するが、原始変数（Te, Ti, Pe, Pi, c_s）および節点速度は更新しない。
remap 直後に以下の reclosure シーケンスを実行する：
1. `compute_cell_geometry`（H7）：新メッシュの体積・面積・特性長を再計算
2. `compute_density`（H8）：mass / V_new → ρ_new
3. `project_cell_velocity_to_nodes`（A4）：セル中心速度 u_r,u_z → 節点速度 v_r,v_z（NUMERICS §3.3.4 質量重み投影）
4. `eos_inverse`（H14）：ρ_new, ee → Te; ρ_new, ei → Ti
5. `eos_forward`（H13）：ρ_new, Te, Ti → Pe, Pi, Cv_e, Cv_i
6. `compute_sound_speed`（H15）：Pe, Pi, ρ_new → c_s
7. `floor_clamp`（U2）：安全策適用

**データ所有**
- Node：`x_r, x_z`（1Dは `x_r` のみ）、`v_r, v_z`、`node_flags`
- Cell：centroid（キャッシュ）、volume、surface metrics
- Face：暗黙的（on-the-fly計算）

---


### 4.3 materials/
**責務**：EOS/opacity/導電率/緩和/Zbar など物性を提供（状態→係数）

- `Materials::EOS`：P_e,P_i,e_e,e_i,Cv など（SESAME/IONMIX/ideal gas）
- `Materials::Opacity`：
  - LTE（`"ionmix"`）：入力 κ^PA_g, κ_R,g [cm²/g] → 出力 σ_a,g = ρ κ^PA_g, σ_R,g = ρ κ_R,g [1/cm]
  - Non-LTE（`"table_nlte"`）：入力 κ^PA_g, κ^PE_g, κ_R,g [cm²/g]（IONMIX 3種分離）→ 出力 σ^PA_g, σ^PE_g, σ_R,g [1/cm]
  - **Planck absorption/emission/Rosseland の使い分けはNUMERICSと一致させる**
- `Materials::Transport`：κ_e（Spitzer+flux limiter）、ν_ei、Zbar
- `Materials::Mixture`：多材料セルの混合則（EOS混合は NUMERICS §1.1.5 (c)、伝導/ソース結合のセル実効量は NUMERICS §1.1.5a、SPECIFICATION §6.4.3）
- `Materials::Tables`：SESAME/IONMIXロード、単位変換、範囲外clamp、診断
- `Materials::DeviceEOSTable`（`src/materials/eos_device_table.hpp`, `eos_device_table.cuh`, `eos_device_table.cu`）：
  table EOS data/view upload and device-side interpolation/inversion helpers.
  `eos_device_table_warp.cuh` (2026-09-25) holds warp-cooperative forms of the
  inversion (`device_inverse_reclose_warp`, `..._with_high_t_tail_warp`,
  `device_eos_T_from_e_monotone_warp`, `cold_inverse_Te_warp`): every lane of a
  warp calls them with the same arguments, the bisections over table rows and
  temperature nodes evaluate the next five levels' 31 midpoints in parallel and
  descend with the sequential loop's decisions, so the decisions are the
  sequential ones and the values agree to rounding (the compiler may contract a
  multiply-add differently; `tests/materials/test_eos_inverse_warp.cu`). The 1D 2T
  closure kernel (`enforce_2t_closure_split_kernel`, hydro_1d.cu) runs the ion and
  electron inversions of a cell on two such warps
- `Materials::ZbarDeviceContext` (`src/materials/zbar_device.hpp`, `zbar_device.cu`,
  `zbar_math.hpp`): driver-owned cache for immutable per-material ionization tables,
  material descriptors, and the host-owned void mask. The 1D updater reads resident
  rho/Te/volFrac and writes resident zbar; tabular interpolation and Thomas–Fermi
  evaluation preserve ordered material mixing. Tabular clamp warnings share the
  host reader's ordered warning budget through a bounded device summary. The
  fixed-ionization path remains a no-op, and `TENRYU_ZBAR_HOST=1` selects the
  original per-step host update for bisection. Non-TMAT Thomas–Fermi decks retain
  that host update because their strict bitwise gate rejects CUDA transcendental
  rounding; the device TF dispatch requires every non-void material to use TMAT.
- `Materials::EOSRhoETable`（`src/materials/eos_rho_e_table.hpp`, `eos_rho_e_table.cpp`, `eos_rho_e_device.hpp`, `eos_rho_e_device.cuh`, `eos_rho_e_device.cu`）：
  hydro-only total-EOS direct table on either \((\log\rho,\log e)\) or \((\rho,e)\). CPU
  initialization resamples raw `total` `P(\rho,T), e(\rho,T)` into `P(\rho,e), T(\rho,e)` on
  the original density grid and a 200-point energy grid (log-uniform by default; linear when
  `Numerics.hydro.rho_e_linear_grid=True`), precomputes natural-cubic second-derivative fields
  \((P_{xx},P_{yy},P_{xxyy})\), \((T_{xx},T_{yy},T_{xxyy})\), and then device evaluators expose
  total \(P,T,c_v,c_s\) from the resulting C² tensor-product spline without runtime total-EOS
  `e→T` inversion.
- `Materials::HelmholtzSpline`（`src/materials/helmholtz_spline.hpp`, `helmholtz_spline.cpp`, `helmholtz_spline_device.hpp`, `helmholtz_spline_device.cuh`, `helmholtz_spline_device.cu`）：
  hydro-only smooth EOS surrogate. Despite the compatibility name, the implementation builds a shape-preserving C¹ bicubic Hermite surrogate for the raw `total` EOS `P(\log\rho,\log T)` and `e(\log\rho,\log T)` on CPU at initialization, stores node data for both fields \((P, P_x, P_y, P_{xy})\), \((e, e_x, e_y, e_{xy})\), and exposes device evaluators for \(T(\rho,e), P, e, c_v, c_s\).
- `Materials::HelmholtzJet`（`src/materials/helmholtz_jet.hpp`, `helmholtz_jet.cpp`, `helmholtz_jet_device.hpp`, `helmholtz_jet_device.cuh`, `helmholtz_jet_device.cu`）：
  hydro-only local projected-jet surrogate for raw `total` EOS tables. CPU initialization builds nodal jets
  \((\phi,\phi_x,\phi_y,\phi_{xx},\phi_{yy},\phi_{xy})\) on \((\log\rho,\log T)\), applies local positivity-preserving clamps for \(c_v\) and \(\partial P/\partial\rho|_T\), then packs per-cell tensor-product biquintic Hermite coefficients. Device evaluators expose \(T(\rho,e), P, e, c_v, c_s\).
- `Materials::HelmholtzBSpline`（`src/materials/helmholtz_bspline.hpp`, `helmholtz_bspline.cpp`）：
  CPU-only Phase 1 fitter for a future thermodynamically consistent hydro EOS backend. It solves a weighted least-squares projection of the raw `total` table onto a quintic tensor-product B-spline representation of \(\phi(\ln\rho,\ln T)=F/T\) using reduced knot density, monotone-Hermite reference fields for \(c_v\) and \(\partial P/\partial\rho|_T\), and an active-set KKT solve that enforces positivity constraints at selected data nodes. It logs nodal reconstruction / positivity / \(T(\rho,e)\) inversion diagnostics and is not yet wired into runtime hydro kernels.
- `Materials::IONMIXBinaryEOS`（M17+）：IONMIX .cn4 バイナリからの EOS データロード（`load_ionmix_binary_eos`）。12 EOS ブロック（Zbar, P_i, P_e, e_i, e_e）を単位変換して `IonmixEOSData` に格納。`ionmix_eos_to_table_pair` で EOSTablePair（total + electron）に変換、`ionmix_eos_to_zbar_table` で `IonmixZbarTable` に変換。
- `Materials::NLTEOpacity`（M17）：IONMIX .cn4 の3種不透明度テーブル（κ^PA_g, κ^PE_g, κ_R,g）のロード・デバイス転送・補間（§4.3.2a）
- `Materials::OpacityDiagnostics`（`src/materials/opacity_diagnostics.hpp`, `opacity_diagnostics.cu`）：
  startup-only host diagnostic for hard-X-ray \(\kappa^{PA}\) audit logging. It reads
  the configured material opacity through the Materials readers and has no dependency on
  `radiation/`.

**禁止**：Hydro/Radiation/Laserがテーブルを直接読むこと（単位事故を防ぐ）。

#### 4.3.1 EOS GPU実行モデル

EOS 評価はセル毎の独立計算で、表 EOS は `__device__` 関数（`materials/eos_device_table.cuh`）を各演算子のカーネルが
インラインで呼ぶ。独立の EOS カーネル（旧設計の `eos_forward` / `eos_inverse`、CUDA_KERNELS H13/H14）は存在しない。

- **表の device 配置**：host の `EOSTable`（`materials/eos_table.hpp`）を `DeviceEOSTable::upload` が device へ写し、
  カーネルは `DeviceEOSTableView`（格子・\(P, e, c_v\) の表・任意の cold branch）を受け取る。多材料のデッキでは材料別の
  view 配列とセルの支配材料の添字（`CellEOSTableSelector`）でセルごとに表を選ぶ（NUMERICS §1.1.5）。
- **順方向**（\(T\to e, P, c_v\)）：`device_eos_energy` / `device_eos_pressure` / `device_eos_cv`（\((\log\rho,\log T)\) の双線形、
  表の端でクランプ。低密度の延長と energy_authoritative の高温側の延長は別関数）。
- **逆方向**（\(e\to T\)）：`device_inverse_reclose`。Newton 反復ではなく、温度の節点の上で根を含む区間を単調に探して
  区間内の線形の式で解く（反復なし）。床・表の端でのクランプと区間を作れない場合（bracket failure）は結果の構造体の
  フラグで返し、閉包が回数を数える（旧設計の `eos_newton_nonconverge` などのエラーフラグは無い）。
- **音速**：`device_eos_sound_speed`（補間関数の解析導関数からの \(\Gamma_1\) 形。\(c_s^2\le0\) なら \(\sqrt{(5/3)P/\rho}\)、
  NUMERICS §1.1.6）。
- **呼び出し元**：hydro の閉包（`enforce_1t/2t_closure_kernel`）・音速・エネルギー更新、伝導後の再閉包、FLD/S\(_N\) の物質更新、
  レーザー・燃焼・輻射の入射の閉包が、それぞれのカーネルの中で上の関数を呼ぶ。
- 理想気体EOS（NUMERICS §1.1.5 (a)）は解析式のためテーブル不要。

#### 4.3.2 EOSTable / OpacityTable 構造体

表は host の構造体で読み込み、device へ写して view でカーネルへ渡す。補間は `(log ρ, log T)` の双線形
（NUMERICS §1.1.5 (b)）。

```cpp
// materials/eos_table.hpp（host）。1 本の表 = 1 つの成分（ion / electron / total）
struct EOSTable {
    std::vector<double> rho_grid, T_grid_eV;       // 密度 [g/cm³]・温度 [eV] の格子（log 格子も保持）
    std::vector<double> P_table, e_table, cv_table; // T-major [j_T * n_rho + i_rho]。cv は e の差分（TMAT の cv 欄があればそれ）、下限 1e-3
    const ColdEquilibriumTable* cold = nullptr;     // 低温平衡構成則の補正（電子の表だけ。NUMERICS §1 (b)）
    void finalize();                                // 導出量の作成
};
struct EOSTableTriplet { EOSTable ion, electron, total; };   // 材料ごと（MatDef::eos_tables、shared_ptr）
// device: DeviceEOSTable::upload → DeviceEOSTableView（materials/eos_device_table.cuh、§4.3.1）
// 断熱指数 γ の表は無い（表 EOS の音速は補間関数の導関数から、NUMERICS §1.1.6）

// materials/ionmix_reader.hpp（host）/ ionmix_reader.cuh（device view）。群別の不透明度表（IONMIX・TMAT 共通）
struct IonmixOpacityData {
    std::vector<double> numdens_cm3, temps_eV;      // 密度軸はイオン数密度 n_i（TMAT も同じ）
    int ngroups; std::vector<double> bounds_eV;     // 群の数と境界（最初の表が Radiation.groups を決める、NUMERICS §6.7）
    std::vector<double> kappa_PA, kappa_PE, kappa_R; // [g][n_i][T]、質量不透明度 [cm²/g]
};
// 補間は log κ の双線形（隅が 1e-30 以下なら κ の線形へ切替）、表の範囲でクランプ（NUMERICS §1.1.5 (b)・§11.3）
```

> **EOS テーブル管理（2T モデル）**：
>
> **SESAME**：テーブル 301（total EOS）とテーブル 304（electron EOS）を**独立グリッド**で読む（NUMERICS §1.1.5 (b)）。
>   - `total`：301 のグリッド。`electron`：304 のグリッド（304 が無い材料は、材料の電離モデルの \(\bar Z\) で 301 を節点ごとに
>     \(\bar Z/(1+\bar Z)\) に分けた表 — `split_sesame_electron_table`、2026-09-29）
>   - `ion`：読み込み時に 301 のグリッド上で `build_sesame_ion_table` が作る（304 を実行時と同じ双線形・対数補間で 301 の節点へ
>     写して差を取る）。負の節点はクランプせずに保持し、数を 1 回警告する（旧設計の `max(0, …)` と `eos_ion_negative` は無い）
>   - 不透明度表 502/505 は `sesame_reader` が読めるが、実行時の不透明度には使わない（`opacity.model="sesame"` は `ConfigError`）
>
> **IONMIX**：IONMIX v4/v6 .cn4 バイナリファイルは `e_i, P_i`（イオン）と `e_e, P_e`（電子）を1ファイルに格納する。
> 密度軸はイオン数密度 \(n_i\) [cm⁻³]（SPECIFICATION §6.4.3 参照）。EOS 単位は J/g, J/cm³ で、cgs 変換（×10⁷）が必要。
> 読み込み時に \(\rho=n_iAm_p\) へ変換して `ion`・`electron`・`total` の 3 表を作る（比熱は数値微分）。
>
> **TMAT-H5**：`/eos/fields` の \(P_i, P_e, e_i, e_e, \bar Z\)（任意で `cv_i`・`cv_e`）から同じ 3 表を作る（§4.3.2b）。
>
> **共通**：Zbar テーブルは材料ごとの別の表（`MaterialsConfig::zbar_tables`）。混合セルは各セルの支配材料の表で閉じる
> （質量分率での合成はしない、NUMERICS §1.1.5 (c)）。

#### 4.3.2a NLTEOpacityTable 構造体（M17: IONMIX 3種不透明度 Non-LTE）

Non-LTE Phase 1（M17）で導入される不透明度テーブル構造体。
IONMIX4/6 `.cn4` バイナリファイルからロードし、デバイスメモリに配置する。
LTE の OpacityTable（§4.3.2）とは独立に管理される。

LTE の OpacityTable は κ^PA と κ_R の2種のみを格納するが、NLTEOpacityTable は
IONMIX4/6 の**3種不透明度**（κ^PA, κ^PE, κ_R）をすべて保持し、
\(\kappa^{PA} \neq \kappa^{PE}\) による非LTE放射輸送を可能にする。

```cpp
// Non-LTE 3種不透明度テーブル — IONMIX .cn4 由来、deviceグローバルメモリに配置
// NUMERICS §6.1.1、SPECIFICATION §6.4.3 準拠
struct NLTEOpacityTable {
    // --- 独立変数グリッド（log空間）---
    double* log_ni_grid;        // [n_dens] log10(n_i [cm⁻³])（イオン数密度）
    double* log_T_grid;         // [n_temp] log10(T_e [eV])
    int     n_dens;
    int     n_temp;
    int     n_groups;           // 群数 G

    // --- 材料パラメータ（密度変換用）---
    double  A;                  // 平均原子量 [amu]（ρ → n_i 変換: n_i = ρ/(A·m_p)）

    // --- 群別不透明度テーブル（3種分離）---
    // メモリレイアウト：group-major — kappa[g * n_dens * n_temp + id * n_temp + it]
    //   群が最外（outermost）、密度が中間、温度が最速（innermost）
    //   IONMIX バイナリのストリーム順序と一致
    double* kappa_R;            // [G × n_dens × n_temp] κ_R,g [cm²/g]（Rosseland）
    double* kappa_pa;           // [G × n_dens × n_temp] κ^PA_g [cm²/g]（Planck absorption）
    double* kappa_pe;           // [G × n_dens × n_temp] κ^PE_g [cm²/g]（Planck emission）

    // --- 群境界 ---
    double* bounds_eV;          // [G+1] 群境界 [eV]（単調増加）

    // --- テーブル範囲（clamp用）---
    double  ni_min, ni_max;     // [cm⁻³]（イオン数密度）
    double  T_min, T_max;       // [eV]

    // --- LTE/Non-LTE 判定フラグ ---
    bool    is_lte;             // |κ^PA - κ^PE| / max(κ^PA, ε) ≤ 1e-6 for all entries

    // --- device function: log(n_i)-log(T) bilinear 補間 ---
    // ρ [g/cm³] → n_i = ρ/(A·m_p) 変換はホスト側 or カーネル内で実施
    __device__ double interp_kappa_R(int g, double log_ni, double log_T) const;
    __device__ double interp_kappa_pa(int g, double log_ni, double log_T) const;
    __device__ double interp_kappa_pe(int g, double log_ni, double log_T) const;
};
```

> **設計方針**：
> - NLTEOpacityTable は材料種毎に1インスタンス（OpacityTable と同様）
> - `opacity.model = "table_nlte"` の場合のみロードされる
> - IONMIX .cn4 のヘッダ・EOS ブロック・不透明度テーブルを順次読み込み。
>   EOS ブロックは既存の IONMIX EOS ローダと共有し、不透明度テーブルのみ NLTEOpacityTable が管理する
> - **密度軸変換**: IONMIX の密度軸はイオン数密度 \(n_i\) [cm⁻³]。
>   TENRYU 内部の質量密度 \(\rho\) [g/cm³] からの変換: \(n_i = \rho / (A \cdot m_p)\)
> - **κ^PE 不在時の LTE フォールバック**: IONMIX ファイルに Planck emission テーブル（3番目）が
>   存在しない場合、\(\kappa^{PE} = \kappa^{PA}\) と仮定する（WARNING 出力、`is_lte = true` に設定）
> - 負の κ^PA, κ^PE, κ_R はロード時の全要素検査で拒否する（クランプしない — `ionmix_reader.cpp` がエラーで止める）
> - Fortran レコードマーカーの整合性チェックを実施（不整合時は `ConfigError`）
> - **η_g はテーブルに格納しない**: η_g = σ^PE_g × c × a_eV × T^4 × b_g(T) として
>   実行時に構成する（1D FLD の不透明度・放出の評価、NUMERICS §6.7。`CellRadiationCoeffs` は退役した IMC の構造体）

#### 4.3.2b TMAT-H5 Material Table

TMAT-H5 は IONMIX4/SESAME の代替として導入する self-describing HDF5 物性テーブル形式であり、
EOS と多群不透明度を 1 ファイルで保持できる。

- **Format identifier**：`tenryu.material_table.hdf5`
- **ファイル拡張子**：`.tmat.h5`
- **単位系**：`cgs_eV`（TENRYU ネイティブ。ロード時の単位変換は行わない）
- **密度軸**：`n_i` [cm⁻³]（IONMIX と同じイオン数密度軸）
- **グリッド**：EOS と opacity は独立グリッドを許容（共有を要求しない）
- **内部変換**：EOS/Zbar は `A_amu` を用いて \(\rho = n_i A m_p\) [g/cm³] に変換して保持し、
  opacity は `n_i` 軸のまま `IonmixOpacityData` に受け渡す

**HDF5 グループ階層（v1.0）**：
- `/`：`format_id`, `schema_version`, `units_system` などのルート属性
- `/material`：組成情報（`Z`, `A_amu`, `mass_fraction`, `Abar_ion_amu`）
- `/eos`：EOS 軸・場（`/grid/ni_cm3`, `/grid/temperature_eV`, `/fields/*`）
- `/opacity`：不透明度軸・場（`/grid/ni_cm3`, `/grid/group_bounds_eV`, `/fields/kappa_*`）
- `/provenance`：生成元・変換履歴（推奨）
- `/extensions`：将来拡張（任意）

**メモリレイアウト**（C row-major）：
- EOS：`[D,T]`（index = `d * nT + t`、`T` fastest）
- Opacity：`[G,D,T]`（index = `g * nD * nT + d * nT + t`、`T` fastest）

**Reader API**（`Materials::Tables`）：
- `load_tmat()`：`.tmat.h5` を読み込み `TmatFile` を構築
- `tmat_eos_to_table_pair()`：`TmatFile.eos` を `EOSTablePair` に変換（`[D,T] -> [T,D]`）
- `tmat_to_ionmix_opacity()`：`TmatFile.opacity` を `IonmixOpacityData` 互換データへ変換（`[G,D,T]` は保持、`ni_cm3` はそのまま転写）

**データフロー**：

```text
material.tmat.h5
   -> load_tmat()
   -> TmatFile {material, eos, opacity}
      -> tmat_eos_to_table_pair() -> EOSTablePair
      -> tmat_to_ionmix_opacity() -> IonmixOpacityData
```

| 項目 | IONMIX4 (`.cn4`) | SESAME (xSESAME ASCII) | TMAT-H5 (`.tmat.h5`) |
|---|---|---|---|
| 形式 | Fortran unformatted binary | 固定幅 ASCII | HDF5 |
| Self-describing metadata | なし | なし | あり |
| 密度軸 | `n_i` [cm⁻³] | `rho` [g/cm³] | `n_i` [cm⁻³] |
| 多群 opacity | あり | grey のみ | あり |
| EOS/opacity 独立グリッド | 不可 | 実質不可 | 可能 |
| TENRYUロード時単位変換 | 必要 | 必要 | 不要（`cgs_eV`） |

#### 4.3.3 Mixture mixing rules

多材料セルの不透明度の混合則（`Materials.mixture.opacity_mix_rule`、NUMERICS §1.1.5 (c)）：
- `linear_mass`（既定）：Planck・Rosseland とも \(\kappa_{mix} = \sum_m Y_m \kappa_m\)（\(Y_m\) = 質量分率、未追跡なら体積分率）
- `harmonic_mass_R`：Planck は線形、Rosseland は調和 \(1/\kappa_{R,mix} = \sum_m Y_m / \kappa_{R,m}\)
- `max`：両方とも存在する材料の最大値
- 表の材料は各材料の部分密度で評価して合成する（1D FLD の `eval_opacity_multimat_kernel`）

**Non-LTE 不透明度の混合則**（M17）：
- \(\kappa^{PA}_{mix,g} = \sum_m Y_m \kappa^{PA}_{m,g}\)（質量分率線形、吸収は additive）
- \(\kappa^{PE}_{mix,g} = \sum_m Y_m \kappa^{PE}_{m,g}\)（質量分率線形、ソース η_g に直結するため additive が物理的に正しい）
- \(\kappa_{R,mix,g}\)：既存の `harmonic_mass_R` を適用（Rosseland 平均は調和平均）
- 上の Non-LTE の記述は退役した IMC の設計で、`compute_opacities` カーネルと `CellRadiationCoeffs` は存在しない（1D FLD の非 LTE の
  混合は NUMERICS §1.1.5 (c) の「第 2 段」「第 3 段」）

EOS は混ぜない：1D の閉包は各セルを支配材料（体積分率が最大の非 void 材料）の EOS だけで閉じる（NUMERICS §1.1.5 (c)。
`Materials.mixture.eos_mix_rule` は受け付けるが無視する）。

伝導・ソース結合向けのセル実効量は非 void 材料の体積分率 \(f_m\)（その和で割り直す）で評価する（NUMERICS §1.1.5a）：
- \(A_{eff} = (\sum_m f_m/A_m)^{-1}\)（調和平均）
- \(\gamma_{eff} = \sum_m f_m \gamma_m\)（線形平均）
- \(n_e = \rho \bar{Z}/(A_{eff} m_p)\)
- \(c_{v,e} = \bar{Z}k_B/(A_{eff}m_p(\gamma_{eff}-1))\)、\(c_{v,i} = k_B/(A_{eff}m_p(\gamma_{eff}-1))\)
- `n_mat == 1` では \(A_{eff}=A_0,\ \gamma_{eff}=\gamma_0\) に退化し単一材料式と一致

#### 4.3.4 Materials トップレベル関数シグネチャ

Materials モジュールは「ステップ関数」を持たず、他モジュールのカーネル内で
`__device__` 関数として呼び出される（§4.3.1）。ホスト側APIは初期化を提供する。

```cpp
// Materials 初期化：テーブルファイル読込 → deviceメモリへ転送
// SESAME: xSESAME ASCII パース → 単位変換 → eos_total + eos_e 構築
// IONMIX: IONMIX v4/v6 .cn4 バイナリパース → eos_e + eos_i 構築
//   opacity.model="table_nlte" 時は NLTEOpacityTable に3種不透明度をロード
void materials_init(
    const Config::MaterialsConfig& mat_cfg,
    EOSTable* eos_e_out,            // [n_materials] 電子EOS（device確保済み）
    EOSTable* eos_i_out,            // [n_materials] イオンEOS（IONMIX時）/ total EOS（SESAME時）
    OpacityTable* opacity_out,      // [n_materials] LTE不透明度テーブル（κ^PA, κ_R）
    NLTEOpacityTable* nlte_out,     // [n_materials] Non-LTE不透明度テーブル（κ^PA, κ^PE, κ_R）
                                    // model="table_nlte" の材料のみ非NULL、それ以外はNULL
    int n_groups                    // 放射群数
);
```

不透明度の評価は輻射ソルバーの中で行う（IMC 時代の前計算カーネル `compute_opacities`（U9）と `CellRadiationCoeffs` は
退役し、記述を `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した）：1D は `fld_1d_gpu.cu` の
`evaluate_fld_opacity_and_emission`・`sn_transport_1d_gpu.cu` の `evaluate_opacity_and_emission`（材料ごとの記述と部分密度で
混合する `eval_opacity_multimat_kernel`、表の非 LTE 係数は `nlte_coeffs.cu` の `compute_nlte_coefficients_cuda_with_pe`・
`_pure_sn`）、2D_RZ は `fld_2d_rz_gpu.cu` の `evaluate_fld_opacity_and_emission`・`sn_transport_2d_gpu.cu` の `evaluate_opacity`。

**MixingRule enum**（`const char*` はGPUカーネルに渡せないため enum を使用）：

```cpp
enum class MixingRule : uint8_t {
    LINEAR_MASS = 0,
    HARMONIC_MASS_R = 1,
    MAX = 2
};
```
`const char*` はデバイスメモリ上の文字列比較が非効率かつ非推奨のため、
Config パース時に文字列→enum変換を行う。

---

### 4.4 hydro/
**責務**：流体更新（運動量・位置・密度・内部エネルギー）、伝導、境界条件

- `Hydro::LagrangianStep`
  - 圧力 + 人工粘性でノード加速 → 速度/位置更新
- `Hydro::EnergyStep`
  - PdV、人工粘性加熱（既定はイオンへ）、e‑i緩和、外部源（laser/rad）を反映
- `Hydro::ALE`（NUMERICS §3.3）
  - rezoning（メッシュ品質、NUMERICS §3.3.3 Winslow equipotential）
  - remap（保存的：質量/運動量/エネルギー/材料分率、NUMERICS §3.3.4）Mesh::Remap を呼び出す
  - `ale_align_monitor.{hpp,cpp}`：Stage 0 の default-off host 診断。
    post-Lagrange の `rho`, `Pe+Pi`, 節点座標、`volFrac`, vacuum mask から
    WLS gradient、structure tensor、coherence、i-face alignment angle を計算し、
    `[ale-align-diag]` へ log-only 出力する。simulation state は変更しない
  - `ale_align_rezone.{hpp,cpp}`：Stage 1--2 の host-only planar/RZ prototype。
    Stage-0 の cell director/coherence を frozen monitor として再利用し、
    family-specific direction-control energy を Q1 \(2\times2\) symmetric
    quadrature で評価する。default planar mode は Stage-1 path を維持し、RZ
    mode は quadrature-point の exact \(2\pi r\) weight、cell-center RZ scale
    付き preconditioner、axis-node projection、および exact signed RZ cell
    volume gate/exposure を追加する。deterministic central-FD gradient、
    fixed-iteration damped update、initial boundary line 上の tangential slip、
    global fixed-ladder line search、および全 Q1 corner Jacobian の relative
    floor は両 mode で共通。reference/HR/remap は含まず、ALE driver からは
    呼ばれない
  - post-remap reclosure は `HydroEOSContext` を受け取り、table-backed electron/ion EOS では共有 inverse reclosure + low-density policy を用いる。context がない成分は従来の理想気体 reclosure に戻る。
  - The env-gated shell protected rezone probe/commit path is owned by
    `src/hydro/ale_driver.cu`.  `Hydro2D::lagrangian_step` returns
    `HydroStepResult::shell_subcycle_committed` when that path commits at
    `post_corrector_commit`; `Coupling::Driver` uses that flag only to apply
    accounting and skip the scheduled post-step multiblock ALE for that step.
- `Hydro::ConservationAudit` (`src/hydro/conservation_audit.{hpp,cu}`)
  - Default-off env-gated (`TENRYU_I1B_CONS_AUDIT`) hydro/ALE
    conservation ledger. It records raw and owner-accounted mass/energy at
    Lagrangian, macro aggregate, BBSW, compatible-work, and ALE remap stage
    boundaries, and reports macro ownership count defects without changing
    state when disabled.
- `Hydro::PerMaterialEOS` (`src/hydro/per_material_eos_accessors.cuh`,
  `src/hydro/per_material_eos_project.{cuh,cu}`)
  - The per-material EOS refresh module. Raw-view
    `TENRYU_DEVICE` accessors reuse the shared materials inverse-reclosure and
    sound-speed helpers. The projection kernel maps per-material
    thermodynamic state back to cell means (mass-weighted
    \(T_e,T_i,\bar Z,c_v\), reduced `ee`/`ei`,
    volume-weighted \(P_e,P_i\), max-over-present
    \(c_s\)). The host launcher owns temporary lazy-cache valid-flag device
    mirrors and local dispatch counters, and exposes cache invalidation
    helpers for the per-material mutation sites.
- `Hydro::ALE1D`（`src/hydro/ale_1d_driver.{cuh,cu}`, `ale_1d_types.cuh`, `ale_1d_sensor.{cuh,cu}`, `ale_1d_rezone.{cuh,cu}`, `ale_1d_remap.{cuh,cu}`, `ale_1d_velocity_project.{cuh,cu}`, `ale_1d_diagnostics.{cuh,cu}`）
  - 1D_SPH solution-adaptive ALE V3 の public API と skip-path diagnostics を保持する。現行版では GPU sensor が `compute_features` で feature list を構築し、`ale_1d_rezone` が monitor、common node mask、CPU equidistribution scratch candidate を構築し、`ale_1d_remap` が volume-coordinate MUSCL/minmod remap（first-order donor fallback と cosine protected-face taper 付き）を caller-owned scratch に書き、`ale_1d_velocity_project` が mass remap の受理済み面質量流束（`Ale1dRemapScratch::mass_flux`）と mass phi を使い、各セルの両端節点速度の組を比量として移流する half-index-shift 法（Benson 1992 §3.5.5）で節点速度 scratch を構築する（2026-09-23 に cell momentum remap + mass-weighted node projection から置換）。`ale_1d_diagnostics` が scratch diagnostics を評価し、`apply_ale_1d` は hard 許容誤差を満たした場合だけ二相 commit で state を更新する。
  - V3 1D ALE は opt-in / experimental。既定は `numerics.ale1d.enabled=False` で、通常の GXII short-pulse は pure Lagrangian を使う。
  - runtime scope は 1D_SPH + deterministic radiation (FLD/S_N) のみ（モンテカルロ輻射は 2026-09-29 にビルドから外した）。
- `Hydro::PLIC` (`src/hydro/plic_geometry.{cuh,cu}`,
  `src/hydro/plic_normal.{cuh,cu}`, `src/hydro/plic_fast_path.{cuh,cu}`,
  `src/hydro/plic_remap.{cuh,cu}`)
  - The PLIC material-interface reconstruction core.  Geometry helpers
    provide RZ Pappus polygon clipping and bracketed alpha bisection; normal
    helpers provide Youngs-seeded LVIRA and deterministic degradation
    fallbacks; fast-path helpers build host-side interface masks.
  - Default-disabled PLIC material-volume remap entry points were added
    in `plic_remap`.  The ALE driver branches once at the material
    volume-fraction remap loop boundary: rho, momentum, electron/ion internal
    energy, diagnostic kinetic energy, and radiation energy stay on the scalar
    remapper, while `f_m V` may use PLIC face material-volume fluxes.
  - Runtime scratch is owned by `core::State` as non-checkpointed buffers:
    interface/active masks, reconstruction validity, normals, alpha,
    interface centroids, face material-volume fluxes, and per-cell residuals.
    The buffers are allocated lazily only when `numerics.plic.enabled` is true,
    `in_run_disabled` is false, and sticky fallback is not engaged.
  - The PLIC material-volume remap is serial-only.  `part.n_ranks > 1` with PLIC enabled is rejected at
    ALE entry; MPI validation is deferred to future work.  PLIC scalar fallback
    and class-(c) ALE escape valves report independently.
- 初期の ALE 整理で削除された旧 1D ALE path は復活させない。2D_RZ の `Hydro::ALE` / Winslow path は2D専用実装として維持する。
  - 2D_RZ ALE backtracking first evaluates the legacy 4-Gauss Jacobian
    `post_tangle` gate, then, when
    `Numerics.ale.corner_jacobian_post_tangle_enabled=True`, a parallel
    active-cell signed corner-J gate. Corner-J failures are reported separately
    in ALE backtrack telemetry and reject the candidate before remap-damage,
    remap-admissibility, or predictive-acceptance gates.
  - If all backtrack candidates fail the corner-J gate and
    `Numerics.ale.local_boundary_repair_enabled=True`, the driver may invoke a
    narrow boundary fallback that moves one non-corner top/bottom or outer-r
    boundary node along its boundary line. Accepted local repairs re-enter the
    normal ALE acceptance gates before remap.
  - If the local boundary interval is empty and
    `Numerics.ale.multi_node_boundary_repair_enabled=True`, the same emergency
    path may expand to a capped 1-ring coordinated repair solved by a small
    host-side linearized half-space projection. The candidate is globally
    revalidated before it can re-enter the normal ALE acceptance gates.
  - If the capped multi-node repair is infeasible and
    `Numerics.ale.emergency_cell_deactivation_enabled=True`, the driver may
    roll the mesh back, transfer the failing cell's mass/internal energy to a
    face-neighbor active cell, mark the failing cell void/inactive, and return
    without an ALE remap. This fallback is opt-in and globally revalidates
    active-cell Gauss/corner-J admissibility before it commits.
  - Driver retry may pass a host-only `AleRequest` (`src/hydro/ale_mode.hpp`)
    into `apply_ale_with_request` without changing the legacy `apply_ale`
    call surface.  The request dispatches local repair operators in
    `src/hydro/local_rezone.{cuh,cu}` and `src/hydro/cd_local_rezone.{cuh,cu}`;
    when no request is present, the scheduled 2D ALE path is unchanged.
- `Hydro::CFL`
  - dt_hydro の計算（NUMERICS §3.1.9, §3.2.13）
  - `src/hydro/rz_geometric_cfl.cu` provides the default-off 2D_RZ predictive
    geometric hydro CFL used by `compute_dt_hydro` to clamp the Lagrangian
    half-step before projected RZ cell volume drops below the configured
    fraction of the current volume or, when enabled, a fixed fraction of
    `State.cell_vol_initial`.  `compute_dt_hydro` can optionally provide a
    force-predicted \(\mathbf{u}^{1/2}\) instead of the current state velocity.
- `Hydro::CornerJacobianQuality`（`src/hydro/corner_jacobian_quality.{cuh,cu}`）
  - 2D_RZ cell corner の signed Jacobian を評価し、default-off の pre-hydro ALE trigger と Hydro2D pre-commit diagnostic に共有 predicate を提供する（NUMERICS §3.2.13）
  - G1 multiblock path guard は `Mesh::ReferenceFlatPath` の immutable typed
    mask を受け、regular corner の既存 strict quadratic を維持しながら
    reference-flat corner とその cell embedding を host exact predicate で合成する。
  - retry active mesh repair 用に、current corner-J の same-cell balance \(q_{bal}=\min J/\max J\) を GPU reduction で評価し、非正・非有限 corner と 100:1 既定 imbalance を dt-independent に検出する（NUMERICS §3.2.13b）
- `Hydro::Ring7SeamQuotientRemap` (`src/hydro/ring7_seam_quotient_remap.{cuh,cu}`)
  - I1-B `Ring7OuterSeamQuotientRemap` の default-off module. Increment 1 defines RZ face/swept-volume primitives, fixed-order compensated packet accumulation, ring-7 seam patch discovery, and diagnostic-only q/eta scanning.
  - Increment 2 adds a Hydro2D StepStart seam transaction under `Numerics.hydro.ring7_quotient_enabled` (or `TENRYU_I1B_RING7_QUOTIENT`). It fires only from a one-shot request armed proactively by an accepted-step multiblock path-margin guard band or reactively when the driver retries a production `mesh_quality_rz_volume` or `multiblock_path_admissibility` failure on the Ring7 seam patch. The no-request path returns before seam allocation, metric evaluation, dt reduction, or state mutation. The reactive path restores the full-step retry snapshot, carries the failing cell in a single-use runtime request, and reruns Hydro2D at the same `dt`; driven pole-cap boundary failures are routed away from the seam remap by the existing candidate-cell predicate.
  - The transaction symmetry-pairs tangential ring-7 seam-node motion, checks exact path admissibility through `Mesh::PathAdmissibility`, and checks B_V through the production multiblock RZ-volume root kernel.
  - Increment 3a does not call the global ALE remap path. The accepted target is consumed only by a dedicated seam packet-geometry scaffold that promotes every cell whose multiblock `cell_node_csr` node list references any moved seam node, enumerates internal/boundary faces from `MultiBlockTopology::unique_internal_faces`, splits signed RZ swept volumes by the existing `State.corner_volume` edge-corner weights, and logs `[ring7_seam_packet]` geometry/patch/core-isolation diagnostics.
  - Increment 3b applies rebuilt geometric submove packet ledgers to host working copies, validates donor-corner positivity, whole-domain mass/total-energy and paired-momentum deltas, and central-core hash/sum isolation, then commits `State.x_r/x_z`, conservative cell fields, node velocities, transported corner masses, and refreshed mesh geometry only after all gates pass. It still does not call the global ALE remap/reference-volume/node-kinematics machinery.
  - Increment 4a keeps the seam packet remap as an interior-seam repair and adds a diagnostic-only driven-pole cap oracle for cell-608-class domain-boundary RZ-volume failures. Hydro2D retains the exact failed production limiter velocity field, the driver requests a single StepStart oracle on the restored retry, and the oracle sweeps small cap sizes and normal-velocity taper options through `compute_mesh_quality_dt_limit()` without committing coordinates or state.
  - Increment 4b commits the selected pole-cap ALE candidate through a separate host packet transaction from the retained failed Lagrangian geometry to the cap mesh geometry. The relative target equals the retained Lagrangian geometry outside the cap, and the coordinate commit writes only cap nodes so the normal hydro step still owns the off-cap Lagrangian motion. The transaction uses deterministic internal-face packets, a driven-boundary volume ledger, donor-corner positivity and central-core/conservation gates, then marks a one-shot validation so the next Hydro2D production limiter call logs the recomputed retry `u_half` pass/fail at the requested pole cell.
- `Mesh::PathAdmissibility` (`src/mesh/path_admissibility.cuh`)
  - Multiblock CSR quad paths \(X^n\rightarrow X^{n+1}\) の signed corner-J,
    2x2 Gauss-J, and polygon-area quadratics are evaluated on CUDA before
    Hydro2D accepts the Lagrangian corrector.  The module is header-only so the
    Hydro2D caller can launch the check without adding a mesh library source.
    Default-off anatomy (`TENRYU_I1B_PATH_ADMIS_ANATOMY`) annotates the winning
    rejection metric kind and block-local cell coordinates.  The separate
    default-off hardening gate (`TENRYU_I1B_PATH_PREDICATE_HARDEN`) switches the
    live predicate to `cell_orientation_sign` orientation, scaled/canonical loop
    guards, and repair-only lambda-zero old-geometry classification; unset uses
    the legacy live predicate path.  The optional `Hydro::PoleAngularCoarsen`
    overlay is consumed here as accepted quotient macro boundaries plus a
    separate fine-child skip mask when `skip_fine_child_paths=true`; the motion
    pilot sets that flag false so covered fine cells are still scanned.
- `Hydro::PoleAngularCoarsen` (`src/hydro/pole_angular_coarsen.{cuh,cu}`)
  - Default-off I1-B Q2 geometry/path pilot.  It constructs the POLAR_SHELL
    radial-band metadata and true quotient macro perimeters for Hydro2D path
    checks.  Macro angular arcs contain only interval endpoints; radial sides
    retain intermediate nodes when the selected q-band spans multiple rows.
    Under `TENRYU_I1B_POLE_MOTION_PILOT=1`, Hydro2D also asks the helper to
    reconstruct candidate mesh position velocities from accepted macro endpoint
    motion for the q-band node rows plus a default-four-row inward smoothstep
    taper controlled by `TENRYU_I1B_POLE_MOTION_TRANSITION_ROWS` and
    `TENRYU_I1B_POLE_MOTION_PROFILE`.  It does not mutate mesh topology,
    hydrodynamic state, pseudo-core
    masks, or ALE/remap data structures.
- `Hydro::PoleAxisBBSW` (`src/hydro/pole_axis_bbsw.hpp`)
  - Header-only host helper for the default-off
    `TENRYU_I1B_POLE_AXIS_BBSW` Hydro2D pole-column closure.  It owns the
    planar corner/edge geometry primitives, the roundoff-scale hard-gap
    function, and the weighted PAVA projection used by `hydro_2d.cu`; it does
    not add state arrays, HDF5 schema, or namelist parameters.
- `Hydro::AntiHourglass`（`src/hydro/anti_hourglass.{cuh,cu}`）
  - 2D_RZ default-off Caramana-Shashkov subzonal pressure anti-hourglass
    force（NUMERICS §3.2.9b）。`Hydro2D::lagrangian_step` が pressure/AV
    acceleration 後、predictor/corrector velocity update 前に呼び出す。
    Compatible work が有効な場合は同じ force work を cell internal energy
    increment へ加える。Pure-Lagrange Hydro2D step では runtime
    `state.corner_mass` and `state.subzonal_mass_corner{0,1,2,3}` は
    IC-time initialization 後に Lagrangian-invariant として保持し、
    `state.corner_volume` だけを current mesh から更新する。ALE remap
    accepted compatible multiblock ALE remap 後は
    `ale_remap_2d_rz.cu` の subzonal-aware remap path が corner-mass
    fractions \(m_{c,k}/M_c\) を passive mass-weighted scalars として
    CSR conservative remap で transport し、\(\sum_k m_{c,k}=M_c\) に
    closure する。compatible-off path は従来通り warning 付き
    post-remap geometry 再初期化を行う。
  - Phase 3 multiblock path は cell-node CSR と reverse node CSR を使い、
    `state.corner_mass[c*4+k]` / `state.corner_volume[c*4+k]` を exact
    centroid+midpoint subpolygon partition で管理する。Nodal force and
    nodal mass assembly は incident corner sum で行い、single-block /
    tri_fan の default-off paths は変更しない。
- `Hydro::CompatibleForceWork2D`
  (`src/hydro/compatible_force_work_2d.{cuh,cu}`)
  - Compatible Lagrangian force/work scratch。`av_model=csw_edge`
    and `subzonal_pressure_enabled=true` の時、Hydro2D は pressure corner
    force, subzonal corner force, and edge AV force を separate in-memory
    buffers に zero/assemble し、corner-to-node and edge-to-node force
    accumulation plus per-cell work scratch を計算する（NUMERICS §3.2.9b）。
  - Legacy `scalar_vnr_legacy + subzonal_pressure_enabled=false` は従来の
    scalar \(p_q\) force assembly を呼び続ける。Compatible work scratch は
    HDF5/checkpoint schema には含めない。T5 以降の compatible path では
    corrector 後に half-state force を再計算し、同じ force と
    \(\bar u=(u^n+u^{n+1})/2\) から pressure/subzonal work を
    \(P_e/(P_e+P_i)\), \(P_i/(P_e+P_i)\)（cold limit 50/50）で 2T split
    し、CSW AV work は ion energy へ全量 deposit する。
- `Hydro::CompatibleAvCsw`
  (`src/hydro/compatible_av_csw.{cuh,cu}`)
  - T3 2D_RZ `av_model=csw_edge` force package。current corner positions
    から pressure operator と同じ full-\(2\pi\) RZ convention の edge median
    vector を作り、CSW edge force, per-cell AV work, compressive-edge count,
    and edge-relative AV CFL component を計算する（NUMERICS §3.2.9b）。
  - Multiblock limiter neighbor lookup は `face_adj_csr_offsets/indices` を
    device temporary として使い、sentinel `-1` seam/boundary は missing
    neighbor ratio 1 として扱う。HDF5/checkpoint schema は変更しない。
- `Hydro::CompatibleSubzonalPressure`
  (`src/hydro/compatible_subzonal_pressure.{cuh,cu}`)
  - `subzonal_pressure_enabled=true` path の canonical
    Caramana-Shashkov subzonal pressure force package。runtime
    `state.corner_mass` / `state.corner_volume` から corner density and
    subzonal pressure perturbation を評価し、cell-corner force and
    per-cell work scratch を compatible force buffer に assemble する
    （NUMERICS §3.2.9b）。HDF5/checkpoint schema は変更しない。
- `Hydro::MeshMotionTrace` (`src/hydro/mesh_motion_trace.hpp`)
  - Phase 3 mesh-freeze diagnosis 用の default-off host trace helper.
    `Numerics.debug.trace_mesh_motion=True` の場合だけ Hydro2D/ALE から
    device fields を host にコピーし、mesh-motion stages を stderr に出力する。
    Kernel force assembly, remap logic, and HDF5 schema are unchanged.
- `Hydro::MeshRegime` (`src/hydro/mesh_regime.{hpp,cuh,cu}`)
  - Default-off regime metadata for the 2D_RZ corner-J guard.  The
    host-readable POD types live in `mesh_regime.hpp`; CUDA declarations and
    the classifier live in `mesh_regime.cuh/.cu`.
  - The driver owns `MeshRegimeDeviceCache` and passes it to Hydro2D, avoiding
    `core::State` schema churn.  The cache holds the current `CellRegime` array
    plus one previous-primary-regime byte per cell for hysteresis.  At
    512x1024 cells this is about 13 MiB with the current compiler padding; no
    allocation occurs while
    `Numerics.hydro.regime_aware_corner_j_guard_enabled=False`.
- `Hydro::LocalRezone` (`src/hydro/ale_mode.hpp`,
  `src/hydro/local_rezone.{cuh,cu}`, `src/hydro/cd_local_rezone.{cuh,cu}`)
  - Host-side retry repair selector and local projection operators for
    driver-requested 2D_RZ ALE retries.  `AleMode` / `AleRequest` /
    `RepairPlan` are host-only data contracts; CUDA kernels are not launched
    from the header.  The legacy Winslow, axis-spine, and boundary repair
    operator bodies remain in their original modules.
- `Hydro::ALEGCL` (`src/hydro/ale_gcl.{hpp,cu}`)
  - The Geometric Conservation Law residual hook for `apply_ale_with_request`
    tail coverage of all ALE invocation paths.  It is a pure diagnostic and
    does not change ALE acceptance, remap state, or physics behavior.  The
    multiblock audit path is CSR-native for cell-node lookup and velocity
    averaging, covering both the legacy three-block mesh and the B-S2
    five-block hydro smoke.
- `Hydro::AxisALERezone` (`src/hydro/axis_ale_rezone.{cuh,cu}`)
  - Target-only primitive for the default-off 5-block half-butterfly axis ALE
    path.  It applies exact \(O(N)\) weighted lower-bound PAVA to the
    positive-mass subset of the ordered physical \(R=0\) chain derived from
    `mesh::build_full_axis_node_chain` and returns target \(Z^*\) plus first
    off-axis ring min-edge/min-altitude diagnostics. `src/hydro/ale_driver.cu`
    owns dormant-cell corner-mass zeroing, active-DOF compaction, the
    post-Lagrangian trigger, target installation, and existing CSR conservative
    remap. The primitive does not mutate state, mesh coordinates, velocity,
    force, pressure, or energy arrays.
- `src/hydro/axis_band_guard.{hpp,cu}` — 2026-07-27
  で transaction 基盤へ移行: band 行 prefix を core::ShadowTransaction の device arena に
  byte-exact capture する AxisBandGuard（旧 axis_band_snapshot.{hpp,cu} の D2H/H2D 実装を
  置換・削除。検証 assert 契約は旧実装から逐語維持）。
- `src/hydro/axis_band_margin.{cuh,cu}` — managed axis-band controller:
  row-K margin diagnostics and K-selection for managed axis-band remap.
- `src/hydro/axis_band_remap.{cuh,cu}` — managed axis-band controller:
  band-only swept-volume equal-volume remap with positivity hard gates and
  conservation diagnostics.
- `src/hydro/ale_axis_band_controller.{cuh,cu}` — managed axis-band controller:
  controller orchestration for margin evaluation, snapshot/restore K fallback,
  band remap commit, and post-remap EOS reclosure.
- `Hydro::ArtificialViscosity`（`src/hydro/artificial_viscosity.hpp`, `artificial_viscosity.cu`）
  - 1D_SPH: Christensen 速度リミタからノード勾配 \(\sigma_j\) とセル圧縮センサ \(\chi_i\) を構成し、
    \(Q_i = \phi_i \rho_i (C_2^2 \Delta r_i^2 \chi_i^2 + C_1 \Delta r_i c_{s,i} \chi_i)\) を評価
  - \(\phi_i = W_{shock,i}\max(0.25,\;W_{comp,i}W_{osc,i})\) とし、
    total pressure \(P=P_e+P_i\) の jump・密度 jump・RH整合性・圧縮Mach数・odd-even 指標で
    source-heated front と実 shock を分離する
  - `W_shock` は developed shock branch と pressure-dominated precursor branch の2分岐を持ち、
    Sedov blast launch のような極端な圧力駆動 shock-support を維持する
  - `W_osc` は Quirk 型の pressure sign-flip を抑制するが、両側 interface が developed shock のときは解除する
  - `av_type="riemann"` では 1D_SPH face で midpoint cell velocity の minmod reconstruction と
    nonlinear impedance \(Z^{eff}=\rho(c_s+\alpha\Delta u^+)\), \(\alpha=(\Gamma_1+1)/4\)
    による acoustic Riemann pressure correction を計算し、
    \(Q_i=0.5(Q_{i-1/2}+Q_{i+1/2})\) として既存の force/work 経路へ渡す。
    この branch は VNR の shock-support gate、mild-compression branch、`av_C1`、`av_C2`、
    `av_limiter_J`、`av_eos_aware` を使用しない
  - 同じ \(\chi_i\) を 1D人工熱流束 `av_heat_C` にも用いる。
  - The per-material physics operators add `compute_q_per_material_2d` for 2D_RZ
    per-material AV pressure scratch. The 2D momentum path still consumes the
    scalar aggregate `state.Qvisc = sum_m volfrac_m Q_m`; per-material energy
    deposition in `hydro_2d.cu` consumes only the scratch to avoid double
    counting. 1D_SPH per-material AV remains future work.
- `Hydro::Conduction`（`src/hydro/conduction.cuh`, `conduction.cu`,
  `conduction_snb_2d.cuh`, `conduction_snb_2d.cu`）
  - 電子熱伝導：Spitzer-Härm + flux limiter（NUMERICS §4）
  - イオン伝導（オプション、既定OFF、1D・2T、2026-09-26）：`ion_conduction_step_1d`
    （`conduction.cu`）が電子の伝導と記帳の後に Braginskii の \(\kappa_i\) で後退 Euler の
    三重対角系（電子の陰解法の組み立てカーネルを共用し、640 セル以下は 1 ブロックの並列巡回縮約、それより長い線は cuSPARSE の gtsv2 で）を解き、
    \(e_i\) に記帳してから `Hydro1D::close_eos` で閉じ直す（NUMERICS §4.6）。呼び出しは
    Driver の伝導段（`run_conduction_phase`）の末尾
  - 1D_SPH：3点トリダイアゴナル離散化（NUMERICS §3.1.7）
  - 2D_RZ：Kershaw 9点ステンシル（NUMERICS Appendix A、§4.3）
  - SNB 非局所電子熱輸送（`nonlocal_model="snb"`、既定OFF）：2D_RZ port は
    `conduction_snb_2d.{cu,cuh}`（群バッチ Kershaw-CSR Jacobi-PCG + iSNB
    Picard、NUMERICS §4.5）。dispatch は `conduction_step_2d_sts` 内
    else-wrap、既存 kernel byte 不変。probe API は verify 専用
    （`snb2d::snb2d_probe`）。1D 実装は feature/1d-brushup ブランチ（merge で合流）
  - 多材料セルでは `Materials::Mixture` の \(A_{eff},\gamma_{eff}\) を使って
    `C1 compute_spitzer_deff` が \(n_e\), \(q_{max}\), \(c_{v,e}\), \(D_{eff}\) を評価
  - **2つのソルバパス**（`conduction.solver` で選択、NUMERICS §4.2.1/§4.2.3）：
    - **`"sts"`（既定）**：Super-Time-Stepping（Chebyshev加速明示的サブサイクリング）
      - ステージ数 \(s = O(\sqrt{N_{sub}})\) で \(O(s^2)\) 倍の安定領域を実現
      - コロナ領域での \(N_{sub}=130\text{–}340\) を \(s=16\text{–}27\) に削減
      - \(D_{eff}\) と Kershaw 係数はスーパーステップ開始時に凍結
    - **`"hypre"`（オプション、`-DTENRYU_ENABLE_HYPRE=ON` 必須）**：Hypre 陰的ソルバ
      - BoomerAMG 前処理付き PCG（Kershaw 行列は SPD）
      - 陰的定式化：\((diag(C_v/\Delta t) + M_{Kershaw})\, T_e^{n+1} = diag(C_v/\Delta t)\, T_e^n\)（C_v = ρc_v）
      - 伝導CFL制約なし（\(\Delta t_{cond} = \infty\)）→ 他演算子のΔtのみがグローバルΔtを決定
      - Kershaw係数（C2カーネル出力）→ `HYPRE_IJMatrix` 変換（デバイスメモリ上）
      - スパーシティパターン（9点固定）は初回のみ構築、以後は値のみ更新
      - Hypre 未ビルド時に `solver="hypre"` 指定 → `ConfigError`
  - 負温度防止clamp（NUMERICS §11.2、§4.2.2）
  - **SNB 非局所電子熱輸送**（`src/hydro/conduction_snb_1d.cuh`, `conduction_snb_1d.cu`;
    NUMERICS §4.4、opt-in `nonlocal_model="snb"`、1D 専用）：群別 H_g 拡散方程式を
    `cusparseDgtsv2StridedBatch` 群バッチで解き（radiation FLD A-3 と同パターン）、
    面補正流束 δq を外側 f_lim cap（θ 形）と合成して STS 超ステップへ注入する
    iSNB Picard 反復（Cao 2015）。stage kernel は歴史 body の byte 不変 clone
    （`snb_stage_kernel<GEOM, KIRCHHOFF>`）。verify 専用 probe API
    `conduction::snb_probe_fluxes`（Te 不変で面流束を返す）を公開
    （VERIFICATION §7.9 の測定基盤）。既定 OFF は歴史経路 bit 恒等
    （GXII golden rel=0 ×6 で gate）。診断は ConductionResult snb_* +
    history `/diagnostics/conduction/snb/v1/*`。
- `Hydro::EOSContext`（`src/hydro/eos_context.hpp`, `eos_context.cu`）
  - per-material table-EOS context for hydro kernels (`HydroEOSContext`)
  - owns raw `DeviceEOSTable` vectors (`ion`, `electron`, `total`), hydro-only `DeviceEOSRhoETable` vectors (`total_rho_e`), hydro-only `DeviceHelmholtzSpline` vectors (`ion_helmholtz`, `electron_helmholtz`, `total_helmholtz`), hydro-only `DeviceHelmholtzJet` vectors (`total_helmholtz_jet`), and hydro-only `DeviceMieGruneisen` vectors (`mie_gruneisen`)
  - stores per-material backend selection (`hydro_backend_kind`) resolved from `Materials.materials[].eos.hydro_backend` (`0=legacy`, `1=helmholtz_spline`, `2=helmholtz_jet`, `3=exact_ideal_gas`, `4=rho_e_table`, `5=mie_gruneisen`)
  - legacy compatibility device view arrays (`d_ion_views`, `d_electron_views`, `d_total_views`) are retained for the raw table path
- `Hydro::BC`（`src/hydro/boundary.cuh`, `boundary.cu`; 2D_RZ semantic types: `src/hydro/bc_2d_rz_semantics.hpp`）
  - Lagrangianメッシュの流体境界条件（NUMERICS §8.1）
  - ゴーストセル/ゴーストノードの値更新
  - 5種別（SPECIFICATION §6.4.7）：free（P_ext=0）、fixed（v=0固定壁）、reflect（スリップ壁 v_n=0、v_t自由）、pressure（P_drive(t)、2D_RZ では r_outer のみ。z_bottom/z_top で pressure 指定は ConfigError）、state_supply（2D_RZ z-face のみ。境界 row の rho/mass/Te/Ti と material v_z を reservoir 供給値へ戻す）
  - 2D_RZ は `BC2DRZConfig` に normal/tangential material 条件、normal/tangential mesh 条件、PR B 用 open-flow remap eligibility、state-supply donor mode を展開する。`boundary_2d.mesh_tangential_target="reference"` では clamped r-face の z 座標と clamped z-face の r 座標を IC reference mesh に戻す。
  - 2D_RZ state_supply z-face の mesh anchoring は `boundary_2d.cu::apply_state_supply_z_bottom_node_kernel` / `apply_state_supply_z_top_node_kernel` が担当し、boundary node `x_z` を \(z_{min}/z_{max}\) に clamped し、mesh node `v_z` をゼロ化する。この constraint は material velocity ownership を持たない。
  - 中心境界（1D r=0）：速度反射、ゼロ流束
  - RZ軸（2D R=0）：v_R=0 強制
- `Hydro::StateSupplyBC`（`src/hydro/state_supply_bc.hpp`, `state_supply_bc.cu`）
  - 2D_RZ z-face state-supply 用の cell mask、zonal override、material velocity restoration、reservoir tally を担当する。`override_state_supply_kernel` は境界 row cell の `rho/mass/Te/Ti` を供給値へ戻し、material `v_z = supply_u_z_cm_per_s` を復元する。`restore_state_supply_material_velocity` は Hydro2D predictor/corrector boundary application 後にも同じ material `v_z` contract を再適用する。EOS 一貫性は Hydro2D 側の既存 EOS closure を再利用する。
  - Hydro2D の position update は state-supply z-boundary node 用の temporary `predictor_pos_z` / `corrector_pos_z` buffers だけをゼロ化して clamped mesh node を動かさない。一方、material `state.v_z` は supply velocity を保持し、hydro flux、diagnostics、reservoir mass/momentum/energy tally に使われる。
  - ALE projection/remap は state_supply を reflect と分離する。`ale_remap_2d_rz.cu::velocity_bc_mode_local` と `ale_driver.cu` は `STATE_SUPPLY` を mode 3 として渡し、`ale_velocity_project.cuh` は mode 3 の z-boundary `v_z` を拘束しない。Conservative remap の active state-supply z-face flux は projected node velocity ではなく `supply_u_z_cm_per_s` を boundary face speed として使う。
  - Reservoir tallies capture mass, z-momentum, and material energy deltas using the restored supplied state. Snap-0001 RH audit checks these fluxes to relative error \(\le 10^{-3}\) and asserts positive `supply_u_z_cm_per_s` on both z_bottom and z_top.

> **呼び出し関係**：Coupling::Driver → Hydro::LagrangianStep → Hydro::BC（ゴースト更新）
> → Hydro::Conduction（Strang splitting 内で独立ステップ）

**HydroEOSContext lifecycle (current implementation)**:
- Created in `coupling::Driver::run` as a stack object: `HydroEOSContext eos_ctx; eos_ctx.initialize(cfg);`
- `initialize(cfg)` uploads per-material raw table EOS (`mat.eos_tables`) into owned `DeviceEOSTable` containers and, when requested by `mat.hydro_eos_backend == "rho_e_table"`, builds the hydro-side direct `total P/T(\rho,e)` table using either the default log grid or the diagnostic linear grid selected by `cfg.numerics.hydro.rho_e_linear_grid`, then uploads it into `total_rho_e`; when requested by `mat.hydro_eos_backend == "helmholtz_spline"`, it builds the hydro-side bicubic `total P/e` surrogate and uploads it into `total_helmholtz`; when requested by `mat.hydro_eos_backend == "helmholtz_jet"`, it builds the hydro-side projected-jet biquintic `total` surrogate and uploads it into `total_helmholtz_jet`; when requested by `mat.hydro_eos_backend == "mie_gruneisen"`, it builds the branchwise affine `P_{ref}(\rho), e_{ref}(\rho), \Gamma(\rho)` fit and uploads it into `mie_gruneisen`; when requested by `mat.hydro_eos_backend == "exact_ideal_gas"`, it keeps the raw tables uploaded but marks the material for analytic ideal-gas closure inside the 1D hydro table-kernel path.
- Hydro entry points query `HydroEOSContext` once per step to select either the legacy raw-table path, the `exact_ideal_gas` path, the `rho_e_table` path, the `helmholtz_spline` path, the `helmholtz_jet` path, or the `mie_gruneisen` path; ALE post-remap reclosure also consumes the context for table-backed electron/ion inverse reclosure and otherwise keeps the ideal-gas fallback. Table backends write the closed energy back by default (`cfg.numerics.hydro.eos_writeback=true`; under the default `eos_closure_mode="energy_authoritative"` a closure that clamps at a table edge or floor keeps the hydro-updated energy instead), and `eos_writeback=false` keeps the hydro-updated `ee/ei` and only repairs NaN / Inf / negative energies with table/surrogate-clamped values. The `mie_gruneisen` path is 1D/2T-only and differs in one important way: predictor/corrector hydro uses only the uploaded affine closure for `Pe/Pi/cs`, while `Driver` refreshes raw-table `Te/Ti/cv_e/cv_i` outside hydro after hydro/source phases. In 1D, `cfg.numerics.hydro.exact_override` can further replace one post-closure quantity (`pressure`, `sound_speed`, or `temperature`) with a diagnostic ideal-gas value on table backends, and `exact_override="no_writeback"` forces writeback off for compatibility. Radiation/opacity modules continue to use the raw table data directly.
- Passed by pointer to hydro entry points (`prepare_initial_sound_speed`, `lagrangian_step`) for both 1D and 2D hydro paths, and to `Hydro::ALE` for 2D_RZ post-remap reclosure.
- `HydroEOSContext` owns GPU memory via RAII (`destroy()` + destructor + move support). Memory is released automatically when `Driver::run` exits.

#### 4.4.1 Hydro トップレベル関数シグネチャ

```cpp
struct HydroResult {
    double dt_cfl;              // [s] CFL制約から算出された推奨Δt
    int n_floor_applied;        // フロア適用回数（このステップ）
    double E_floor_injected;    // [erg] フロア注入エネルギー（このステップ合計）
};
```

`hydro_step` は `HydroResult` を返す。

```cpp
// Hydro半ステップ（Strang splitting H(Δt/2)）— NUMERICS §3.1, §3.2
HydroResult hydro_step(
    State& state,                   // メッシュ・流体場・hydro_active
    const EOSTable* eos_e,          // 電子EOS（device）
    const EOSTable* eos_i,          // イオンEOS（device）
    double dt,                      // 半ステップ幅 Δt/2 [s]
    const Config::NumericsConfig& num,  // AV係数、境界種別
    const PartitionInfo& part,      // 並列情報
    CommBuffers& comm,              // ハロー交換バッファ
    cudaStream_t stream
);

// 電子熱伝導フルステップ（Strang splitting C(Δt)）— NUMERICS §4, §4.2.1/§4.2.3
// ConductionConfig::solver が "sts" → STS明示的、
// "implicit" → 1D_SPH backward Euler + 三重対角直接解法、
// "hypre" → 2D_RZ Hypre陰的 を分岐
void conduction_step(
    State& state,
    const EOSTable* eos_e,
    double dt,                      // フルステップ幅 Δt [s]
    const Config::NumericsConfig::ConductionConfig& cond,
    const PartitionInfo& part,
    CommBuffers& comm,
    cudaStream_t stream,
    HypreSolver* hypre = nullptr   // solver="hypre" 時のみ非null。Hypre未ビルド時は常にnullptr
);

// イオン熱伝導（1D・2T、ion_conduction=True のとき）— NUMERICS §4.6。電子の伝導と
// その記帳の後に Driver が呼ぶ。e_i を記帳し、Hydro1D::close_eos で T_i, P_i を閉じ直す
// ところまでを自身で行う（下の契約は電子側のみ）。
IonConductionResult ion_conduction_step_1d(
    State& state, double dt, const Config& cfg,
    const PartitionInfo& part, cudaStream_t stream,
    const HydroEOSContext* eos_ctx);
```

**Post-conduction EOS sync 契約**：
STS/Hypre はいずれも `Te` を直接更新するが、`ee`（電子内部エネルギー密度）は更新しない。
`conduction_step` 完了後、呼び出し元（Driver）は以下を実行する義務がある：
1. `floor_clamp`（U2）：Te 安全策適用
2. `eos_forward`（H13）：更新後の Te から ee, Pe, Cv_e を再計算

これにより、後続の Radiation 演算子と2回目の Hydro 半ステップが
Te/ee/Pe/Cv_e の整合した状態を参照できる。
これがないと、伝導更新後の状態が放射係数や後続エネルギー更新に反映されない。CUDA_KERNELS §9 参照。

#### 4.4.2 Hypre 陰的拡散ソルバ（オプション）

`-DTENRYU_ENABLE_HYPRE=ON` 時のみコンパイルされる。

```cpp
#ifdef TENRYU_ENABLE_HYPRE
#include <HYPRE.h>
#include <HYPRE_parcsr_ls.h>

// Hypre ソルバの永続コンテキスト（初回構築、以後再利用）
struct HypreSolver {
    HYPRE_IJMatrix    A_ij;       // Kershaw行列 + 質量対角（HYPRE_MEMORY_DEVICE）
    HYPRE_ParCSRMatrix A_parcsr;  // 内部ParCSR形式へのビュー
    HYPRE_IJVector    b_ij;       // RHS: diag(C_v/Δt) × T_e^n
    HYPRE_IJVector    x_ij;       // 解: T_e^{n+1}（初期推定 = T_e^n）
    HYPRE_Solver      solver;     // PCG ソルバ
    HYPRE_Solver      precond;    // BoomerAMG 前処理
    bool              structure_built;  // スパーシティパターン構築済みフラグ

    // 設定パラメータ
    double rtol;         // 相対収束判定（既定 1e-8）
    int    max_iter;     // PCG最大反復数（既定 50）
    int    amg_coarsen;  // BoomerAMG粗視化タイプ（既定 HMIS=10）
    int    amg_relax;    // BoomerAMG緩和タイプ（既定 l1-Jacobi=18、GPU向き）
    int    amg_interp;   // BoomerAMG補間タイプ（既定 ext+i=6）
    int    amg_levels;   // 最大AMGレベル数（既定 25）

    void init(const PartitionInfo& part, int n_local_cells, int stencil_width);
    void update_matrix(const double* stencil_9pt, const double* rho_Cv, double dt,
                       int n_cells, cudaStream_t stream);
    int  solve(double* Te_out, const double* Te_in, cudaStream_t stream);
    void destroy();
};
#endif
```

> **データフロー（Hypre パス）**：
> 1. C1 カーネル：`D_eff` 計算（STS パスと共通）
> 2. C2 カーネル：Kershaw 9点ステンシル係数計算（STS パスと共通）
> 3. `HypreSolver::update_matrix`：ステンシル係数 → `HYPRE_IJMatrixSetValues`（デバイスメモリ上）
>    - 対角に \(C_v / \Delta t\) を加算（質量行列項、C_v = ρc_v）
>    - スパーシティパターンは初回のみ `HYPRE_IJMatrixSetRowSizes` → 以後は値のみ更新
> 4. `HypreSolver::solve`：BoomerAMG + PCG（デバイスメモリ上で完結）
> 5. 解 \(T_e^{n+1}\) を State に書き戻し → U2 + EOS forward（H13）で ee, Pe, Cv を再同期（§4.4 Post-conduction 契約）
>
> **性能特性**：
> - AMG setup：\(O(N)\) work、大きな定数（粗視化・補間構築）。毎ステップ実行（\(D_{eff}\) 変化のため）
> - PCG solve：\(O(N)\) per iteration × 5–20 iterations（典型）
> - STS との損益分岐点：STS ステージ数 \(s \gtrsim 15\text{–}20\) で Hypre が有利
> - 主な利点は Δt 制約の除去（伝導CFL free）による総ステップ数の削減

#### 4.4.3 Hydro 安定化・エネルギー制御機能

以下の機能はすべて `src/hydro/hydro_1d.cu` に実装。各機能は Config フラグで独立に有効/無効化可能。「既定」は namelist で
指定しないときの値（2026-09-29 の監査で実装に合わせた。GXII の例題デッキ `gxii_solid_1D_fld.py` などは
`av_type="csw"`・`compatible_energy=True`・`odd_even_damping_C=1.0`・`ee_odd_even_C=0.5`・`deposit_smooth_passes=3` を与える）。

| 機能 | ファイル | Config パラメータ | 既定 |
|------|---------|-----------------|---------|
| **CSW 人工粘性** | `artificial_viscosity.cu` | `av_type="csw"` | 1D の既定（`av_type` 未指定なら builder が `"csw"`、2026-08-03） |
| **VNR 人工粘性** | `artificial_viscosity.cu` | `av_type="vnr"`, `av_linear`, `av_quadratic` | 明示すれば使える |
| **Adaptive AV gate** | `shock_tracker.cu`, `adaptive_av_gate.cu`, `artificial_viscosity.cu` | `adaptive_av.enabled` | OFF (診断/改善用、VNR 専用) |
| **Riemann 人工粘性** | `artificial_viscosity.cu` | `av_type="riemann"` / `"riemann_compatible"` | OFF |
| **1D mesh motion** | `hydro_1d.cu`; optional V3 sensors in `ale_1d_sensor.cu` | `mesh.motion="lagrangian"`; `numerics.ale1d.enabled` | ON for pure Lagrangian; optional solution-adaptive ALE path OFF by default |
| **Odd-even 圧力フィルタ** | `hydro_1d.cu` | `odd_even_damping_C` | OFF（既定 0。GXII デッキは 1.0） |
| **電子 odd-even ダンピング** | `hydro_1d.cu` | `ee_odd_even_C` | OFF（既定 0。GXII デッキは 0.5） |
| **Compatible-energy 2T** | `hydro_1d.cu` | `compatible_energy` | OFF（既定。GXII デッキは True） |
| **高波数速度ダンパー** | `hydro_1d.cu` | `hk_velocity_damper_C` | OFF (フロント近接) |
| **イオン人工熱伝導** | `hydro_1d.cu` | `ion_art_heat_C` | OFF (Pe支配で無効) |

**Compatible-energy**: 正確な ΔIE = -ΔKE をノード力仕事分解で計算。Odd-even ダンピング力を compatible work に含み、separate heat_oe を除去。legacy PdV 散逸を除去するため、単独では振動が悪化。明示的安定化と組み合わせが必要。

**Riemann AV**: 非線形インピーダンス Z_eff = ρ(cs + α Δu⁺) による音響リーマン解。minmod 再構成。VNR のセンサー/リミッター不使用。現状では ICF 爆縮に対して散逸不足。

**Adaptive AV gate**: 1D_SPH + VNR 専用。hydro step 先頭で base VNR probe から leading shock cluster を同定し、`State.adaptive_av_gate` の履歴付き gate で `C1/C2/heat_C/Cpsv/cbulk` をセルごとに補間する。tracker は `State.adaptive_av_r0`、前回 shock 半径/速度、bounce latch を保持し、predictor/corrector 内では同じ gate field を固定して使う。`Cpsv` は Stage 1 では既存 post-shock nodal damping の per-cell 係数として渡し、compatible-energy rewrite は別段階の対象。

**高波数速度ダンパー**: 3ノード線形フィット → 残差 dv → 保存的ペアインパルス → KE→ion heat。25セル前方バッファガード付き。アブレーション面近接で多段衝撃波を誘発するため無効化。

**イオン人工熱伝導**: 衝撃波制限 κ = C_H × S_comp × S_contact × β × cv_i。二次AVに連動。Pe が全圧力の 73% を占め、Te≈Ti のため、イオンのみの熱伝導は間違った圧力成分に作用。

#### 4.4.4 レーザーデポジットスムージング

`src/laser/deposit_transfer.cu` に実装。保存的質量重み付き Laplacian フィルタを laser 沈着パワーに適用。レイの離散的サンプリングによるステアケース状沈着パターンを平滑化し、多段衝撃波の発生を防止。

| Config | 値 |
|--------|-----|
| `deposit_smooth_passes` | 0（既定 = 平滑化なし。1D の例題デッキの一部は 3） |
| `deposit_smooth_alpha` | 0.25 |

---

### 4.5 radiation/

> **【現行の輻射モデル】** 採用モデルは決定論の **FLD（`mode="multigroup_diffusion"`, NUMERICS §6.7）** と **\(S_N\)（`mode="sn_transport"`, NUMERICS §6.8）** の 2 つ。構成：`Rad::FLD1D`/`Rad::FLD2DRZ`（`fld_1d_gpu`/`fld_2d_rz_gpu`, `driver_fld_energy`；線形 solver は cuSPARSE tridiag (1D) / AmgX-CG・cuSPARSE CG variants (2D)）と `Rad::SNTransport1D`/`Rad::SNTransport2DRZ`（`sn_transport_1d_gpu`/`sn_transport_2d_gpu`, `sn_dsa`, `sn_material_newton`；加速は DSA + RKL2/AMGX）。driver からの入口は `Rad::RadiationStep`。退役したモンテカルロ輻射（IMC・DDMC・ランダムウォーク・HOLO・difference 定式化：`Rad::IMC`・`Rad::DDMC`・`Rad::Diffusion*`・`Rad::Tally`・PhotonPool 等）は 2026-09-29 にビルドから外し、コード・試験・デッキを `retired/radiation_monte_carlo/` へ、本節にあったその設計記述（モジュール・`CellRadiationCoeffs`・旧トップレベル関数・カーネル起動仕様・分散低減・ハイブリッド輸送）を `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した。

**責務**：1D_SPH/2D_RZ multigroup **FLD**（既定）と 1D_SPH/2D_RZ pure **\(S_N\)**。群構造と Planck 分率、不透明度の評価、物質との結合（Newton）、沈着 `rad_dep`・放出 `rad_emit` の publish。

- `Rad::RadiationStep`（`src/radiation/radiation_step.hpp`, `radiation_step.cpp`）：driver が輻射演算子 \(\mathcal{R}(\Delta t)\) として呼ぶ入口
  （2026-09-29 まで `radiation::IMC::transport_step` がこの役を兼ねていた）。輻射が無効・\(\Delta t \le 0\)・セルや材料が無いときは
  何もしない。2D_RZ では境界の組（内側 reflect、外側 vacuum/reflect、z 面 vacuum/reflect/marshak）を検査する。群境界は
  `group_bounds_eV`、無ければ `compute_T_range_eV` の対数等分。`PlanckTable` は群境界・温度範囲・`compute_N_T` が変わったときだけ
  作り直す（run の最初の輻射ステップで作る）。mode と次元で `advance_radiation_step_fld_1d` / `_fld_2d_rz` / `_sn_1d` /
  `_sn_2d_rz` を呼ぶ。最大値原理の超過（NUMERICS §11.8）は driver が輻射の後に測って `set_last_overshoot_metrics` で渡し、
  history の `radiation/overshoot_count`・`radiation/overshoot_max` と `Numerics.safety.overshoot_warn` の判定に使う。

- `Rad::Groups`：群境界（eV）、代表値、Planck fraction b_g(T)（計算/テーブル）
  - GPU側問い合わせ：`__device__ double planck_fraction(int g, double T, const PlanckTable* table)`
  - テーブルは温度グリッド（既定 200 点、log-uniform、0.01–100.0 eV）で計算し device メモリに保持（SPECIFICATION §6.4.5 既定）
  - 実行中は \(\ln T\) の 3 次 Hermite 多項式で累積分率 \(C_g\)（または裾の和 \(D_g=1-C_g\)）の対数を補間し、\(b_g\) を隣り合う値の差として得る
    （和は構成上 1。2026-09-23 まで \(b_g\) を \(T\) の線形で補間していた。NUMERICS §6.1）
- `Rad::GroupStructure`（`src/radiation/group_structure.hpp`, `group_structure.cu`）：
  optional hard-X-ray group-boundary repacking and host-side opacity-table resampling.
  It depends on `materials::IonmixOpacityData` and preserves the existing group count.
- `Rad::FLD1D`（`src/radiation/fld_1d_gpu.cuh`, `fld_1d_gpu.cu`,
  `nlte_coeffs.cu`）：
  `Radiation.mode="multigroup_diffusion"` 専用の 1D_SPH multigroup flux-limited diffusion 経路。
  cell×group の `rad_E` と persistent
  `rad_E_old` を有限体積 backward Euler で更新する。群ごとの三重対角系は
  cuSPARSE `gtsvStridedBatch` で GPU 上に solve し、HYDRA-aligned Fleck linearization
  (`f·σ^PA` total removal diagonal、`f·η + (1-f)·c·σ^PA·E^n` RHS) で stiff な
  物質-放射結合を放射線形系へ implicit に組み込む（NUMERICS §6.7、FLD-FIX-1）。
  物質結合は GPU Newton で `Te`, `ee`, `Pe`, `rad_dep`, `rad_emit` を更新し、
  σ^PA を吸収・σ^PE を放射に分離する PA/PE-consistent residual を用いる。
  TMAT/table EOS が利用可能な場合は `materials::DeviceEOSTableView` 経由で
  `e_e(ρ,T)` と `c_{v,e}(ρ,T)` を Newton iteration 内で評価し、収束後の
  `ee`/`Pe` を同じテーブルから書き戻す（電子 EOS device view が無い場合は
  定比熱 + ideal-gas 圧力へ fallback）。
  Persistent state は `rad_E_old`, `fld_sigma_a`, `fld_sigma_pe`,
  `fld_sigma_R`, `fld_eta`, `fld_nlte_f_work`, `fld_nlte_sigma_eff_work`,
  `fld_D_cell`, `fld_lower/diag/upper/rhs`, `fld_Te_old` を使う。
- 1D の多材料の係数（2026-09-24、FLD と \(S_N\) が共有）：
  `src/radiation/multimat_opacity_1d.cuh`（材料ごとの不透明度の記述
  `MatOpacityDesc` と、セルの材料から部分密度で不透明度を混合する
  `eval_opacity_multimat_kernel`、\(S_N\) の物理散乱 `eval_scattering_multimat_kernel`。
  無名名前空間なので取り込む翻訳単位ごとに実体を持つ）と
  `src/radiation/material_electron_eos_1d.hpp`/`.cu`（材料ごとの電子 EOS 表の device
  配列とセルごとの選択 `cell_electron_table_selector_1d`、物質結合の Newton が使う）。
- `Rad::FLD2DRZ`（`src/radiation/fld_2d_rz_gpu.cuh`, `fld_2d_rz_gpu.cu`,
  `src/radiation/amgx_solver.hpp`, `amgx_solver.cpp`）：
  `Radiation.mode="multigroup_diffusion"` かつ `Main.dimension="2D_RZ"` の
  multigroup FLD 経路。R/Z 4-face finite-volume CSR system を群ごとに組み立て、
  AmgX が link されている場合は `linear_solver_2d="amgx_cg"`、未検出 build では
  WARNING を出して `cusparse_cg_jacobi` debug fallback で solve する。
  `cusparse_cg_zline` は radial line ごとの z-tridiagonal を cuSPARSE `cusparseDgtsv2`
  で解く z-line block-Jacobi preconditioned CG option である。R軸は reflect、
  outer R は vacuum、Z端は `z_boundary` / `boundary.z` で vacuum、reflect、
  Marshak、または grey one-group state_supply Dirichlet を選ぶ。state_supply
  Dirichlet の供給温度は hydro z-face state-supply config から取得し、step/cumulative
  reservoir tally は `State::fld_state_supply_*` に保持する。
  HYDRA-aligned Fleck linearization、PA/PE-consistent matter Newton、TMAT-aware
  electron EOS 連携は 1D_SPH と同一仕様（NUMERICS §6.7、FLD-FIX-1）。
  `State.fld_fleck[n_cells]` は output-only の per-cell Fleck factor diagnostic として
  matter update で publish する。
- `Rad::SNTransport1D`（`src/radiation/sn_transport_1d_gpu.cuh`,
  `sn_transport_1d_gpu.cu`, `sn_dsa_1d_gpu.cu`,
  `sn_material_newton_gpu.cu`; 線形不連続法 `sn_ld_1d_gpu.{cuh,cu}`,
  共有部 `sn_transport_1d_internal.hpp`（不透明度評価・角度求積・状態配列）と
  `sn_electron_eos.cuh`（電子 EOS の尾部・逆算・比熱））：
  `Radiation.mode="sn_transport"` 専用の 1D_SPH pure \(S_N\) production 経路。
  `spatial_scheme="linear_discontinuous"`（1D の既定、2026-09-25）は
  `advance_radiation_step_sn_1d` が Marshak 入射・体積源の設定後に
  `sn_ld::advance_step` へ渡し、節点強度の掃引（群ごとに 1 warp。セル行列の逆行列は
  Newton 反復ごとに前計算）、整合 P1 系の block Thomas（Newton 反復ごとに 1 回の組み立てと
  ブロック消去、対角ブロックは部分ピボットの Gauss–Jordan で逆行列にして保持、右辺ごとの代入）、節点温度の Newton と
  吸収率密度の GMRES、節点物質更新（線形化と物質更新は節点ごとに 1 warp、群の Planck 項を
  レーンで並列、EOS の逆算は warp 並列の二分法）をすべて GPU 上で行う（host は GMRES の
  Hessenberg 小行列と収束判定のみ。NUMERICS §6.8.4）。以下は `"linear_characteristic"` の記述。
  group-parallel/angle-serial
  spherical sweep、group-batched DSA tridiagonal solve、GPU Newton material
  coupling を CUDA 上で実行する。
  Pure SN は Fleck linearization を bypass し（\(f=1\)、Fleck-derived effective
  scattering = 0）、raw \(\sigma^{PA}\) を sweep 吸収・raw \(\sigma^{PE}\) を
  emission に渡す（NUMERICS §6.8、Cut-FIX-4）。
  1D_SPH sweep は Morel-Adams angular-edge redistribution のため group 内で
  ordinate order を保持し、pair-parallel angle sweep は debug diagnostic のみ。
  各 cell では Cut-FIX-3 の metric scaling \(a_{c,n+1/2}=(\Delta A/w_n)\alpha_{n+1/2}\)
  により球面 LTE fixed-point cancellation identity を保つ（NUMERICS §6.8.1）。
  1D_SPH DSA は outer-vacuum Robin leakage と \(r=0\) scalar-parity no-flux
  boundary terms を transport boundary convention と一貫させて組み立てる。
  Material Newton は conservative active-set + face-flux + donor-theta +
  AP face-blend closure に固定され、σ^PA 吸収・σ^PE 放射の PA/PE split residual を用い、
  TMAT/table EOS が利用可能な場合は `materials::DeviceEOSTableView` 経由で
  `e_e(ρ,T)` と `c_{v,e}(ρ,T)` を Newton iteration 内で評価する（Cut-FIX-5；
  EOS view が null のときは定比熱 + ideal-gas fallback）。
  Persistent state は `rad_E_old`, `sn_psi_scratch`, `sn_dsa_*`,
  `sn_phi_*`, `sn_eta`, `sn_sigma_a` (= raw σ^PA), `sn_sigma_pe` (= raw σ^PE),
  `sn_sigma_s` (Fleck bypass scattering; AP face_blend では \(\sigma_R\) として再利用),
  face-flux mode 用の
  `sn_face_flux_raw[(n_cells+1)×G]`, donor-theta limiter 用の
  `sn_face_flux_limited[(n_cells+1)×G]`, `sn_stream_theta[n_cells×G]`,
  AP face_blend 用の `sn_face_flux_diff[(n_cells+1)×G]`,
  `sn_face_alpha[(n_cells+1)×G]`, `sn_E_star_flux[n_cells×G]` を使い、deterministic
  `rad_dep`/`rad_emit` を publish する。
- `Rad::SNTransport2DRZ`（`src/radiation/sn_transport_2d_gpu.cuh`,
  `sn_transport_2d_gpu.cu`, `sn_transport_gpu.cu`）：
  `Radiation.mode="sn_transport"` かつ `Main.dimension="2D_RZ"` の pure \(S_N\)
  経路。product level-symmetric quadrature、R軸 reflect parity、outer R vacuum、
  configurable Z vacuum/reflect boundary、および GPU 2D DSA Jacobi correction を使う。
  Fleck bypass と TMAT-aware electron EOS 連携は 1D_SPH と共通だが、material
  coupling closure は 2D Concern 1-4 実装まで legacy non-implicit closure を内部で使う。
  AP face-blend 後に `State.sn_tau_R`, `State.sn_reduced_flux`, `State.sn_ap_alpha`
  へ per-cell transition diagnostics を publish する。
  Cut-2 では機能検証優先であり、GXII-scale 2D \(S_N\) 性能最適化は別タスクである。

**PlanckTable**（`src/radiation/planck_table.cuh`）：host のクラス `PlanckTable`（`build(groups, n_T, T_min_eV, T_max_eV)`、
picket-fence 用の `build_constant_fractions`）が温度格子とその対数、群ごとの \(b_g\)・累積分率 \(C_g\)・裾の和 \(D_g\)、
\(\ln C_g\)・\(\ln D_g\) とその \(\ln T\) 微分を device に置き、kernel へは `PlanckTableDeviceView`（`interpolate_b(g, T)`、
多群を同じ温度で引く `locate_b(T)`）として渡す。表の範囲外の温度は端の値に丸める（host 側の評価は最初の 1 回だけ WARNING を出す）。
`planck_fraction.method="tabulate"` は受理して保存するが使われず、計算した分率に戻す（WARNING。SPECIFICATION §6.4.5）。


#### 4.5.1 Radiation トップレベル関数シグネチャ

```cpp
// 輻射演算子 R(Δt)（NUMERICS §2.1、§6.7、§6.8）。driver が 1 ステップに 1 回（Radiation.imc.two_stage では
// 半ステップずつ 2 回）呼ぶ。物質の更新（Te, ee, Pe）と rad_dep / rad_emit の publish は各ソルバーの中で行う。
class RadiationStep {
 public:
  void step(core::State& state, const core::Config& cfg, double dt,
            const parallel::PartitionInfo& part = {},    // MPI の分割（2D_RZ の r-slab）
            parallel::CommBuffers* bufs = nullptr,       // n_ranks > 1 では必須
            double drive_time_s = NaN);                  // 1D の時間依存の境界駆動を引く時刻（NaN = state.t）
  void set_last_overshoot_metrics(std::int64_t count, double max_ratio);  // step() が 0 に戻す
  std::int64_t last_overshoot_count() const;             // history radiation/overshoot_count
  double last_overshoot_max() const;                     // history radiation/overshoot_max
};
```

#### 4.5.2 Radiation GPU カーネル起動仕様

現行の FLD・\(S_N\) のカーネルと起動設定は CUDA_KERNELS §6.7（FLD）・§6.8（\(S_N\)）を参照。
（本節にあった IMC/DDMC のカーネル起動仕様は `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した。）

#### 4.5.3 放射輸送の分散低減・加速機能

退役（モンテカルロ輻射の分散低減・加速機能。記述は `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した）。

#### 4.5.4 ハイブリッド輸送（研究ブランチ、本番 OFF）

退役（IMC + DDMC/PGRW + 決定論拡散の 3 モード構成。記述は `retired/radiation_monte_carlo/docs/ARCHITECTURE_monte_carlo.md` へ移した）。

---

### 4.6 laser/
**責務**：レーザービーム追跡と沈着

- `Laser::Beams`：ビーム定義（方向、焦点、F値、D/R、プロファイルテーブル、波形テーブル）
  - 初期化時にプロファイル関数をサンプリングしてテーブル化（Python呼び出し禁止方針に準拠）
  - `spot`（非推奨）が指定された場合は内部で `profile` に自動変換
  - `radial_absorption_1d` では各ビームの `power(t)` のみを合算し、方向・F値・焦点・D/R・プロファイル・レイ本数は吸収分布に使わない
- `Laser::CoordinateTransform`：座標変換
  - 1D_SPH：1D球座標 \(r\) → ビームローカルRZ座標 \((R,Z)\) への射影（球対称仮定 \(Q(\sqrt{R^2+Z^2})\)）
  - 1D_SPH：ビームローカルRZ → 1D球座標 \(r\) への逆射影（吸収エネルギーの転写用）
  - ビーム方向ベクトルからRZ座標系の原点・軸を構築
  - 2D_RZ：3D Lab座標 → LaserMesh座標の変換 \((R,Z)=(\sqrt{x^2+y^2},z)\)（NUMERICS §5.3.4）
  - 2D_RZ：ビーム軸直交平面の正規直交基底 \((\hat{\mathbf{u}},\hat{\mathbf{w}},\hat{\mathbf{d}})\) 構築
- `Laser::Mesh`：**1つの**LaserMesh（2D RZ構造格子）を生成
  - 1D_SPH：代表ビーム軸をZ軸とするビームローカル2D RZ格子
  - 2D_RZ：**流体対称軸に沿う** 2D RZ格子（ビームローカルではない、NUMERICS §5.7.1）
  - 1D_SPH：\(\rho(r), T_e(r), \bar Z(r)\) をRZメッシュ上に \(\rho(\sqrt{R^2+Z^2})\) としてマッピング
  - 1D_SPH：ray trace 用に midplane から radial side arrays
    \((r,\hat n,\hat n_{raw},A_{smooth},d\hat n/dr)\) も併せて保持
  - 2D_RZ：2D_RZ HydroMeshの \(\rho(R,Z), T_e(R,Z), \bar Z(R,Z)\) を直接マッピング（NUMERICS §5.7.3 (b)）
  - `critical_clip`：節点の \(\hat n\) を `critical_margin` で頭打ちにする（値の上限。1D の格子は臨界面より奥も含む）
  - 1D は写像のたびに節点を作り直す：臨界面の近くを細かく、内外へ幾何級数で粗くする graded 配置（NUMERICS §5.7.2）。
    ゴーストコロナが外半径を超えると外半径を広げて作り直す（2026-09-29）。2D は初期化時の一様格子。
    （密度勾配で伸縮する格子は無い — `stretch_method`・`min_ratio` はどのメッシュも読まず、指定すると警告）
  - 節点（node-centered）に \(\hat n = n_e/n_{crit}\), \(T_e\), \(\bar Z\), \(\nabla\hat n\) を保持
  - 節点での中心差分による密度勾配 \(\nabla(n_e/n_{crit})\) の計算
- `Laser::Cbet`（`cbet.cu/.cuh`、v1 = 1D_SPH `raytrace_2d` + 2D_RZ `raytrace_3d` opt-in）：Marozas 型保存的 pairwise CBET
  - trace kernel の record モード（`template<bool kCbetRecord>`、OFF 実体化は従来と構造同一）が ray 毎のセル横断記録を生成
  - CbetWorkspace（grow-only device 常駐）上で決定論的固定点反復（tally → 反対称交換+donor cap → IB/2·CBET·IB/2 propagate）
  - 沈着・未吸収は per-ray 行 + 固定順 reduction でビーム毎に集計し、既存の deposit 再配分・skip cache 経路へ接続（NUMERICS §5.10）
  - 2D_RZ CBET は theta-group ごとの record-mode trace → joint exchange solve → per-group LaserMesh node deposit を生成し、既存の 2D transfer path に接続する。Workspace singleton は 1D と共有し、recorder template 実体化は defining TU に閉じる；nvcc+RDC では header 側 template declaration を増やすと OFF path まで再実体化されるため、新規宣言は wrapper/header isolation で分離する。
- `Laser::HotElectron1D`（`hot_electron_1d.cuh/.cpp`、`hot_electron_1d_gpu.cu`）：1D hot-electron preheat: capture reduction, cone quadrature, chord walkers, CSDA pipelines。本番の cone の経路は device（`hot_electron_1d_gpu.cu`、`laser.cu` から起動）、host の実装は参照用
- `Laser::HotEEtaModel`（`hot_e_eta_model.cpp`）：\(\eta(t)\) の閾値モデルと緩和（NUMERICS §5.11.3）
- `Laser::PortSection`（`port_geometry`、`port_section_chi`、`port_section_overlap`、`sector_phase_space`、`sector_adapter`）：
  実ポート配置の多ビーム CBET（`cbet.geometry_mode="port_section"`、NUMERICS §5.10.8）
- `Laser::HotElectron2D`（`hot_electron_2d.cuh/.cpp`）：2D RZ hot-electron transport: topology-agnostic MeshView2D, revolved-face chord walker, 3D band quadrature, capture reduction, host cone pipeline（reference path）
- `Laser::HotElectron2DGpu`（`hot_electron_2d_gpu.cuh/.cu`）：device chord pipeline（1 thread/chord, deterministic host fold）+ scratch-pooled staging

**LaserMesh 構造体**（NUMERICS §5.7 準拠）：

```cpp
// レーザーレイトレース用 2D RZ 構造格子（Laser モジュール内部管理）
// NUMERICS §5.7.1–§5.7.5 準拠
//
// LaserMesh はリクティリニア（テンソル積）グリッドである。
// ノード座標は 1D 配列に因数分解される: node_R[nr+1], node_Z[nz+1]
// 2D ノード (i,j) の物理位置は (node_R[i], node_Z[j]) で決定される。
// ALE による一般四辺形メッシュ（HydroMesh / Mesh §4.2.2）とは異なり、
// LaserMesh は常に直交格子を維持する（2D_RZ の双線形補間と
// 1D_SPH の radial side-array 抽出の基盤）。
struct LaserMesh {
    // --- 格子構造 ---
    int     nr, nz;                 // セル数（既定 128×256）
    int     n_nodes_r, n_nodes_z;   // 節点数 = (nr+1), (nz+1)

    // --- 節点座標（deviceメモリ、ストレッチ格子対応）---
    // テンソル積構造：1D配列 node_R × node_Z で2Dグリッドを定義
    double* node_R;                 // [n_nodes_r] R方向節点座標 [cm]
    double* node_Z;                 // [n_nodes_z] Z方向節点座標 [cm]

    // --- 節点物理量（node-centered、deviceメモリ）---
    // メモリレイアウト：row-major [i * n_nodes_z + j]（i=R方向, j=Z方向）
    double* n_e_hat;                // [(nr+1)×(nz+1)] 正規化電子密度 n_e/n_crit（臨界でクリップ）[dimensionless]
    double* n_e_hat_raw;            // [(nr+1)×(nz+1)] クリップ前の n_e/n_crit [dimensionless]
    double* T_e;                    // [(nr+1)×(nz+1)] 電子温度 [eV]
    double* Zbar;                   // [(nr+1)×(nz+1)] 平均電荷数 [dimensionless]
    double* smooth_kappa_factor;    // [(nr+1)×(nz+1)] 逆制動輻射の smooth 係数（事前計算）

    // --- 密度勾配（node-centered、中心差分で計算）---
    double* grad_n_hat_R;           // [(nr+1)×(nz+1)] ∂(n_e/n_crit)/∂R [1/cm]
    double* grad_n_hat_Z;           // [(nr+1)×(nz+1)] ∂(n_e/n_crit)/∂Z [1/cm]

    // --- 1D の光線追跡の径方向プロファイル [radial_n_nodes ≤ radial_capacity] ---
    // 2026-09-24 以降は map_from_hydro_1d が hydro セルに結び付けて作る節点
    // （面・臨界の対・プロファイルの節点、laser_mesh_bodies::build_trace_profile_nodes_1d）。
    // それ以前は 2D レーザー格子の軸の列（Z=0 上の [nr+1]）だった。
    double* radial_node_r;          // 節点の半径 r [cm]
    double* radial_n_hat;           // クリップ後の n̂
    double* radial_n_hat_raw;       // クリップ前の n̂
    double* radial_smooth_kappa;    // smooth 係数（compute_smooth_kappa が埋める）
    double* radial_T_e;             // 電子温度 [eV]
    double* radial_Zbar;            // 平均電荷数
    double* radial_dn_dr;           // d(n̂)/dr [1/cm]
    int     radial_n_nodes;         // 使用中の節点数
    int     radial_capacity;        // 確保済みの節点数
    bool    trace_profile_1d;       // map_from_hydro_1d が設定（プロファイルを使う）
    TraceProfileMap1D trace_profile_map; // 直近の 1D 写像の入力（外表面・臨界のセル、ゴーストの幅・密度など）
    int     geometry_code;          // 写像した流体の Mesh.geometry_1d（0 球、1 円筒、2 平板）

    // --- 沈着配列（node-centered）---
    double* deposit;                // [(nr+1)×(nz+1)] 吸収パワー [erg/s]（NUMERICS §5.5）

    // --- メタ情報 ---
    double  R_max;                  // R方向上限 [cm]
    double  Z_min, Z_max;           // Z方向範囲 [cm]
    double  n_crit;                 // 臨界密度 [1/cm³]
    double  n_hat_margin;           // [dimensionless] 臨界密度クリップ閾値（既定 1-eps_crit=0.9999）（NUMERICS §5.7.1）

    // --- 1D_SPH map device buffers ---
    double* hydro_A_eff_device;     // [hydro_cell_capacity] 1D_SPH map用 A_eff scratch
    uint8_t* hydro_cell_is_void_device; // [hydro_cell_capacity] 1D_SPH map用 void flag scratch
    int     hydro_cell_capacity;    // hydro scratch capacity [cells]

    // （1D の写像は step 間平滑化をしないので状態を持たない。2026-09-24 に
    //   prev_n_hat_device と State::laser_nhat_smoothing_{r,n_hat} を撤去、NUMERICS §5.7.4）

    // --- ゴーストコロナ設定（1D_SPH、NUMERICS §5.7.5）---
    bool    ghost_corona_enabled;           // ゴーストコロナ有効フラグ
    int     ghost_n_out;                    // ゴーストセル数（既定 12）
    double  ghost_ne_min_frac;              // ゴースト密度下限比 n̂_min [dimensionless]
    double  ghost_ne_max_frac;              // ゴースト密度上限比 n̂_max [dimensionless]
    double  ghost_Te_min_eV;                // ゴースト電子温度下限 [eV]
    double  ghost_zbar_min;                 // ゴースト Z̄ 下限 [dimensionless]
    double  ghost_zbar_max;                 // ゴースト Z̄ 上限 [dimensionless]
    int     ghost_handoff_cells;            // ハンドオフステンシル深さ（既定 4）
    double  ghost_handoff_decay;            // ハンドオフ指数関数減衰長（既定 1.5）

    // --- ブローオフ遷移モデル（NUMERICS §5.7.5.4）---
    bool    ghost_transition_enabled;       // 遷移モデル有効フラグ
    double  ghost_transition_resolved_nhat; // resolved-corona 判定閾値 n̂ [dimensionless]（既定 0.9。ゴーストの減衰にも使う）
    int     ghost_transition_resolved_cells;// ゴーストが消え、ハンドオフが復帰する亜臨界セル数（既定 3）
    double  ghost_transition_density_exponent; // 密度バイアス指数 α_ρ（既定 1.0）
    double  last_ghost_transition_blend;    // 直近のブレンド係数 β（診断用、毎ステップ更新）
    int     last_ghost_transition_resolved_cells; // 直近の resolved セル数（診断用、毎ステップ更新）
    double  last_ghost_width;               // 直近の 1D マッピングのゴースト幅 W [cm]（0＝なし。診断用、NUMERICS §5.7.5.2）
    double  last_trace_unabsorbed_power;    // raytrace/skip 直後の未吸収パワー [erg/s]（診断用）
    double  last_transfer_blocked_power;    // transfer で受け皿がなく捨てたパワー [erg/s]（診断用）
    double  last_unabsorbed_power;          // transfer 後の最終未吸収パワー [erg/s]（診断用）
    double  last_commanded_energy;          // 当該 step の入射エネルギー [erg]（診断用）
    int64_t last_tail_closure_count;        // tail closure で終了した ray 数 [count]（診断用）
    double  last_tail_closure_absorbed_power; // tail closure で吸収されたパワー [erg/s]（診断用）
    int64_t last_critical_surface_hit_count; // 臨界面で打ち切られた ray 数 [count]（診断用）

    // device function: 位置 (R,Z) [cm] から補間済み物理量を取得（双線形補間）
    __device__ double interp_n_hat(double R, double Z) const;  // 戻り値: n̂ [dimensionless]
    __device__ void   interp_grad(double R, double Z,          // 出力: ∂n̂/∂R, ∂n̂/∂Z [1/cm]
                                  double& dndR, double& dndZ) const;

    // 格子外判定
    __device__ bool is_outside(double R, double Z) const;
};
```

> **所有権**：`LaserMesh` は Laser モジュールが `init()` 時に確保し、ステップ毎に
> 物理量を HydroMesh から再マッピングする（NUMERICS §5.7.4）。
> State のメンバーではない（§5.2 の注記参照）。
> 2D の格子点配置は初期化時に固定。1D は写像のたびに節点を作り直し（`map_from_hydro_1d`、NUMERICS §5.7.2）、
> 径方向の配列は hydro セルに結び付いた節点（面・臨界の対・プロファイルの節点）を持つ。上の構造体は主要な欄だけを示し、
> 容量の管理（`*_capacity`）、ステップごとの作業領域（`scratch_*`）、CBET の診断（`last_cbet_*`）などは
> `src/laser/laser_mesh.cuh` を参照。

- `Laser::RayInit`：レイ初期条件の生成
  - 1D_SPH：F値と集光位置からレイの初期位置（R方向1D配列）・方向を計算（NUMERICS §5.6.3 (a)）
  - 2D_RZ：ビーム軸直交平面上の2D断面配列としてレイを初期化（NUMERICS §5.6.3 (b)）
    - 正規直交基底 \((\hat{\mathbf{u}},\hat{\mathbf{w}})\) 上の格子点座標 → 3D Lab座標への変換
    - 初期方向は3D焦点座標に向かうベクトル
  - ビームプロファイルからレイのパワー重みを計算（1D_SPH：環状面積 \(2\pi R\Delta R\)、2D_RZ：断面面積 \(\Delta u\Delta w\)）
  - 2D_RZ：極角θとパラメータによるビームグループ化（NUMERICS §5.6.4）
- `Laser::RayTrace`：幾何光学（屈折）+ IB吸収
  - 1D_SPH `raytrace_2d`：既定の `integrator="auto"` は特性曲線積分（`ray_trace_characteristic.cuh`、NUMERICS §5.3.6）で、
    臨界半径で反射する（`terminate=False` が既定）。`integrator="leapfrog"` は 2D ベクトル \((R,Z)\) の刻み幅可変の Verlet 行進
    （NUMERICS §5.3.2、臨界で終了）。円筒・平板の 1D は特性曲線積分だけ
    - 場参照は 2D bilinear ではなく radial side array への 1D linear lookup
    - 勾配は \(d\hat n/dr\) から \((\partial\hat n/\partial R,\partial\hat n/\partial Z)\) を再構成
    - 吸収パワーは Hydro の 1Dセル配列へ直接蓄積（決定論の固定順）
  - 1D_SPH `radial_absorption_1d`：レイ初期化を行わず、全ビームパワー合計を `launch_radial_absorption_1d` へ渡して外側セルから
    内側セルへ動径積分する（256 スレッドで 1024 セルずつ並列に評価。MPI では全 rank が同じ全線を計算し、所有窓で適用 — 2026-09-29）
  - 2D_RZ：**3Dベクトル \((x,y,z)\)** でLeapfrog追跡（NUMERICS §5.3.4）
    - 2D勾配→3D変換：\(\partial\hat n/\partial x = (\partial\hat n/\partial R)(x/R)\) 等
    - R=0特異性処理：\(R < R_{floor}\) で勾配の横方向成分を零とする
  - 任意点での場参照は次元依存（1D_SPH: radial 1D linear、2D_RZ: 2D bilinear）
  - IB吸収は台形公式による光学厚 \(S\) 計算（NUMERICS §5.4準拠）
  - 臨界処理は2段階（`terminate=True` の経路。1D の特性曲線積分の既定は反射）
    - 通常は \(\varepsilon_{crit}\) 面までセグメントを切り詰めて terminate する（NUMERICS §5.2）
    - 臨界近傍では `critical-layer mode` に切り替え、carried \(\kappa\) から再構成した \(A_{entry}\) と \(|\nabla \hat n|\) で解析 tail 光学厚 \(\tau_{tail}\) を計算し、1D_SPH は entry 半径を含む Hydro 1Dセル、2D_RZ は entry 点の bilinear nodes に沈着して終了する（NUMERICS §5.4.4）
- `Laser::DepositMap`：LaserMesh → HydroMesh の写像
  - 吸収パワーの空間分配は次元依存（1D_SPH: 1D cell direct、2D_RZ: 4-node bilinear）
  - 2D_RZ：3D中間位置 \((x,y,z)\) → \((R,Z)=(\sqrt{x^2+y^2},z)\) でLaserMeshセルを特定して分配
  - 1D_SPH：ray trace または `radial_absorption_1d` 中に Hydro の 1Dセル配列へ直接沈着し、その後 host 側で blocked/ghost handoff を適用
  - 2D_RZ：LaserMesh沈着を **直接** 2D_RZ HydroMeshへ双線形補間で分配（NUMERICS §5.8.1 (b)、1D球座標転写は不要）
  - 多ビーム：1D は各ビームをそれぞれのパワーで追跡し、全ビームのキーが一致するときだけ 1 回の追跡を使い回す。
    2D は極角グループごとに追跡してグループ内パワー合計でスケーリング（NUMERICS §5.6.4）
  - エネルギー保存検証（転写前後の差分 ≤ \(10^{-10}\)）
  - `laser_dep` の `ee` 注入は Coupling 側 `inject_laser_source_terms` で実施し、
    \(e_e \leftrightarrow T_e\) クロージャにはセルごとの \(A_{eff},\gamma_{eff}\)（§4.3.3）を用いる

#### 4.6.1 Laser トップレベル関数シグネチャ

```cpp
// Laserフルステップ（Strang splitting L(Δt)）— NUMERICS §5
void laser_step(
    State& state,                       // 流体場（ρ,Te,Zbar読取）+ laser_dep 書込
    LaserMesh& lmesh,                   // レーザーメッシュ（内部管理、物理量を再マッピング）
    const Config::LaserConfig& laser,   // ビーム定義、レイ数
    const CellField& zbar,               // 平均電離度 Z̄（IB吸収計算用、1D=CellField1D / 2D=CellField2D）
    double dt,                          // フルステップ幅 Δt [s]
    double t,                           // 現在時刻 [s]（波形評価用）
    const PartitionInfo& part,          // 並列情報（LaserMesh全rank複製）
    cudaStream_t stream
);
```

---

### 4.6a burn/
**責務**：核燃焼（1D_SPH v1 / 2D_RZ port）— Bosch-Hale 反応率、per-cell 種ネットワーク、
2D 局所沈着および Corman 多群荷電粒子拡散、Li-Petrasso e/i 分配表（NUMERICS §14）

- `tenryu_burn`（STATIC、依存は `tenryu_core` のみ；State/Config 非依存の純関数層）
  - `burn_constants.hpp`：種/反応 enum、質量（proton-mass 単位）、エネルギー分配表、
    Bosch-Hale Table VII 係数 — すべて `__host__ __device__` アクセサ関数
    （RDC下で runtime-index の namespace-scope constexpr 配列は device 不可視のため）
  - `reactivity.cuh`：`bosch_hale_sv(reaction, T_keV)`（床・天井クランプ込み）
  - `network.cuh`：`burn_network_step()` — 凍結温度 RK2 subcycle、決定論 scale-back
    正値性、counts 一次主義の台帳恒等（host/device 共有単一実装）
  - `network_gpu.cu/.cuh`：セル並列 device kernel（host-device identity ctest 済み）
  - `deposition.cuh`：`alpha_rho_lambda()`（Fraley 3d×δ_log、媒質係数 `FraleyRangeMedium`）、
    種 range スケール、`point_sphere_deposited_fraction(u,τ)` 閉形式
  - `field_ions.hpp`：減速の場イオン `FieldIons`（イオンあたりの平均 A・Z・Z²・Z²/A）、
    燃焼在庫と材料の体積分率からのセル毎の組成 `cell_field_ions()`（host）、
    局所 range の媒質係数 `fraley_range_medium()`、Corman/MC カーネル用のセル配列 `FieldIonCells`
  - `partition.hpp/.cpp`（host）/ `partition_device.cuh`：LP 減速積分の初期化時タブレーション
    （\(T_e\) 64 × \(T_i\) 16 × \(n_e\) 16 の log 格子 × 10 slot — 荷電生成物 6 と中性子の弾性反跳 4）、Fraley Eq.4 knob。表の構築は、減速項を
    エネルギーごとに 1 回評価して各軸で使い回し（電子の減速は Ti に、イオンの速度項は
    ne によらない）、生成物 slot × Te の作業を優先度の低いスレッド群で並列に計算する。
    driver は初期化時に `std::shared_future` で背景構築を始め、燃焼の段が最初に表を
    要するときに待つ（局所沈着は燃料が反応しうるまで段を飛ばす）。1 項ずつの定義と
    ビット一致を `tests/burn/test_burn_partition_table.cpp` で検査（2026-09-25）
  - `burn_stage.hpp/.cpp`：`compute_burn_step_1d()` — 燃料域/column 幾何、
    ネットワーク呼び出し、per-cell 沈着/分配、台帳（プレーン配列 in/out、単体テスト可能）。既定では
    `burn_stage_gpu.cu/.cuh` の `compute_burn_step_1d_device_stage` が同じ段を GPU で実行し、host の実装は
    `TENRYU_BURN_HOST_STAGE=1` のときと参照・試験用（host/device の一致は `test_burn_stage_gpu_parity`、rel 1e-12）
  - `corman_diffusion.cu/.cuh`：1D の多群荷電粒子拡散（`Burn.scheme="diffusion"`、NUMERICS §14.7）
  - `mc_transport.cu/.cuh`：直線 CSDA の MC α 輸送（`Burn.scheme="mc"`、NUMERICS §14.9）
  - `neutron_heating.hpp/.cpp`・`neutron_heating_device.cuh`：中性子の最初の衝突による加熱（`Burn.neutron_heating`、NUMERICS §14.11）
  - `neutron_moments.hpp/.cpp`：Brysk の中性子スペクトルのモーメント（診断、NUMERICS §14.8）
  - `screening.hpp/.cpp`・`screening_device.cuh`：Salpeter / Chugunov–DeWitt の遮蔽（NUMERICS §14.1）
- driver 結線（coupling/ 所有）：`callbacks.burn`（laser 直後・radiation 前）、
  比在庫 `State::burn_n_host` [1/g]、`inject_burn_source_terms`（source_terms.cu、
  2T 再閉包）、dt lineage "burn"、budget `E_burn_in`
- テスト：`tests/burn/`（reactivity anchors / network closed-forms / deposition
  kernel / stage synthetic spheres / rung-2 6-run 実 run ctest）
  - `burn_stage_2d.hpp/.cpp`（host）：`compute_burn_step_2d()` — 2D_RZ per-cell
    ネットワーク、局所沈着/分配、台帳（プレーン配列 in/out、単体テスト可能）
  - `corman_diffusion.cuh`：次元共通 Corman 係数・群・出生 binning
  - `corman_diffusion_2d.cu/.cuh`：2D RZ 5 点 FV assembly、Jacobi-CG、
    Post-Wilson/Milne 境界、群 cascade と逃逸台帳
- driver 結線（coupling/ 所有）：`callbacks.burn`（laser 直後・radiation 前）、
  比在庫 `State::burn_n_host` [1/g] と `burn_Yg` [1/g]、`inject_burn_source_terms`
  （source_terms.cu、2T/per-material 再閉包）、dt lineage "burn"、budget `E_burn_in`
- テスト：`tests/burn/`（reactivity / network / screening / deposition / 2D stage /
  `burn_net0_2d` / `burn_remap_2d` / Corman 2D）

### 4.7 coupling/
**責務**：演算子分割、dt制御、ソース項統合

- `Coupling::Driver`
  - `DriverRecloseContext` (`driver_reclose.hpp/.cu`) also owns device snapshots
    of conduction-entry Te/cv. Its 1D temperature and energy-increment helpers
    reuse `HydroEOSContext` raw EOS views and preserve each old helper's field
    write set and high-temperature tail. Device table availability is checked
    before selecting snapshots; unavailable tables retain the old host path.
    `TENRYU_CONDUCTION_EOS_HOST=1` selects the original host EOS handoff.
  - dt = min(hydro, cond, rad, user, output)（NUMERICS §2.2）。成長制限 ≤1.2×dt^n は別途適用。レーザーは独立Δt制約を持たない（hydroサブステップに包含、NUMERICS §2.2(d)）
  - **マルチrank同期**：各rank がローカル dt を計算後、`MPI_Allreduce(MPI_MIN)` でグローバル最小 dt を全rankで共有（NUMERICS §2.2）。成長制限 1.2×dt^n はグローバル dt に対して適用する
  - **Hydro開始温度チェック**（NUMERICS §2.1.1）：
    - State に `int8_t* hydro_active`（セル単位の一方向フラグ、`Field<>`はdouble専用のため生ポインタ管理）を保持
    - 初期化時に `hydro_active[c] = (T_start_eV == 0.0)` で設定（§8 ステップ9a）
    - 各ステップ冒頭で非活性セルのみ `T_e[c] >= T_start_eV` を判定（活性セルはスキップ）
    - 非活性セルは圧力・人工粘性の力寄与をゼロ化、ノードは隣接セルのOR論理で移動判定
    - 全セル非活性時：Δt_hydro を除外
    - MPI並列時：ハロー交換でゴーストセルの `hydro_active` を交換（NUMERICS §12.2.2）
  - **Strang splitting 演算子順序**（NUMERICS §2.1）：
    ```
    L(Δt) → H(Δt/2) → C(Δt) → R(Δt) → H(Δt/2)
    ```
    演算子間同期：同一 compute_stream 上で逐次起動。ハロー交換は comm_stream、cudaEvent で依存管理。
    - H = Hydro（Lagrangian step + BC）— セル単位で `hydro_active[c]` に基づき力を計算。ALE は **2回目の H(Δt/2) 後にのみ** 条件付きで実行（2D_RZ: NUMERICS §3.3、1D_SPH: NUMERICS §3.4）
    - C = Conduction（Spitzer-Härm 電子/イオン熱伝導。Q_ei e-i緩和はHydro Corrector内で適用、NUMERICS §1.1.3）
    - L = Laser（ray trace + deposition。ステップ先頭で full-step を1回だけ適用）
    - R = Radiation（`Rad::RadiationStep::step`：FLD または \(S_N\) が輻射場と物質（`Te`, `ee`, `Pe`）を同時に進め、
      `rad_dep`・`rad_emit` を診断として publish する。沈着を後から物質へ注入する段は無い）
  - 1D Hydro の high-k velocity damper の光学的厚さのゲートはセルごとの Rosseland 不透明度の最大値を受け取るが、
    それを与えていたのは退役したモンテカルロ輻射（`IMC::last_sigma_R_max()`）だけで、2026-09-29 から damper は
    このゲートなしで動く（NUMERICS §3.1.4）。
  - 各演算子の前後でハロー交換を挿入（NUMERICS §12.2.3）
  - **演算子間 EOS 再クロージャ**（NUMERICS §2.1、CUDA_KERNELS §9 参照）：各演算子が Te/ee を更新した後、後続演算子向けに EOS 同期を実行する。C(Δt) 後：U2→H13(Te→ee,Pe,Cv)。L(Δt) 後：H14(ee→Te)→H13→U2。R(Δt) 後：H14→H13→U2。
  - `src/coupling/driver_safety_audit.{hpp,cu}` provides the device-side
    post-operator safety audit used by `Driver::run`: Te/Ti/rho/ee/ei
    non-finite flags and finite-Te maximum are reduced on the GPU, then a
    small result is copied to host for the existing `nan_fatal` and overshoot
    decision logic.
  - `src/coupling/driver_fld_energy.{hpp,cu}` provides the device-side FLD/SN
    radiation energy diagnostic used by `Driver::run`: `max(rad_E,0)*vol`
    contributions are formed on the GPU, block-reduced deterministically, and
    accumulated on host from block partials for epsilon-budget inputs.
  - `src/coupling/thermal_subcycle_scan.{hpp,cu}` provides the two device
    scans of the thermal subcycle (NUMERICS §2.1): the compressed floor-hit
    test after each substep and the initial substep-count prediction (minimum
    Te margin above the floor guard). Each returns one value to the host
    instead of copying Te and rho. The attempt backups and substep tallies are
    driver-owned buffers kept across steps.
  - `src/coupling/driver_retry_snapshot.{hpp,cu}` provides the State
    snapshot/restore primitive for driver-level full-step retry on an
    inadmissible hydro corrector.
    Scope is the deterministic radiation modes (FLD/SN).
  - `src/coupling/dispatcher_decision.{hpp,cpp}` provides the pure free-function
    retry dispatcher classifier consumed by `Driver::run` when
    `Numerics.hydro.dispatcher_state_sensitive_bypass_enabled=True`.
  - `src/coupling/profile_observability.{hpp,cpp}` provides the driver-owned
    ICF standard ALE provenance counters/classifier. `Driver::run` resets it at
    run start when `Numerics.profile.icf_standard_ale.enabled=True`, threads a
    nullable pointer through Hydro2D/ALE geometry soft-fail sites, and logs
    `[ale_provenance]` state at run start, fatal-abort, and run end.
    The V22 restart/output contract also defines the production_comparable
    gate structure for per-material conservation: seven criteria, residual-aware status enum, and the
    PASS/PARTIAL-A/PARTIAL-B/INCONCLUSIVE/FAIL/DISABLED classifier used by the
    planned follow-up empirical rerun.
  - When `Numerics.hydro.driver_full_step_retry_enabled=True`, the main driver
    loop wraps each outer step in a retry epoch: capture State at step entry,
    run split operators, accept only if Hydro reports an admissible corrector,
    otherwise restore, halve dt, and retry up to
    `driver_full_step_retry_max_attempts`.  History, snapshot, checkpoint, and
    cumulative energy accounting are below the acceptance boundary, so rejected
    attempts do not publish output.
  - When `Numerics.hydro.driver_retry_active_mesh_repair_enabled=True`, the
    retry epoch also evaluates current corner-J balance before hydro.  Attempt
    0 records diagnostics only; retry attempts with failed balance force the
    existing 2D ALE path via `apply_ale(..., force_rezone=true)`, then
    recompute dt and continue through the ordinary retry acceptance boundary.
    ALE strategy remains the configured 2D ALE mode (`axis_spine_only`,
    `full_winslow`, etc.); the driver adds only the retry-time invocation
    policy.
  - Production-audit infrastructure in `src/coupling/driver.cpp`
    initializes at run start when
    `Numerics.diagnostics.production_audit.enabled=True`, consumes per-step
    escape-valve events, launches the positivity scan, enforces the Tier-A
    termination gate, and emits `audit_summary` output through
    `tools/validation/audit_summary.py`.  The `tenryu_coupling` library now
    PUBLIC links `tenryu_verification` for the shared audit data model.

- `Coupling::SourceTerms`
  - レーザー・燃焼の沈着を e_e へ加える（保存性を保証）
  - `inject_laser_source_terms` は `source_injection` 後の \(e_e \leftrightarrow T_e\) クロージャで
    セルごとの \(A_{eff},\gamma_{eff}\) から構成した \(c_{v,e}\) を用い（NUMERICS §1.1.5a）、cell-local な `laser_dep[c]` を
    そのまま \(e_e\) へ注入し、退化セルでは `E_numerical_loss` へ退避する。燃焼の沈着は `inject_burn_source_terms`。
  - 輻射の沈着は注入しない：FLD・\(S_N\) は物質の更新を自分の Newton の中で行う。モンテカルロ輻射の沈着を注入していた
    `inject_radiation_source_terms`（正味電子ソースの平滑化を含む）は 2026-09-29 に退役し、
    `retired/radiation_monte_carlo/src/coupling/source_terms_radiation.cu` に保管した。
  - **保存性検証**：injection前後の `Σ(ρ·ee·V)` の差と注入量（`Σ laser_dep` 等）の一致を
    Kahan summation で計算し、`|差| / |入力| < 1e-14` を assert

> **注**：電子熱伝導の物理実装（Spitzer-Härm + flux limiter + STS + 負温度防止）は
> `hydro/conduction.*` に配置する（Hydro::Conduction、§4.4）。
> Coupling::Driver が Strang splitting の中で Hydro::Conduction を呼び出す。

---

### 4.8 diagnostics/
**責務**：ICF向け診断量

- `Diag::ArealDensity`：ρR（角度指定の線積分）
- `Diag::Sphericity`：RZのPℓモード、殻半径R(θ)抽出
- `Diag::EnergyBudget`：入射/吸収/流出/系内の収支
- `Diag::LaserPattern`：吸収分布、臨界終了統計、入射角
- `Diag::MCStats`：分散推定、粒子数統計、CI計算
- `Diag::TemperatureMaximumPrinciple`：放射演算子後の温度最大原理違反（`overshoot_count`, `overshoot_max`）の検出・記録
- `corner_collapse_ledger.{cu,hpp}`：`TENRYU_I1B_COLLAPSE_LEDGER` で有効化する read-only の geometry-collapse 診断。
- `Diag::MeshDeformAttribution`（`src/diagnostics/mesh_deform_attribution.{hpp,cuh,cu}`）：default-off の 2D_RZ mesh failure root-cause diagnostics。`Hydro2D::lagrangian_step` invocation ごとに opt-in workspace が start node positions と per-source displacement buffers を所有し、failure 時だけ `mesh_failure_attribution.jsonl` に per-source corner-J degradation を書く。HDF5 schema と `dt_lineage.jsonl` format は変更しない。
- `Diag::MeshDegeneracyForensics`（`src/diagnostics/mesh_degeneracy_forensics.{hpp,cu}`）：default-off の repeated pre-commit `mesh_quality_*` / `in_hydro_*` failure diagnostics。`Hydro2D` は opt-in 時だけ failing cell の4 node position/velocity/acceleration sample を `HydroStepResult` に載せ、`Coupling::Driver` retry path が同一 `(cell, corner, stage)` count と `sigma_safe` threshold を評価して `mesh_degeneracy_forensics.jsonl` へ J(σ), nodal velocity, hourglass amplitude, material/work context を追記する。HDF5 schema と physics state は変更しない。
- `Diag::IcfShellDiagnostics`, `Diag::HotspotGasDiagnostics`, と `Diag::OperatorEnergyResiduals`（`src/diagnostics/diagnostics.{hpp,cu}`, `operator_energy_residuals.{hpp,cu}`）：default-off の ICF shell IFAR/CR、inert gas-hotspot tracer compression metrics、per-operator energy residual。`Coupling::Driver` が history cadence で初期 shell 半径、hotspot tracer state、operator 境界、明示的 `delta_E_ext` を渡し、`HistoryWriter` が `/diagnostics/icf/v1/`, `/diagnostics/hotspot_gas/v1/`, `/diagnostics/conservation/v1/`, `/diagnostics/ale_provenance/v1/` に path-versioned HDF5 series を追記する。HDF5 root `schema_version` は変更しない。
- `Diag::HistoryAppendFile` (`src/diagnostics/history_writer.cpp`): transaction-local
  1D history buffers group already computed rows by dataset and append one
  hyperslab per dataset within the existing flush/copy/rename transaction.
  Dataset handles and owned variable-length strings live until flush; pending
  lengths and last times preserve monotonic rejection and consistency checks.
  Numeric values, attributes, datatypes, shapes and cadence are unchanged.
  Public raw-HDF5 writers and 2D retain immediate appends;
  `TENRYU_HISTORY_BATCH_WRITES=0` selects that path for 1D bisection.
  Since 2026-09-25 the batches are written by one worker thread of the
  writer: the first row of a run is written synchronously (it creates the
  file), later batches (64 rows or 30 s) are queued to the worker, and
  `flush_pending()` (outputs, run end, restart) queues the pending rows and
  waits for the queue to drain; the destructor drains and joins. HDF5 may be
  built without thread safety, so every HDF5 call of the time loop (history
  worker and appenders, snapshot and checkpoint writers, checkpoint and TMAT
  readers, escape-valve and positivity histories) runs under the
  process-wide `core::hdf5_mutex()` (`src/core/hdf5_mutex.hpp`, a recursive
  mutex).
- `Diag::CornerBCAudit`（`src/diagnostics/history_writer.cpp`）：`dt_breakdown_history_enabled=True` の history writer が、CFL winner が r_outer-reflect ∩ z_top-state_supply corner halo に入った step だけ `/diagnostics/corner_bc_audit/v1/` へ interior/ghost state と local dt/cs/Qvisc を追記する diagnostic-only HDF5 group。physics state と HDF5 root `schema_version` は変更しない。
- `Diag::EscapeValveHistory` (`src/diagnostics/escape_valve_history.{hpp,cpp}`):
  Fixed-column HDF5 writer for
  `/diagnostics/escape_valve_audit/v1`.
- `Diag::PositivityHistory` (`src/diagnostics/positivity_history.{hpp,cpp}`):
  Fixed-field HDF5 writer for `/diagnostics/positivity/v1`.
- `Diag::RadialFourierAudit` (`src/diagnostics/radial_fourier_audit.{hpp,cu}`):
  default-off 2D_RZ per-operator radial-null-mode audit. `Coupling::Driver`
  emits before/after stage samples inside the configured time window. The v1
  CUDA kernel computes direct radial DFT amplitudes for `rho`, `Te`, `Ti`,
  cell `u_r`, cell `u_z`, and total `E_rad`, and `HistoryWriter` appends
  `/diagnostics/radial_fourier_audit/v1/` rows without modifying physics state.
  PR G2-A adds an independent v2 fixed-mode complex-coefficient path gated by
  `per_operator_radial_fourier_complex_enabled`; it records selected hidden
  variables (`M`, `V`, momenta, internal/radiation energies, mesh centers/areas,
  `Q_visc`, `f_Fleck` where available) under
  `/diagnostics/radial_fourier_audit_v2/v1/`. The v2 path allocates scratch and
  launches kernels only when that flag is enabled. `TENRYU_RFA_V2_MODE` provides
  a build-time Heisenbug verification matrix: `OFF` compiles the v2 compute path
  out, `STUB` keeps the API but suppresses v2 launches, `DUMMY_BUFFER` launches
  the fixed-mode kernel into temporary device records while suppressing host
  capture/HDF5 append, and `FULL` preserves PR G2-A output. The
  `audit_heisenbug_4config_smoke` ctest builds those four variants, runs the
  same small I1 configuration, and compares final HDF5 state datasets for
  bit-exact or roundoff-bounded drift.
- `Diag::FldSubstageAudit` (`src/radiation/fld_substage_audit_drain.cpp`,
  `src/radiation/fld_2d_rz_gpu.cu`, `src/diagnostics/history_writer.cpp`):
  default-off FLD-internal radial Fourier substage audit gated by
  `Radiation.multigroup_diffusion.diagnostic_radial_fourier_substage_enabled`.
  FLD appends records to a thread-local batch during the radiation solve,
  `src/coupling/driver.cpp` drains the batch immediately after the radiation
  stage, stamps cycle/time metadata, and `HistoryWriter` appends
  `/diagnostics/fld_substage_audit/v1/` on rank 0 only. The path is additive
  and does not modify physics state or root HDF5 `schema_version`.
- Production-audit diagnostics library invariant: `tenryu_diagnostics` now PUBLIC links
  `tenryu_verification` for the shared audit summary data model.  The schema is
  additive-only: existing `/diagnostics/ale_provenance/v1` output is unchanged,
  and new audit output uses new `/diagnostics/*/v1` paths.

```cpp
// DiagOutput は各ステップ終了時の診断スナップショット。
// 瞬時量はメッシュ/粒子から計算し、累積量は State のメンバーから読み出す（§5.2 State 参照）。
struct DiagOutput {
    // --- 瞬時量（現ステップのメッシュ/粒子状態から計算）---
    double E_kinetic;            // 運動エネルギー [erg]
    double E_internal_e;         // 電子内部エネルギー [erg]
    double E_internal_i;         // イオン内部エネルギー [erg]
    double E_radiation;          // 放射エネルギー [erg] = Σ_c Σ_g rad_E[c,g] V_c（FLD・S_N の場。NUMERICS §10.2）
    double rhoR_avg;             // 面密度 [g/cm²]
    double shell_radius_mean;    // 殻平均半径 [cm]
    double shell_radius_min;     // 殻最小半径 [cm] (ρ > 0.1*ρ_max の最小 r_c)
    double a_2;                  // Legendre mode 2 振幅 [dimensionless]（Pℓ分解、§4.8 Diag::Sphericity）
    double Zbar_mean;            // 質量重み平均電離度 [dimensionless]
    double Zbar_max;             // 最大電離度 [dimensionless]
    double T_e_max;              // 最大電子温度 [eV]
    double T_i_max;              // 最大イオン温度 [eV]
    double rho_max;              // 最大密度 [g/cm³]
    int clamp_count;             // フロアクランプ回数 (per step, reset each step)
    int overshoot_count;         // 最大原理違反セル数 (per step, Radiation直後。history radiation/overshoot_count)
    double overshoot_max;        // 最大超過率 δ_max (per step, Radiation直後。history radiation/overshoot_max)
    // --- 累積量（State のメンバーから読み出し — §5.2 Cumulative diagnostics）---
    double E_laser_deposited;    // = State.E_laser_deposited [erg]
    double E_laser_escaped;      // = State.E_laser_escaped [erg]
    double E_rad_escaped;        // = State.E_rad_escaped [erg]
    double E_floor_injected;     // = State.E_floor_injected [erg]
    double E_safety;             // = State.E_safety [erg]
    double E_numerical_loss;     // = State.E_numerical_loss [erg]
};
DiagOutput compute_diagnostics(
    const Mesh& mesh,
    const State& state,
    const Config& cfg,
    const LaserMesh* laser_mesh,     // nullable（laser無効時は nullptr）
    const EOSTable* eos_e,           // [n_materials] 電子EOS（C_v計算等に使用）
    const double* zbar,              // [n_cells] 現ステップの Z̄ [dimensionless]
    const RadiationResult& rad_result,
    cudaStream_t stream
);
```

---

### 4.9 io/
**責務**：入出力、再始動、メタデータ

- `IO::HDF5Writer`：HDF5 出力（rank 0 だけが書き、MPI-IO は使わない）。snapshot と checkpoint の deflate 付き dataset のうち 64 KiB 以上は、プロセス共通の圧縮スレッド群（`DeflatePool`。スレッド数は使える CPU 数 − 2、最低 2。使える CPU 数は Linux ではプロセスの affinity と cgroup の CPU 上限で制限する — RunPod の pod はハードウェアスレッド 128・上限 13.6 CPU）が圧縮（zlib `compress2`、HDF5 の deflate フィルタと同じ形式・同じレベル）し、ファイルを閉じる前に `H5Dwrite_chunk` で書く（`DeferredDeflateChunks`、2026-09-25）。512 KiB を超える dataset は第 1 次元に沿って約 256 KiB のチャンクに分けて保存する（端のチャンクは 0 で埋める）。時間ループの snapshot（`OutputManager::write_snapshot`）は `HDF5Writer::write_snapshot_in_background` で書く: ファイルの作成・群・属性・小さい dataset の書き込みと大きい dataset のチャンクの圧縮依頼までを行って返り（データはこの時点で複製済み）、完了（圧縮の待ち・チャンクの書き込み・close・`.tmp` からの rename による公開）は 1 本の完了スレッド（`SnapshotFinisher`）が書いた順に行う（2026-09-25。完了待ちは 2 個まで、超えると書き込み側が待つ。完了スレッドのエラーは次の書き込みか `HDF5Writer::wait_for_snapshot_writes` で再送出され、driver は run の最後の snapshot の後で待つ）。`HDF5Writer::write_snapshot` は同じ書き込みの後、公開まで待ってから返る（戻った時点でファイルがある）。HDF5 の呼び出しはプロセス共通の再帰ロック（`core::hdf5_mutex`）で直列化する（完了スレッドも同じロックの下で書く）。スナップショットの各群は NVTX 区間 `io.snapshot.*`（完了スレッドの close は `io.snapshot.close`）で計測できる
- `IO::Checkpoint`：State + Mesh（schema 2。schema 1 にあった光子粒子 `particles/` と RNG 状態 `rng/` は 2026-09-29 に廃止）
- `IO::Restart`
- `IO::Schema`：互換性ルール（スキーマ破壊禁止）

**IO方式**：
- `write_snapshot`：rank 0 だけが 1 つのファイルに書く（MPI-IO は使わない。1D の MPI 実行では書く前に各 rank の担当区間を rank 0 に集める）
- `write_checkpoint`：rank 0 だけが 1 つのファイル `checkpoints/<case>_ckpt_NNNN.h5` に書く（旧形式のランク別ファイル `_rNNNN.h5` は読み込める）
- `read_checkpoint`：各 rank が同じチェックポイントファイルを全体読み込む
  - **ランク数変更可**：各 rank がファイル全体を読むので、保存時と rank 数が違ってもよい（SPECIFICATION §7.4）
  - 粒子の再配布：`cell_id` から新パーティションの所属 rank を判定し MPI 送信
  - `config_hash` 不一致時：WARNING 出力（凍結パラメータ変更は `ConfigError`、SPECIFICATION §7.4 参照）
  - RNG復元：`curand_init(global_id ^ user_seed, step_number, rng_counter)` で O(1) 復元（rank非依存、NUMERICS §12.7.1 準拠）

```cpp
// === IO モジュール (§4.9) ===
namespace IO {
    // --- HDF5 スキーマバージョニング ---
    // ルートグループ attrs: schema_version = 1（v1.0 初版、SPECIFICATION §7.5準拠）
    // 読込時のバージョン互換ルール：
    //   - schema_version == 現行 → そのまま読込
    //   - schema_version < 現行  → 後方互換リーダーが欠落フィールドに既定値を補完し WARNING 出力
    //   - schema_version > 現行  → ConfigError("checkpoint schema version {v} is newer than code version {current}")
    //   - schema_version 属性が存在しない（v0 以前）→ version=0 として後方互換リーダーを適用
    // バージョン変更条件：HDF5 group/dataset の追加・削除・型変更時にインクリメント
    static constexpr int SCHEMA_VERSION = 1;

    // Snapshot: HDF5 ファイル構造（SPECIFICATION §7.2 準拠）
    // データセット名は State フィールド名と 1:1 対応（名前マッピング不要）
    // 例外: mesh/cell_material_id は State 直接フィールドではなく IO 時に派生生成（SPECIFICATION §7.2）
    // /metadata/          : {namelist_source, frozen_config, group_bounds_eV, schema_version, attrs...}
    // /mesh/x_r           : double[n_nodes]   — 節点R座標 [cm] (State.x_r)
    // /mesh/x_z           : double[n_nodes]   — 節点Z座標 [cm] (2D only, State.x_z)
    // /mesh/v_r           : double[n_nodes]   — 節点R速度 [cm/s] (State.v_r)
    // /mesh/v_z           : double[n_nodes]   — 節点Z速度 [cm/s] (2D only, State.v_z)
    // /hydro/rho          : double[n_cells]   — 質量密度 [g/cm³] (State.rho)
    // /hydro/Te           : double[n_cells]   — 電子温度 [eV] (State.Te)
    // /hydro/Ti           : double[n_cells]   — イオン温度 [eV] (State.Ti)
    // /hydro/ee           : double[n_cells]   — 電子比内部エネルギー [erg/g] (State.ee)
    // /hydro/ei           : double[n_cells]   — イオン比内部エネルギー [erg/g] (State.ei)
    // /hydro/Pe           : double[n_cells]   — 電子圧力 [dyne/cm²] (State.Pe)
    // /hydro/Pi           : double[n_cells]   — イオン圧力 [dyne/cm²] (State.Pi)
    // /hydro/Qvisc        : double[n_cells]   — 人工粘性圧 [dyne/cm²] (State.Qvisc)
    // /hydro/mass         : double[n_cells]   — セル質量 [g] (State.mass)
    // /hydro/vol          : double[n_cells]   — セル体積 [cm³] (State.vol)
    // /hydro/zbar         : double[n_cells]   — 平均電離度 Z̄ [-] (State.zbar)
    // /hydro/per_material/v1/{mass,Ee,Ei}
    //                      : double[n_cells,n_materials] — per-material conservation
    //                        authoritative extensive per-material state.
    // /diagnostics/conservation/v1/per_material_*_residual
    //                      : scalar residual diagnostics for Σ_m conserved
    //                        state against cell-mean projections.
    // /diagnostics/per_material/v1/*
    //                      : cumulative per-material event counters.
    // /metadata/dispatch_counters/*
    //                      : dispatch counter regression-hash inputs; disabled
    //                        per-material conservation mode persists all-zero per-material counts.
    // /mesh/topology/v2/*  : optional group written only when
    //                        topology_scheme="multiblock_cart_core_polar_shell";
    //                        path-versioned topology extension, no root
    //                        schema_version bump. Sub-datasets:
    //                        cell_block_id[int32,n_cells_total],
    //                        cell_id_stable[int32,n_cells_total],
    //                        cell_node_csr_offsets[int32,n_cells_total+1],
    //                        cell_node_csr_indices[int32,total_corners],
    //                        face_adj_csr_offsets[int32,n_cells_total+1],
    //                        face_adj_csr_indices[int32,total_faces],
    //                        face_bc_tags[int32,total_faces],
    //                        block_counts[int32,7] =
    //                        {block_count,n_cells_core,n_cells_bridge,
    //                         n_cells_shell,n_nodes_core,
    //                         n_nodes_bridge_interior,n_nodes_shell}.
    //                        Readers first probe /mesh/topology/v2. If absent,
    //                        they fall back to v1/single-block reconstruction.
    //                        Unknown newer path versions hard-fail rather than
    //                        changing root schema_version semantics.
    // /mesh/topology/v3/*  : optional group written only when
    //                        topology_scheme="multiblock_half_butterfly_5block";
    //                        variable-block topology extension for the B-S1
    //                        five-block half-butterfly. Readers probe v3,
    //                        then v2, then v1/single-block. Sub-datasets:
    //                        block_count[int32]=5,
    //                        block_id[int32,block_count],
    //                        block_role[int32,block_count],
    //                        block_n_i_cells[int32,block_count],
    //                        block_n_j_cells[int32,block_count],
    //                        block_cell_begin[int32,block_count],
    //                        block_cell_count[int32,block_count],
    //                        block_owned_node_begin[int32,block_count],
    //                        block_owned_node_count[int32,block_count],
    //                        seam_* tables with orientation/index ranges,
    //                        cell_block_id[int32,n_cells_total],
    //                        cell_id_stable[int32,n_cells_total],
    //                        cell_orientation_sign[int32,n_cells_total],
    //                        cell_node_csr_offsets/indices,
    //                        face_adj_csr_offsets/indices,
    //                        face_bc_tags[int32,total_faces].
    //                        The root schema_version remains unchanged.
    // /hydro/eta_compatible : double[n_cells] — optional legacy compatible-volume mismatch [cm³] (State.eta_compatible)
    // /hydro/volFrac      : double[n_cells x n_mat] — 体積分率 (State.volFrac)
    // /radiation/energy_density    : double[n_cells x G] — E_g [erg/cm³]
    // /radiation/rad_dep           : double[n_cells x G] — [erg] (当該ステップ累積)
    // /radiation/rad_emit          : double[n_cells x G] — emission diagnostic [erg]（当該ステップ累積）
    // /radiation/deposited_power   : double[n_cells x G] — [erg/cm³/s] = rad_dep / (V × dt)
    // /radiation/fleck_factor      : double[n_cells] — 2D_RZ FLD Fleck factor f_i [-]
    // /radiation/sn_tau_R, sn_reduced_flux, sn_ap_alpha
    //     : double[n_cells] — 2D_RZ S_N AP transition diagnostics [-]
    // /radiation/diag_rad_E_pre, diag_rad_E_post,
    // /radiation/diag_rad_emission_at_Tn, diag_rad_emission_at_Tnp1,
    // /radiation/diag_rad_absorption, diag_clip_energy, diag_clip_full_deficit,
    // /radiation/diag_chi_opacity, diag_F_first_moment, diag_E_star_flux, diag_stream_theta
    //     : double[n_cells x G] — 1D S_N plateau investigation diagnostics (output-only)
    // /radiation/diag_ap_alpha_face : double[n_faces x G] — AP face_blend weight (output-only)
    // （schema 2（2026-09-29）で /radiation/ddmc_flag・/radiation/delta_E_rad_prev・/holo/*・/difference/* を廃止。
    //  いずれも退役したモンテカルロ輻射の出力。OUTPUT_SCHEMA・SPECIFICATION §7.5）
    //
    // Checkpoint: 上記 + 以下を追加（SPECIFICATION §7.4 準拠）
    // hydro_flags/hydro_active  : int8[n_cells]（SPECIFICATION §7.4 準拠、State は int8_t*）
    // config_hash               : uint64 attrs (frozen config の hash — restart 時検証)
    // time_state/*              : E_laser_deposited, E_laser_escaped, E_rad_escaped,
    //                              E_floor_injected, E_safety, E_numerical_loss,
    //                              E_pdV_bdry, E_Marshak_in, E_solver
    //                            （State フィールド名と一致。SPECIFICATION §7.4 time_state/* 準拠）
    // output_state/t_next_*     : double × 3（SPECIFICATION §7.4 出力タイミング状態）
    // time_state/t              : float64（現在時刻 [s]、State.t と対応）
    // time_state/step           : int32（**最後に完了したステップ番号**、State.step と対応。
    //                              リスタート時は step+1 から実行再開する。燃焼の α 粒子 Monte Carlo は
    //                              step を Philox の subsequence に使うので、off-by-one はストリーム重複を
    //                              引き起こす — この契約を厳守すること）
    // time_state/dt             : float64（タイムステップ幅 [s]、NUMERICS §2.2 Δt成長制限復元用）
    // time_state/ale_last_applied_step : int32（1D V3 ALE の最後の commit step。
    //                              旧checkpointでは -1 で補完）
    // time_state/E_safety            : float64（伝導安全補正累積、State.E_safety と対応）
    // time_state/E_numerical_loss    : float64（退化セル損失累積、State.E_numerical_loss と対応）
    // time_state/E_laser_deposited   : float64（レーザー沈着累積、State.E_laser_deposited と対応）
    // time_state/E_laser_escaped     : float64（レーザー脱出累積、State.E_laser_escaped と対応）
    // time_state/E_rad_escaped       : float64（放射脱出累積、State.E_rad_escaped と対応）
    // time_state/E_floor_injected    : float64（フロア注入累積、State.E_floor_injected と対応）
    // time_state/E_pdV_bdry          : float64（境界PdV仕事累積、State.E_pdV_bdry と対応）
    // time_state/E_Marshak_in        : float64（Marshak入射累積、State.E_Marshak_in と対応）
    // time_state/E_solver            : float64（Hypre残差累積、State.E_solver と対応。v1.0=0）

    void write_snapshot(const std::string& path, const State& state,
                        const Mesh& mesh, const Config& cfg, int step, double time);
    void write_checkpoint(const std::string& path, const State& state,
                          const Mesh& mesh, const Config& cfg, int step, double time);
    State load_checkpoint(const std::string& path, const Config& cfg,
                          const PartitionInfo& part);
    // 出力は rank 0 だけが書く（MPI-IO・collective write は使わない）
    // 大きい dataset のチャンク分割と圧縮は上の IO::HDF5Writer の説明を参照
}
```

---

### 4.10 verification/
**責務**：verification references, audit policy data models, and validation-facing summaries

- Existing analytic/reference modules remain in `src/verification/`:
  `marshak.cu`, `sedov_analytic.cpp`, `noh_analytic.cpp`, and `laser_analytic.cu`
  (`diffusion_ref.cpp`, the diffusion reference of the retired DDMC gates, moved to
  `retired/radiation_monte_carlo/` on 2026-09-29).
- `Verification::TierThreshold` (`src/verification/tier_threshold.{hpp,cu}`):
  The 9-tier threshold framework.
- `Verification::AuditSummary` (`src/verification/audit_summary.{hpp,cpp}`):
  shared audit summary data model and JSON serialization.
- `Verification::EscapeValveAudit`
  (`src/verification/escape_valve_audit.{hpp,cpp}`): 6-flag escape-valve
  counter and Tier-A/Tier-B policy.
- `Verification::PositivityTracker`
  (`src/verification/positivity_tracker.{hpp,cu}`): per-step positivity scanner
  over six fields.
- The `tenryu_verification` library now PUBLIC links `tenryu_parallel` for the
  Reduction primitive used by production-audit scanners.
- Architectural invariants from commits `cc62bad5`, `f56f8612`,
  `547d894c`, and `f08646b8`:
  - all infrastructure is default OFF, preserving the bit-exact baseline when
    `production_audit.enabled=false`;
  - frozen-config schema advances V22 to V23 additively;
  - Tier-A verification requires zero escape-valve firings, while Tier-B
    engineering uses the 6-condition escape-valve policy.

### 4.11 tools/validation/
**責務**：production-audit validation postprocessing

- `tools/validation/audit_summary.py` assembles `<case>.audit.json` from
  `run_info.json`, history HDF5, and reproducibility metadata.
- `tools/validation/reproducibility_check.py` evaluates same-architecture byte
  reproducibility, cross-architecture tolerance, and emits
  `cross_arch_metadata.json`.

---
