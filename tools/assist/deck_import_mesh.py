"""Mesh sample coordinates for the binary-free deck comparison.

Graded ladders follow src/mesh/mesh.cu::build_graded_nodes. These coordinates
supplement dense domain cuts; they do not reconstruct multiblock topology,
autozoning, Voronoi tessellation, or polar-in-box morph/collar cells.
"""

import math


def sampling_note(blocks):
    mesh = blocks.get("Mesh", {})
    limitations = []
    if "auto_regions" in mesh or "zoning_intent" in mesh:
        limitations.append("solver autozoning nodes")
    logical = mesh.get("logical_mesh_2d", "rectangular_rz")
    if logical != "rectangular_rz":
        limitations.append(logical+" final topology/cell centroids")
    if any(key in mesh for key in ("topology_scheme", "multiblock", "voronoi")):
        limitations.append("topology-specific nodes")
    if limitations:
        return "Spatial comparison includes dense domain cuts and available uniform/graded/explicit ladders; it does not reconstruct " + ", ".join(limitations) + ". Mesh kwargs and source-managed functions are retained."
    return ""


def graded_nodes(segments, grading, dimension=0, pin=False):
    ratio = grading.get("edge_ratio", 0.1)
    sigma = grading.get("sg_sigma", 0.7)
    order = grading.get("sg_order", 4)
    exact = grading.get("mapping") == "exact_measure_v2" and dimension in (1, 2, 3)
    widths = []
    for segment in segments:
        a, b, n = segment["r_start"], segment["r_end"], segment["nr"]
        length = b-a
        weights = [ratio+(1-ratio)*math.exp(-(abs(2*(k+0.5)/n-1)/sigma)**order) for k in range(n)]
        if exact:
            total, cumulative, previous, values = sum(weights), 0.0, a, []
            for k, weight in enumerate(weights):
                cumulative += weight
                measure = a**dimension+cumulative/total*(b**dimension-a**dimension)
                radius = b if k+1 == n else math.copysign(abs(measure)**(1/dimension), measure)
                values.append(radius-previous)
                previous = radius
        elif 1-ratio <= 1e-12:
            values = [length/n]*n
        else:
            reference = length/math.sqrt(n) if a < 1e-12 else 0.0
            values = [weight/((a+length*(k+0.5)/n)**2+reference**2) for k, weight in enumerate(weights)]
            scale = length/sum(values)
            values = [value*scale for value in values]
        widths.append(values)
    targets = [math.sqrt(a[-1]*b[0]) for a,b in zip(widths,widths[1:])]
    nodes = [segments[0]["r_start"]]
    for i, values in enumerate(widths):
        length = segments[i]["r_end"]-segments[i]["r_start"]
        left = targets[i-1] if i else None
        right = targets[i] if i < len(targets) else None
        if len(values) == 1:
            values = [length]
        elif len(values) == 2 and left is not None and right is not None:
            values = [left,length-left]
        elif left is not None or right is not None:
            old = length-(values[0] if left is not None else 0)-(values[-1] if right is not None else 0)
            new = length-(left or 0)-(right or 0)
            scale = new/old
            values = [left if k==0 and left is not None else right if k==len(values)-1 and right is not None else value*scale for k,value in enumerate(values)]
        for width in values:
            nodes.append(nodes[-1]+width)
        if pin:
            nodes[-1] = segments[i]["r_end"]
    nodes[-1] = segments[-1]["r_end"]
    return nodes


def axis_nodes(mesh, axis, lo, hi, count, dimension=0):
    explicit = mesh.get("explicit_nodes" if axis=="r" else "explicit_nodes_"+axis)
    if isinstance(explicit, (list,tuple)) and len(explicit)>=2:
        return list(explicit)
    grid = mesh.get("grid_"+axis, mesh.get("grid", {}) if axis=="r" else {})
    if isinstance(grid, dict) and grid.get("segments"):
        segments, previous = [], lo
        for segment in grid["segments"]:
            end = segment.get(axis+"_end", segment.get("r_end"))
            segments.append({"r_start":segment.get(axis+"_start", previous),"r_end":end,"nr":segment.get("n"+axis, segment.get("nr"))})
            previous = end
        grading = grid.get("grading", mesh.get("grid", {}).get("grading", {}) if isinstance(mesh.get("grid"),dict) else {})
        return graded_nodes(segments, grading, dimension, pin=dimension==0)
    return [lo+(hi-lo)*i/count for i in range(count+1)]


def mesh_points(blocks):
    mesh = blocks.get("Mesh", {})
    is2d = blocks.get("Main", {}).get("dimension") == "2D_RZ"
    dimension = 0 if is2d else 2 if blocks.get("Main",{}).get("dimension")=="1D_CYL" or mesh.get("geometry_1d")=="cylindrical" else 1 if mesh.get("geometry_1d")=="planar" else 3
    nr, nz = max(1,int(mesh.get("nr",300))), max(1,int(mesh.get("nz",64)))
    lo, hi = mesh.get("r_min",0), mesh.get("r_max",0.05)
    radial = axis_nodes(mesh,"r",lo,hi,nr,dimension)
    if not is2d:
        points = [(r,) for r in radial]
        for a,b in zip(radial,radial[1:]):
            points.append(((a+b)/2,))
            points.append((dimension/(dimension+1)*(b**(dimension+1)-a**(dimension+1))/(b**dimension-a**dimension),))
        return points
    logical = mesh.get("logical_mesh_2d","rectangular_rz")
    if logical in ("spherical_polar_halfplane","polar_in_box"):
        smax = mesh.get("spherical_polar_s_max",hi)
        if not mesh.get("explicit_nodes") and not (isinstance(mesh.get("grid"),dict) and mesh["grid"].get("segments")) and not mesh.get("grid_r"):
            kappa = mesh.get("spherical_polar_kappa",0.5) if mesh.get("polar_center_treatment","annular")=="annular" else 0
            radial = [(i+kappa)*smax/(nr+kappa) for i in range(nr+1)]
        theta_min = mesh.get("polar_theta_min",0)
        theta = axis_nodes(mesh,"theta",theta_min,math.pi,nz)
        if mesh.get("polar_equal_mu_zoning"):
            mu = math.cos(theta_min)
            theta = [math.acos(max(-1,min(1,mu-(1+mu)*i/nz))) for i in range(nz+1)]
        center = mesh.get("box_center_z",0) if logical=="polar_in_box" else 0
        vertices = [[(0.0 if t in (0,math.pi) else r*math.sin(t),center+r*math.cos(t)) for t in theta] for r in radial]
        result = [p for row in vertices for p in row]
        for i in range(len(radial)-1):
            for j in range(len(theta)-1):
                corners = [vertices[i][j],vertices[i+1][j],vertices[i+1][j+1],vertices[i][j+1]]
                result.append(tuple(sum(p[k] for p in corners)/4 for k in (0,1)))
        return result
    axial = axis_nodes(mesh,"z",mesh.get("z_min",-hi),mesh.get("z_max",hi),nz)
    result = [(r,z) for r in radial for z in axial]
    result.extend(((a+b)/2,(c+d)/2) for a,b in zip(radial,radial[1:]) for c,d in zip(axial,axial[1:]))
    return result
