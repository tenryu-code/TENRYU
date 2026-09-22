"""Initialization-only helpers embedded in imported Studio decks (stdlib only).

Real setters are called once per block unless repeated structured calls must
retain their order. Source evaluation retains Python closures in an isolated namespace.
This is not a security sandbox; it has the same privileges as the deck loader.
"""


def _studio_capture(source, filename):
    import copy
    import os
    import sys
    import types

    names = ("Main", "Mesh", "Materials", "Geometry", "Radiation", "Laser",
             "Numerics", "Output", "Diagnostics", "Parallel", "Burn")
    blocks = {}
    calls = {}
    module = types.ModuleType("tenryu_namelist")

    def setter(name):
        def record(**kwargs):
            snapshot = copy.deepcopy(kwargs)
            calls.setdefault(name, []).append(snapshot)
            blocks.setdefault(name, {}).update(snapshot)
        return record

    for name in names:
        setattr(module, name, setter(name))
    module.Material = lambda **kwargs: kwargs
    module.LaserBeam = lambda **kwargs: kwargs
    module.__all__ = list(names) + ["Material", "LaserBeam"]
    previous = sys.modules.get("tenryu_namelist")
    search_path, argv = sys.path[:], sys.argv[:]
    directory = os.path.dirname(os.path.abspath(filename))
    local_directories = {directory, os.getcwd()}
    # Local helper modules may import the recording setters. Reload them for
    # each evaluation instead of retaining a previous recorder's closures.
    local_modules = {}
    for name, loaded in list(sys.modules.items()):
        path = getattr(loaded, "__file__", None)
        if name != "__main__" and path and os.path.dirname(os.path.abspath(path)) in local_directories:
            local_modules[name] = sys.modules.pop(name)
    namespace = {"__name__": "__main__", "__file__": filename}
    try:
        sys.modules["tenryu_namelist"] = module
        sys.path[:0] = [directory, os.getcwd()]
        sys.argv = [filename]
        exec(compile(source, filename, "exec"), namespace)
    finally:
        sys.path[:] = search_path
        sys.argv[:] = argv
        if previous is None:
            sys.modules.pop("tenryu_namelist", None)
        else:
            sys.modules["tenryu_namelist"] = previous
        for name, loaded in list(sys.modules.items()):
            path = getattr(loaded, "__file__", None)
            if name != "__main__" and path and os.path.dirname(os.path.abspath(path)) in local_directories:
                sys.modules.pop(name, None)
        sys.modules.update(local_modules)
    return blocks, namespace, calls


def _studio_import_prepare(source, filename, rules, active, current_file=None):
    import os
    import tenryu_namelist as real

    if not os.path.isdir(os.path.dirname(filename)) and current_file:
        filename = os.path.join(os.path.dirname(os.path.abspath(current_file)), os.path.basename(filename))
    original, _, calls = _studio_capture(source, filename)
    policies = {tuple(rule["path"]): rule["kind"] for rule in rules}
    activated = [tuple(path) for path in active]
    # Keys only the source wrote (sourceOnly) whose form field was edited: the
    # edit is bound to exactly that path, so the form value replaces the source.
    editable = {tuple(rule["path"]) for rule in rules if rule.get("sourceOnly")} & set(activated)
    called = set()
    missing = object()

    def is_active(path):
        return any(path[:len(prefix)] == prefix for prefix in activated)

    def has_active_child(path):
        return any(prefix[:len(path)] == path for prefix in activated)

    def merge(old, new, path):
        policy = policies.get(path)
        if policy == "passthrough":
            if path in editable and new is not missing:
                return new
            return old
        if policy in ("mapped", "approximated"):
            return new
        if policy == "omitted" and not is_active(path) and not has_active_child(path):
            return missing
        if old is missing:
            if is_active(path):
                return new
            if isinstance(new, dict) and has_active_child(path):
                result = {}
                for key, candidate in new.items():
                    value = merge(missing, candidate, path + (key,))
                    if value is not missing:
                        result[key] = value
                return result if result else missing
            return missing
        if new is missing:
            return old
        if isinstance(old, dict) and isinstance(new, dict):
            result = {}
            for key in dict.fromkeys(list(old) + list(new)):
                value = merge(old.get(key, missing), new.get(key, missing), path + (key,))
                if value is not missing:
                    result[key] = value
            return result
        if isinstance(old, (list, tuple)) and isinstance(new, (list, tuple)) and len(old) == len(new):
            values = [merge(a, b, path + (i,)) for i, (a, b) in enumerate(zip(old, new))]
            return tuple(values) if isinstance(old, tuple) else values
        return new if is_active(path) else old

    def setter(name):
        def apply(**kwargs):
            if name in called:
                raise RuntimeError("Imported deck generated a repeated block: " + name)
            called.add(name)
            if policies.get((name,)) == "passthrough" and len(calls[name]) > 1:
                for arguments in calls[name]:
                    getattr(real, name)(**arguments)
                return
            value = merge(original.get(name, missing), kwargs, (name,))
            if value is not missing:
                getattr(real, name)(**value)
        return apply

    def finish():
        for name in original:
            if name not in called:
                for arguments in calls[name]:
                    getattr(real, name)(**arguments)

    names = ("Main", "Mesh", "Materials", "Geometry", "Radiation", "Laser",
             "Numerics", "Output", "Diagnostics", "Parallel", "Burn")
    result = {name: setter(name) for name in names}
    result.update(Material=real.Material, LaserBeam=real.LaserBeam, _studio_import_finish=finish)
    return result
