import os
import runpy
from pathlib import Path

os.environ.setdefault("TENRYU_C1_MODE", "zeldovich_raizer")

runpy.run_path(
    str(Path(__file__).with_name("2d_rz_c1_conduction.py")),
    run_name="__main__",
)
