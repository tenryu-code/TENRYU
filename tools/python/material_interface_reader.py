#!/usr/bin/env python3
"""Stage 30 material-interface HDF5 reader skeleton."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd


class MaterialInterfaceReader:
    """Reader for /diagnostics/material_interface/v1 skeleton data."""

    GROUP_PATH = "/diagnostics/material_interface/v1"

    def __init__(self, h5_file_path: str | Path) -> None:
        self.h5_file_path = Path(h5_file_path)
        self.file = h5py.File(self.h5_file_path, "r")

    def is_plic_enabled(self) -> bool:
        if self.GROUP_PATH not in self.file:
            return False
        group = self.file[self.GROUP_PATH]
        return int(group.attrs.get("plic_enabled", 0)) == 1

    def get_attributes(self) -> dict[str, Any]:
        if self.GROUP_PATH not in self.file:
            return {}
        attrs: dict[str, Any] = {}
        for key, value in self.file[self.GROUP_PATH].attrs.items():
            if isinstance(value, bytes):
                attrs[key] = value.decode("utf-8")
            else:
                attrs[key] = value
        return attrs

    def get_history_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_plic_events_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame()

    def get_class_d_matrix(self) -> np.ndarray:
        return np.zeros((1, 3, 3), dtype=np.int32)

    def close(self) -> None:
        self.file.close()

    def __enter__(self) -> "MaterialInterfaceReader":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_file_path")
    args = parser.parse_args()
    with MaterialInterfaceReader(args.h5_file_path) as reader:
        print(f"is_plic_enabled={reader.is_plic_enabled()}")
        print(f"attributes={reader.get_attributes()}")


if __name__ == "__main__":
    main()
