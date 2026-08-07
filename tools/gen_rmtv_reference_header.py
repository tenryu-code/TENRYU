#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "tmp" / "rmtv_reference_table.csv"
OUTPUT = ROOT / "src" / "verification" / "rmtv_reference_table.hpp"


def format_array(name, values):
    lines = [f"inline constexpr std::array<double, kNRows> {name} = {{"]
    for i in range(0, len(values), 4):
        chunk = values[i:i + 4]
        lines.append("    " + ", ".join(f"{value:.12e}" for value in chunk) + ",")
    lines.append("};")
    return "\n".join(lines)


def main():
    rows = []
    with INPUT.open(newline="") as fh:
        for line in fh:
            if line.lstrip().startswith("#") or not line.strip():
                continue
            rows.append([float(value) for value in next(csv.reader([line]))])

    xi = [row[0] for row in rows]
    u = [row[1] for row in rows]
    theta = [row[2] for row in rows]
    g = [row[3] for row in rows]
    w = [row[4] for row in rows]

    text = f"""#pragma once
#include <array>
namespace tenryu::verification::rmtv {{
inline constexpr double kBeta0 = 7.197533115e7;
inline constexpr double kAlpha = 9.0 / 13.0;
inline constexpr double kKappaRho = -19.0 / 9.0;   // ambient rho exponent
inline constexpr double kGamma = 1.25;
inline constexpr double kXiShock = 1.0;
inline constexpr double kXiFront = 2.0;
inline constexpr double kXiCertifiedLo = 0.3405;   // ODE-certified band low edge
inline constexpr double kTheta0 = 0.113203;        // center T plateau constant
inline constexpr double kIEnergy = 12.916834;
// dimensional instance (tmp/rmtv_gate_params.py, round-trip beta0 exact):
inline constexpr double kGammaGas = 1.915767e12;   // (1+Z) kB/(A mp) erg/(g eV), Z=1 A=1
inline constexpr double kG0 = 1.0e-3;              // g cm^{{-3-kappa}}
inline constexpr double kZeta = 8.5e4;             // cm/s^alpha
inline constexpr double kE0 = 1.077204e12;         // erg
inline constexpr double kChi0 = 1.402400e7;        // test-kappa hook value
inline constexpr std::size_t kNRows = {len(rows)};
{format_array("kXi", xi)}
{format_array("kU", u)}
{format_array("kTheta", theta)}
{format_array("kG", g)}
{format_array("kW", w)}
}}  // namespace tenryu::verification::rmtv
"""
    OUTPUT.write_text(text)


if __name__ == "__main__":
    main()
