#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Tuple, Union, Optional, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import linecache
import sys


# ----------------------------- Core helpers -----------------------------
def thermo_extractor(path, wanted_property=None):
    """
    Read a LAMMPS thermo-style text file.
    If wanted_property is provided, return a 2xN array [time; values] and the header list.
    Otherwise, return stacked columns [time; col2; col3; ...] and the header list.
    """
    header = linecache.getline(path, 1)
    header_list = header.split()
    array_of_header = np.arange(len(np.array(header_list)))

    if wanted_property is not None:
        if wanted_property not in header_list:
            raise ValueError(f"Property '{wanted_property}' not found in header: {header_list}")
        wanted_property_index = header_list.index(wanted_property)
        timestep_data_index = header_list.index('Step')
        wanted_property_data = np.genfromtxt(path, usecols=wanted_property_index, skip_header=2)
        timestep_data = np.genfromtxt(path, usecols=timestep_data_index, skip_header=2)
        wanted_property_data_array = np.vstack((timestep_data, wanted_property_data))
        return wanted_property_data_array, header_list

    timestep_data_index = header_list.index('Step')
    thermostyle_data = np.genfromtxt(path, usecols=timestep_data_index, skip_header=2)

    for ii in array_of_header[1:]:
        thermostyle_collum = np.genfromtxt(path, usecols=ii, skip_header=2)
        thermostyle_data = np.vstack((thermostyle_data, thermostyle_collum))

    return thermostyle_data, header_list

def avg_last_N(x, N):
    """
    Return the average of the last N values of array-like x.
    If fewer than N points are available, average the available tail.
    """
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("Input series is empty.")
    take = min(int(N), x.size)
    return float(np.nanmean(x[-take:]))

def t_cutoff_setter(
    thermo_property: Tuple[np.ndarray, np.ndarray],
    temperature: Union[int, float],
    tolerance: float = 0.02,
    *,
    window: int = 100,
    tol_units: str = "fraction",   # "fraction" (0.02 = 2%) or "percent" (2 = 2%)
    rolling_window: Optional[int] = None,
    graphing: bool = False,
    label: str = "Thermo_property",
    save: Union[bool, str, Path] = False,
    return_details: bool = False,
) -> Union[
    Tuple[np.ndarray, float],
    Tuple[np.ndarray, float, Dict[str, Any]]
]:
    """
    Determine an equilibration cutoff based on cumulative running average (CRA).
    Cutoff = first block boundary (size=window) where CRA change < tolerance.
    """
    timestep_data = np.asarray(thermo_property[0]).flatten()
    property_data = np.asarray(thermo_property[1]).flatten()

    if timestep_data.size != property_data.size:
        raise ValueError("time and property arrays must have the same length")
    n = property_data.size
    if n == 0:
        raise ValueError("empty property series")
    if window <= 0:
        raise ValueError("window must be a positive integer")

    # Short series → return all as equilibrated
    if n < max(3, window + 1):
        cra = np.cumsum(property_data) / np.arange(1, n + 1)
        rw = rolling_window or window
        rolling = pd.Series(property_data).rolling(rw, min_periods=1).mean().to_numpy()
        eq_time = timestep_data
        eq_vals = property_data
        mean_val = float(np.nanmean(eq_vals))
        stderr = float(np.nanstd(eq_vals, ddof=1) / np.sqrt(max(1, eq_vals.size)))
        details = dict(
            converged=False,
            cutoff_index=0,
            cutoff_time=float(eq_time[0]),
            tolerance_used=float(tolerance),
            tol_units=tol_units,
            window=window,
            n_equil=int(eq_vals.size),
            mean=mean_val,
            stderr=stderr,
            cra=cra,
            rolling=rolling,
            pct_changes_times=np.array([], dtype=float),
            pct_changes_vals=np.array([], dtype=float),
        )
        eq_stack = np.vstack([eq_time, eq_vals])
        return (eq_stack, mean_val, details) if return_details else (eq_stack, mean_val)

    # Handle NaNs by interpolation for CRA stability
    p = property_data.astype(float)
    if np.isnan(p).any():
        idx = np.arange(n)
        good = ~np.isnan(p)
        if good.sum() == 0:
            raise ValueError("all property values are NaN")
        p = np.interp(idx, idx[good], p[good])

    # CRA and rolling
    csum = np.cumsum(p)
    cra = csum / np.arange(1, n + 1)
    rw = rolling_window or window
    rolling = pd.Series(p).rolling(rw, min_periods=1).mean().to_numpy()

    # CRA change over block
    cra_front = cra[window:]
    cra_back = cra[:-window]
    denom = np.maximum(np.abs(cra_front), 1e-15)
    frac_change = np.abs(cra_front - cra_back) / denom

    tol_frac = tolerance / 100.0 if tol_units.lower().startswith("per") else float(tolerance)

    # Check block boundaries: window, 2*window, ...
    step_positions = np.arange(window, n, window)
    usable = step_positions - window
    block_changes = frac_change[usable] if usable.size else np.array([], dtype=float)

    tolerance_met = False
    cutoff_idx = 0
    if block_changes.size:
        below = np.where(block_changes < tol_frac)[0]
        if below.size:
            cutoff_idx = (below[0] + 1) * window
            tolerance_met = True
        else:
            cutoff_idx = step_positions[-1]
            tolerance_met = False

    cutoff_idx = int(np.clip(cutoff_idx, 0, n - 1))

    eq_time = timestep_data[cutoff_idx:]
    eq_vals = property_data[cutoff_idx:]
    mean_val = float(np.nanmean(eq_vals))
    stderr = float(np.nanstd(eq_vals, ddof=1) / np.sqrt(max(1, eq_vals.size)))

    if graphing:
        fig, ax1 = plt.subplots()
        ax1.set_xlabel("time [fs]")
        ax1.set_ylabel(f"{label}  T={temperature} K")
        ax1.plot(timestep_data, property_data, label=label)
        ax1.plot(timestep_data, cra, label="Cumulative Running Average")
        ax1.plot(timestep_data, rolling, label=f"Rolling Running Average (w={rw})")
        ax1.axvline(timestep_data[cutoff_idx], linestyle="--", label="Cutoff", color="red")

        ax2 = ax1.twinx()
        ax2.set_ylabel("% Change of CRA (block-to-block)")
        block_times = timestep_data[step_positions] if step_positions.size else np.array([])
        pct_vals = (block_changes * 100.0) if block_changes.size else np.array([])
        if block_times.size:
            ax2.plot(block_times, pct_vals, marker="o", linestyle=":", label="%Δ CRA per block", color="black")

        ax1.legend(loc="upper right")
        ax2.legend(loc="upper left")
        fig.tight_layout()

        if save:
            outpath = Path(save) if isinstance(save, (str, Path)) else Path(f"./{label}_vs_Time_@_T={temperature}_K.png")
            fig.savefig(outpath, dpi=200, bbox_inches="tight")

    details = dict(
        converged=tolerance_met,
        cutoff_index=cutoff_idx,
        cutoff_time=float(timestep_data[cutoff_idx]),
        tolerance_used=float(tolerance),
        tol_units=tol_units,
        window=window,
        n_equil=int(eq_vals.size),
        mean=mean_val,
        stderr=stderr,
        cra=cra,
        rolling=rolling,
        pct_changes_times=(timestep_data[step_positions] if step_positions.size else np.array([], dtype=float)),
        pct_changes_vals=(block_changes * (100.0 if tol_units.startswith("per") else 1.0)),
    )

    equilibrium_production_data = np.vstack([eq_time, eq_vals])
    if return_details:
        return equilibrium_production_data, mean_val, details
    return equilibrium_production_data, mean_val


# ----------------------------- CLI wrapper -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute equilibration cutoff from LAMMPS thermo output and report equilibrated mean."
    )
    p.add_argument("file", type=Path, help="Path to thermo-style text file (header on line 1, data from line 3).")
    p.add_argument("-p", "--property", default="Volume", help="Property name in header (default: Volume).")
    p.add_argument("--avg-last", type=int, default=None,
               help="If provided, skip cutoff and return the average of the last N values.")
    p.add_argument("-T", "--temperature", type=float, required=True,
                   help="Temperature in K (used for labeling/plots).")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="Tolerance for CRA change. If --tol-units fraction, 0.02=2%%; if percent, 2=2%%.")
    p.add_argument("--tol-units", choices=["fraction", "percent"], default="fraction",
                   help="Interpretation of --tolerance (default: fraction).")
    p.add_argument("-w", "--window", type=int, default=100, help="Block/window size for CRA comparison (default: 100).")
    p.add_argument("--rolling-window", type=int, default=None, help="Rolling window for plotting (default: window).")
    p.add_argument("--label", default=None, help='Label for plots (default: "<property>").')
    p.add_argument("--plot", action="store_true", help="Show a diagnostic plot.")
    p.add_argument("--save-plot", type=Path, default=None, help="Path to save the plot (PNG).")
    p.add_argument("--json", type=Path, default=None, help="Write details JSON here.")
    p.add_argument("--print-only", choices=["Lx", "mean", "cutoff_time"], default=None,
                   help="Print only a single scalar for scripting.")
    p.add_argument("--cube-root", action="store_true",
                   help="After mean is computed (assumed volume), print Lx = mean**(1/3).")
    return p.parse_args()


def main():
    args = parse_args()
    path: Path = args.file
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(2)

    label = args.label if args.label is not None else f"{args.property}"

    # Load (time, property) for requested column
    thermo_data, header_list = thermo_extractor(str(path), wanted_property=args.property)

    # If using avg-last mode → skip cutoff and return the average tail value
    if args.avg_last is not None:
        # thermo_data = (time, property)
        time, values = thermo_data[0], thermo_data[1]
        mean_val = avg_last_N(values, args.avg_last)
        Lx = float(mean_val) ** (1.0 / 3.0)
    
        if args.print_only == "Lx":
            print(f"{Lx:.6f}")
            return
        if args.print_only == "mean":
            print(f"{mean_val:.6f}")
            return
    
        print(f"\nFile: {path}")
        print(f"Property: {args.property}")
        print(f"Temperature: {args.temperature} K")
        print(f"Average of last {args.avg_last} points = {mean_val:.6f}")
        if args.cube_root:
            print(f"Lx = mean**(1/3) = {Lx:.6f}")
        return
    
    # Run cutoff
    production, mean_val, info = t_cutoff_setter(
        thermo_data,
        temperature=args.temperature,
        tolerance=args.tolerance,
        tol_units=args.tol_units,
        window=args.window,
        rolling_window=args.rolling_window,
        graphing=(args.plot or args.save_plot is not None),
        label=label,
        save=(str(args.save_plot) if args.save_plot else False),
        return_details=True
    )

    # Optional JSON
    if args.json:
        # Convert numpy arrays in details to lists for JSON
        serializable = {
            **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in info.items()},
            "file": str(path),
            "property": args.property,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        with args.json.open("w") as f:
            json.dump(serializable, f, indent=2)

    # Compute Lx if requested (assumes mean is volume in Å^3)
    Lx = float(mean_val) ** (1.0 / 3.0)

    # Print mode for scripting
    if args.print_only == "Lx":
        print(f"{Lx:.6f}")
        return
    if args.print_only == "mean":
        print(f"{mean_val:.6f}")
        return
    if args.print_only == "cutoff_time":
        print(f"{info['cutoff_time']:.6f}")
        return

    # Human-readable summary
    print(f"\nFile: {path}")
    print(f"Property: {args.property}")
    print(f"Temperature: {args.temperature} K")
    print(f"Window: {args.window}  |  Tolerance: {args.tolerance} ({args.tol_units})")
    print(f"Header: {header_list}")
    print(f"Converged at index = {info['cutoff_index']}  (time = {info['cutoff_time']})")
    print(f"Equilibrated N points: {info['n_equil']}")
    print(f"Mean({args.property}) = {mean_val:.6f}")
    if args.cube_root:
        print(f"Lx = mean**(1/3) = {Lx:.6f}")
    print("")  # trailing newline


if __name__ == "__main__":
    main()
