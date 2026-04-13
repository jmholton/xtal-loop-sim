"""
crystal-harvester CLI

Builds a loop_sim scene YAML file from physical parameters.

Usage examples
--------------
Hampton 300 µm CryoLoop, default crystal:
    crystal-harvester --loop-type hampton --loop-size 300 --output scene.yaml

Hampton 500 µm, hexagonal crystal, custom lattice:
    crystal-harvester --loop-type hampton --loop-size 500 \\
        --crystal hexagonal --crystal-dims 0.06 0.03 \\
        --a-axis 10.5 0 0 --b-axis 0 10.5 0 --c-axis 0 0 27.3 \\
        --solvent-volume 0.003 --output hampton_500.yaml

MiTeGen M2-L18SP-200 with plate crystal:
    crystal-harvester --loop-type mitegen --model M2-L18SP-200 \\
        --crystal plate --crystal-dims 0.08 0.02 \\
        --output mitegen_200.yaml

List available MiTeGen models:
    crystal-harvester --list-mitegen
"""

import argparse
import sys
import yaml

from .hampton_loops  import build_hampton_scene, HAMPTON_PRESETS
from .mitegen_mounts import build_mitegen_scene, list_models


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="crystal-harvester",
        description="Build a loop_sim scene YAML from physical parameters.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument("--loop-type", choices=["hampton", "mitegen"],
                   default="hampton",
                   help="Type of sample mounting hardware (default: hampton)")

    # --- Hampton options ---
    hg = p.add_argument_group("Hampton CryoLoop options")
    hg.add_argument("--loop-size", type=int, default=300, metavar="UM",
                    help="Loop diameter in µm (default: 300). "
                         f"Standard sizes: {sorted(HAMPTON_PRESETS)}")
    hg.add_argument("--fiber-diameter", type=float, default=None, metavar="UM",
                    help="Fiber diameter in µm (overrides preset)")
    hg.add_argument("--loop-shape", choices=["teardrop", "oval", "circular"],
                    default=None,
                    help="Loop shape (default: teardrop)")
    hg.add_argument("--stem-length", type=float, default=None, metavar="MM",
                    help="Stem length in mm (default: 5.0)")
    hg.add_argument("--stem-pitch-ratio", type=float, default=None,
                    metavar="RATIO",
                    help="Stem helix pitch / fiber diameter (default: 5.0)")
    hg.add_argument("--youngs-modulus", type=float, default=None, metavar="GPA",
                    help="Young's modulus of nylon in GPa (default: 2.5)")

    # --- MiTeGen options ---
    mg = p.add_argument_group("MiTeGen Micromount options")
    mg.add_argument("--model", default="M2-L18SP-200", metavar="MODEL",
                    help="MiTeGen model number (default: M2-L18SP-200)")
    mg.add_argument("--list-mitegen", action="store_true",
                    help="List available MiTeGen model numbers and exit")

    # --- Pin options ---
    pg = p.add_argument_group("Pin options")
    pg.add_argument("--pin-diameter", type=float, default=None, metavar="MM",
                    help="Pin outer diameter in mm (default: 0.5)")
    pg.add_argument("--pin-length", type=float, default=None, metavar="MM",
                    help="Pin visible length in mm (default: 6.0)")
    pg.add_argument("--pin-bevel", type=float, default=None, metavar="DEG",
                    help="Pin bevel angle in degrees (default: 45)")

    # --- Solvent options ---
    sg = p.add_argument_group("Solvent options")
    sg.add_argument("--solvent-volume", type=float, default=0.002, metavar="MM3",
                    help="Solvent volume in mm³ (default: 0.002)")
    sg.add_argument("--contact-angle", type=float, default=30.0, metavar="DEG",
                    help="Solvent-on-nylon contact angle in degrees (default: 30)")

    # --- Crystal options ---
    cg = p.add_argument_group("Crystal options")
    cg.add_argument("--crystal",
                    choices=["cube", "plate", "needle", "hexagonal", "none"],
                    default="hexagonal",
                    help="Crystal habit preset (default: hexagonal; 'none' omits crystal)")
    cg.add_argument("--crystal-dims", type=float, nargs="+", metavar="MM",
                    default=None,
                    help="Crystal half-widths in mm (preset-dependent). "
                         "cube: 3 values; plate/needle/hexagonal: 2 values")
    cg.add_argument("--a-axis", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    default=[10.5, 0.0, 0.0],
                    help="Real-space a-axis vector in Angstroms (default: 10.5 0 0)")
    cg.add_argument("--b-axis", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    default=[0.0, 10.5, 0.0],
                    help="Real-space b-axis vector in Angstroms (default: 0 10.5 0)")
    cg.add_argument("--c-axis", type=float, nargs=3, metavar=("X", "Y", "Z"),
                    default=[0.0, 0.0, 27.3],
                    help="Real-space c-axis vector in Angstroms (default: 0 0 27.3)")

    # --- Output ---
    p.add_argument("--output", "-o", default="scene.yaml", metavar="FILE",
                   help="Output YAML file path (default: scene.yaml)")

    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)

    if args.list_mitegen:
        print("Available MiTeGen model numbers:")
        for m in list_models():
            print(f"  {m}")
        return 0

    crystal_preset = None if args.crystal == "none" else args.crystal
    lattice_abc = {
        "a_axis": args.a_axis,
        "b_axis": args.b_axis,
        "c_axis": args.c_axis,
    }

    # Collect pin kwargs (only pass if specified)
    pin_kwargs = {}
    if args.pin_diameter is not None:
        pin_kwargs["pin_diameter_mm"] = args.pin_diameter
    if args.pin_length is not None:
        pin_kwargs["pin_length_mm"] = args.pin_length
    if args.pin_bevel is not None:
        pin_kwargs["pin_bevel_deg"] = args.pin_bevel

    if args.loop_type == "hampton":
        hampton_kwargs = {}
        if args.fiber_diameter is not None:
            hampton_kwargs["fiber_diameter_um"] = args.fiber_diameter
        if args.loop_shape is not None:
            hampton_kwargs["loop_shape"] = args.loop_shape
        if args.stem_length is not None:
            hampton_kwargs["stem_length_mm"] = args.stem_length
        if args.stem_pitch_ratio is not None:
            hampton_kwargs["stem_pitch_ratio"] = args.stem_pitch_ratio
        if args.youngs_modulus is not None:
            hampton_kwargs["youngs_modulus_gpa"] = args.youngs_modulus

        scene_dict = build_hampton_scene(
            loop_diameter_um   = args.loop_size,
            solvent_volume_mm3 = args.solvent_volume,
            contact_angle_deg  = args.contact_angle,
            crystal_preset     = crystal_preset,
            crystal_dims_mm    = args.crystal_dims,
            lattice_abc        = lattice_abc,
            **hampton_kwargs,
            **pin_kwargs,
        )

    else:  # mitegen
        scene_dict = build_mitegen_scene(
            model           = args.model,
            crystal_preset  = crystal_preset,
            crystal_dims_mm = args.crystal_dims,
            lattice_abc     = lattice_abc,
            **pin_kwargs,
        )

    with open(args.output, "w") as f:
        yaml.dump(scene_dict, f, default_flow_style=False, sort_keys=False)

    print(f"Scene written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
