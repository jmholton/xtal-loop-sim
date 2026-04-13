#!/usr/bin/env python3
"""
Assemble a complete scene.yaml from component files and a scene template.

Each component YAML (loop.yaml, crystal.yaml, droplet.yaml, ...) contributes
an 'objects:' list.  The first component that has a 'motor:' key supplies the
motor offset; later ones are ignored.  Camera, geometry, beam, and materials
come from the template.

Usage:
    python3 generate_scene.py loop.yaml [crystal.yaml] [droplet.yaml] ...
                              [--template template.yaml]
                              [--output scene.yaml]

Then render with:
    python3 render.py scene.yaml

Component YAML format (motor is optional in all but the first):
    motor:
      tx: -0.00017
      ty: -0.02345
    objects:
      - name: loop
        material: nylon
        shape: {type: tube, ...}
      - name: crystal
        ...

Template YAML format (camera / geometry / beam / materials — no objects, no motor):
    geometry: ...
    camera:   ...
    beam:     ...
    materials: ...
"""
import sys, os, argparse
sys.path.insert(0, '/home/jamesh/projects/loop_sim/claude')

import yaml


_DEFAULT_TEMPLATE = os.path.join(os.path.dirname(__file__), 'template.yaml')
_DEFAULT_OUTPUT   = os.path.join(os.path.dirname(__file__), 'scene.yaml')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('components', nargs='+',
                   help='Component YAML files (loop.yaml, crystal.yaml, ...)')
    p.add_argument('--template', default=_DEFAULT_TEMPLATE,
                   help=f'Scene template YAML (default: {_DEFAULT_TEMPLATE})')
    p.add_argument('--output', default=_DEFAULT_OUTPUT,
                   help=f'Output scene YAML (default: {_DEFAULT_OUTPUT})')
    return p.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Load template (camera / geometry / beam / materials)
    # ------------------------------------------------------------------
    with open(args.template) as f:
        scene = yaml.safe_load(f)

    for key in ('objects', 'motor'):
        if key in scene:
            print(f"WARNING: template '{args.template}' has '{key}' key — ignoring.",
                  file=sys.stderr)
            scene.pop(key)

    # ------------------------------------------------------------------
    # Merge component files
    # ------------------------------------------------------------------
    motor_source = None
    all_objects  = []

    for path in args.components:
        with open(path) as f:
            comp = yaml.safe_load(f)

        if 'motor' in comp and motor_source is None:
            scene['motor'] = comp['motor']
            motor_source   = path

        objects = comp.get('objects', [])
        all_objects.extend(objects)
        print(f"  {path}: {len(objects)} object(s)"
              + (f"  motor from here" if path == motor_source else ""),
              file=sys.stderr)

    if motor_source is None:
        print("WARNING: no component supplied a 'motor:' key — defaulting tx=ty=0.",
              file=sys.stderr)
        scene['motor'] = {'tx': 0.0, 'ty': 0.0}

    scene['objects'] = all_objects

    # ------------------------------------------------------------------
    # Write scene.yaml
    # ------------------------------------------------------------------
    with open(args.output, 'w') as f:
        yaml.dump(scene, f, default_flow_style=None, sort_keys=False)

    names = [o.get('name', '?') for o in all_objects]
    print(f"Scene:  {len(all_objects)} objects: {names}", file=sys.stderr)
    print(f"Scene → {args.output}", file=sys.stderr)
    print(f"Render: python3 render.py {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
