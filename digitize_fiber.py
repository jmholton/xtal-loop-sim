#!/usr/bin/env python3
"""
digitize_fiber.py  —  click on a microscope image to define the fiber path.

Usage:
    python3 digitize_fiber.py IMAGE [--pixel-size UM] [--output PATH]

    --pixel-size  µm per pixel (default: 0.8233 for real_loop_himag.jpg)
    --output      YAML or JSON output path (default: prints to stdout)

Controls:
    Left-click   Add a waypoint at the cursor
    Right-click  Remove the last waypoint
    d            Delete the origin and start over
    o            Set/reset the origin (next left-click becomes the origin)
    Enter / q    Finish — print waypoints and exit
    Escape       Quit without saving
    z            Zoom 2× around cursor (scroll wheel also works)

The first left-click (or 'o' then left-click) sets the ORIGIN — the stem
attachment point.  All subsequent clicks are recorded relative to that origin,
converted to mm.

Output waypoints have  z = 0  (fiber lies in the image plane).  To add a
3-D component (e.g. tilt), edit the z column afterwards.
"""

import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
import yaml


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('image', help='Input image file')
    p.add_argument('--pixel-size', type=float, default=0.8233,
                   help='µm per pixel (default: 0.8233)')
    p.add_argument('--output', default=None,
                   help='Output file (.yaml or .json).  Default: stdout.')
    p.add_argument('--diameter', type=float, default=0.020,
                   help='Fiber diameter in mm (default: 0.020).  Written to output.')
    return p.parse_args()


def main():
    args = parse_args()
    px_um = args.pixel_size        # µm / pixel
    px_mm = px_um / 1000.0        # mm / pixel

    img = np.asarray(Image.open(args.image).convert('RGB'))
    H, W = img.shape[:2]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, origin='upper')
    ax.set_title(
        f'{args.image}  |  {px_um:.4f} µm/px\n'
        'Left-click: add point  |  Right-click: undo  |  o: set origin  |  Enter/q: done  |  Esc: quit',
        fontsize=9)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)

    state = {
        'origin_px': None,       # (col, row) of origin in image pixels
        'set_origin_next': True, # first click sets origin
        'waypoints_px': [],      # list of (col, row) relative to origin (image pixels)
        'dots': [],              # matplotlib artists
        'line': None,
    }

    origin_marker = [None]

    def update_line():
        if state['line'] is not None:
            state['line'].remove()
            state['line'] = None
        if state['origin_px'] and state['waypoints_px']:
            ox, oy = state['origin_px']
            xs = [ox] + [ox + p[0] for p in state['waypoints_px']]
            ys = [oy] + [oy + p[1] for p in state['waypoints_px']]
            state['line'], = ax.plot(xs, ys, 'c-', lw=1, alpha=0.7)
        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is not ax: return
        col, row = event.xdata, event.ydata

        if event.button == 1:  # left-click
            if state['set_origin_next'] or state['origin_px'] is None:
                # Set origin
                state['origin_px'] = (col, row)
                state['set_origin_next'] = False
                if origin_marker[0] is not None:
                    origin_marker[0].remove()
                origin_marker[0] = ax.plot(col, row, 'g+', ms=14, mew=2)[0]
                # Clear existing waypoints when origin changes
                for d in state['dots']: d.remove()
                state['dots'].clear()
                state['waypoints_px'].clear()
                print(f'Origin set at pixel ({col:.1f}, {row:.1f})', file=sys.stderr)
            else:
                # Add waypoint relative to origin
                ox, oy = state['origin_px']
                state['waypoints_px'].append((col - ox, row - oy))
                dot, = ax.plot(col, row, 'r.', ms=6)
                state['dots'].append(dot)
                n = len(state['waypoints_px'])
                dx_mm = (col - ox) * px_mm
                dy_mm = -(row - oy) * px_mm  # image Y is flipped vs physical Y
                print(f'  point {n:3d}: pixel ({col:.1f},{row:.1f})  →  ({dx_mm:+.4f}, {dy_mm:+.4f}, 0.000) mm', file=sys.stderr)
            update_line()

        elif event.button == 3:  # right-click — undo
            if state['waypoints_px']:
                state['waypoints_px'].pop()
                state['dots'][-1].remove()
                state['dots'].pop()
                update_line()
                print(f'  undone — {len(state["waypoints_px"])} points remain', file=sys.stderr)

    def on_key(event):
        if event.key in ('enter', 'q'):
            finish()
        elif event.key == 'escape':
            print('Cancelled.', file=sys.stderr)
            plt.close(fig)
        elif event.key == 'o':
            state['set_origin_next'] = True
            print('Click to set new origin.', file=sys.stderr)
        elif event.key == 'd':
            state['set_origin_next'] = True
            state['origin_px'] = None
            for d in state['dots']: d.remove()
            state['dots'].clear()
            state['waypoints_px'].clear()
            if origin_marker[0] is not None:
                origin_marker[0].remove(); origin_marker[0] = None
            update_line()
            print('Reset — click to set origin.', file=sys.stderr)

    def finish():
        if not state['waypoints_px']:
            print('No waypoints recorded.', file=sys.stderr)
            plt.close(fig); return

        ox, oy = state['origin_px'] if state['origin_px'] else (0, 0)
        # Origin itself is (0,0,0); cast all values to plain Python float
        # so yaml.dump emits clean decimal numbers, not numpy binary blobs.
        pts = [[0.0, 0.0, 0.0]]
        for (dpx, dpy) in state['waypoints_px']:
            x_mm = float(round(float(dpx) * px_mm, 6))
            y_mm = float(round(-float(dpy) * px_mm, 6))  # flip Y
            pts.append([x_mm, y_mm, 0.0])
        pts.append([0.0, 0.0, 0.0])  # close the loop back to origin

        result = {
            'pixel_size_um': px_um,
            'image': {'width': W, 'height': H},
            'origin_px': {'col': round(float(ox), 2), 'row': round(float(oy), 2)},
            'fiber': {
                'diameter_mm': float(args.diameter),
                'waypoints': pts,
            },
        }

        out_str = yaml.dump(result, default_flow_style=None, sort_keys=False)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(out_str)
            print(f'\nWrote {len(pts)} waypoints to {args.output}', file=sys.stderr)
            print(f'Next: python3 add_stem.py {args.output}', file=sys.stderr)
        else:
            sys.stdout.write(out_str)

        plt.close(fig)

    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)

    print(f'Image: {args.image}  ({W}×{H} px,  {px_um} µm/px)', file=sys.stderr)
    print('Left-click to set origin, then click along the fiber path.', file=sys.stderr)
    print('Right-click to undo last point.  Enter or q to finish.', file=sys.stderr)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
