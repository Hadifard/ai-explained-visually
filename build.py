#!/usr/bin/env python3

# ==============================================================================
# Author  : Hadi Sarhangi Fard  |  GitHub: @Hadifard
# ==============================================================================

"""Build the video (MP4) and the GIF.

    python build.py                 # full video + GIF into ./output
    python build.py --frames 90     # quick preview: only the first 3 seconds
    python build.py --no-gif        # video only

Requires: Python 3.9+, numpy, matplotlib and ffmpeg on your PATH.
"""
import argparse, os, shutil, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'src'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=os.path.join(HERE, 'output'), help='output folder (default: ./output)')
    ap.add_argument('--name', default='AI_explained_for_beginners', help='base file name')
    ap.add_argument('--frames', type=int, default=None, help='render only the first N frames (preview)')
    ap.add_argument('--no-gif', action='store_true', help='skip the GIF')
    ap.add_argument('--gif-width', type=int, default=400)
    ap.add_argument('--gif-fps', type=int, default=10)
    args = ap.parse_args()

    if shutil.which('ffmpeg') is None:
        sys.exit('ffmpeg was not found on your PATH. Install it first (https://ffmpeg.org).')

    import render
    total = int(render.DUR * render.FPS)
    n = min(args.frames or total, total)
    os.makedirs(args.out, exist_ok=True)
    mp4 = os.path.join(args.out, args.name + '.mp4')
    gif = os.path.join(args.out, args.name + '.gif')

    print(f'Rendering {n} frames ({n / render.FPS:.1f} s) ...')
    subprocess.check_call([sys.executable, os.path.join(HERE, 'src', 'render.py'), 'video', '0', str(n), mp4])

    if not args.no_gif:
        print('Making GIF ...')
        vf = (f'fps={args.gif_fps},scale={args.gif_width}:-1:flags=lanczos,split[a][b];'
              '[a]palettegen=max_colors=256:stats_mode=diff[p];'
              '[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle')
        subprocess.check_call(['ffmpeg', '-y', '-v', 'error', '-i', mp4, '-vf', vf, gif])
    print('Done:', args.out)


if __name__ == '__main__':
    main()
