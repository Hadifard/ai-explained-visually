"""
AI, explained visually: from one multiplication to ChatGPT.

Renders a 720x1280 (vertical) explainer video frame by frame with matplotlib
and pipes the frames straight into ffmpeg (H.264).
"""

# ==============================================================================
# Author  : Hadi Sarhangi Fard  |  GitHub: @Hadifard
# ==============================================================================

import os, sys, subprocess, textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
from matplotlib.collections import LineCollection

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'DejaVu Sans'

W, H, FPS = 720, 1280, 30
BG = '#050a0e'
BLUE, ORANGE, TEAL, GOLD, GREEN = '#5aa9ff', '#ff9a4d', '#35d0c8', '#f2c14e', '#5fd38d'
PURPLE, WHITE, MUTED = '#b794ff', '#f2f5f7', '#8392a0'
CARD = '#122030'

fig = plt.figure(figsize=(W / 100, H / 100), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.patch.set_alpha(0)

# soft vignette background (built once)
yy, xx = np.mgrid[0:H, 0:W]
d = np.sqrt(((xx - W / 2) / (W * 0.75)) ** 2 + ((yy - H * 0.42) / (H * 0.62)) ** 2)
g = np.clip(1 - d, 0, 1) ** 1.6
base = np.array(matplotlib.colors.to_rgb(BG)); lift = np.array([0.035, 0.075, 0.105])
BGIMG = np.clip(base[None, None, :] + g[..., None] * lift[None, None, :], 0, 1)
rng = np.random.RandomState(7)
PART = rng.rand(46, 4)  # x, y, phase, size
# static background image lives on the figure, so ax.cla() each frame does not redraw it
fig.figimage((BGIMG * 255).astype(np.uint8), 0, 0, origin='upper', zorder=-5)


def setup(T):
    ax.cla()
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis('off')
    fig.patch.set_facecolor(BG)
    px = PART[:, 0] * W + 14 * np.sin(T * 0.5 + PART[:, 2] * 6.28)
    py = PART[:, 1] * H + 18 * np.cos(T * 0.35 + PART[:, 2] * 6.28)
    al = 0.05 + 0.10 * (0.5 + 0.5 * np.sin(T * 0.8 + PART[:, 2] * 9))
    cols = np.zeros((len(px), 4)); cols[:] = matplotlib.colors.to_rgba(TEAL); cols[:, 3] = al
    ax.scatter(px, py, s=4 + PART[:, 3] * 14, c=cols, linewidths=0, zorder=1)


def ease(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def prog(t, t0, t1):
    return ease((t - t0) / (t1 - t0))


def rgba(c, a):
    r = matplotlib.colors.to_rgb(c)
    return (r[0], r[1], r[2], max(0.0, min(1.0, a)))


def text(x, y, s, size=22, color=WHITE, a=1.0, ha='center', va='center', **kw):
    if a <= 0.01:
        return
    ax.text(x, y, s, fontsize=size, color=rgba(color, a), ha=ha, va=va, zorder=6, **kw)


def rich(x, y, parts, size=40, a=1.0):
    if a <= 0.01:
        return
    r = fig.canvas.get_renderer()
    objs, widths = [], []
    for s, c in parts:
        o = ax.text(0, y, s, fontsize=size, color=rgba(c, a), ha='left', va='center', zorder=6)
        widths.append(o.get_window_extent(r).width); objs.append(o)
    cx = x - sum(widths) / 2
    for o, w in zip(objs, widths):
        o.set_x(cx); cx += w


def box(x, y, w, h, ec, fc=CARD, a=1.0, lw=2.4, r=14, ls='-', z=3):
    if a <= 0.01:
        return
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={r}',
                                ec=('none' if ec == 'none' else rgba(ec, a)),
                                fc=('none' if fc == 'none' else rgba(fc, a)), lw=lw, ls=ls, zorder=z))


def node(x, y, r, s, ec, a=1.0, fc=CARD, size=26, tc=WHITE, glow=0.0, lw=3.0):
    if a <= 0.01:
        return
    if glow > 0:
        for m, al in [(2.4, 0.04), (1.9, 0.07), (1.45, 0.12)]:
            ax.add_patch(Circle((x, y), r * m, fc=rgba(ec, a * al * glow), ec='none', zorder=3))
    ax.add_patch(Circle((x, y), r, fc=rgba(fc, a), ec=rgba(ec, a), lw=lw, zorder=4))
    if s:
        text(x, y, s, size=size, color=tc, a=a)


def line(p0, p1, color, a=1.0, lw=3.0, p=1.0, z=2):
    if a <= 0.01 or p <= 0.01:
        return
    x = p0[0] + (p1[0] - p0[0]) * p; y = p0[1] + (p1[1] - p0[1]) * p
    ax.plot([p0[0], x], [p0[1], y], color=rgba(color, a), lw=lw, solid_capstyle='round', zorder=z)


def arrow_head(tip, direction, color, a=1.0, size=13):
    dx, dy = direction; n = np.hypot(dx, dy) + 1e-9; dx, dy = dx / n, dy / n
    tip = np.array(tip); l = np.array([-dy, dx]); b = tip - np.array([dx, dy]) * size
    ax.add_patch(Polygon([tip, b + l * size * 0.5, b - l * size * 0.5], closed=True, fc=rgba(color, a), ec='none', zorder=5))


def curve(p0, p1, lift_, p=1.0, color=GOLD, a=1.0, lw=3.0):
    if p <= 0.01 or a <= 0.01:
        return
    (x0, y0), (x1, y1) = p0, p1
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2 + lift_
    ts = np.linspace(0, p, 40)
    xs = (1 - ts) ** 2 * x0 + 2 * (1 - ts) * ts * cx + ts ** 2 * x1
    ys = (1 - ts) ** 2 * y0 + 2 * (1 - ts) * ts * cy + ts ** 2 * y1
    ax.plot(xs, ys, color=rgba(color, a), lw=lw, solid_capstyle='round', zorder=3)


def dot(x, y, r, color, a=1.0):
    for m, al in [(3.0, 0.07), (2.1, 0.14), (1.4, 0.3)]:
        ax.add_patch(Circle((x, y), r * m, fc=rgba(color, a * al), ec='none', zorder=5))
    ax.add_patch(Circle((x, y), r, fc=rgba(WHITE, a), ec='none', zorder=6))


def step_label(a, s):
    text(W / 2, 78, s, size=17, color=MUTED, a=a)


def caption(t, a, items):
    """items: [(t0, t1, text)] ; card + crossfading lines"""
    best = 0.0
    for (t0, t1, s) in items:
        k = min(1.0, (t - t0) / 0.35, (t1 - t) / 0.35) if t0 <= t <= t1 else 0.0
        k = max(0.0, k)
        best = max(best, k)
    if best <= 0.01:
        return
    box(34, 1010, 652, 190, '#223548', fc='#0c1620', a=a * best * 0.92, lw=1.6, r=26, z=7)
    for (t0, t1, s) in items:
        k = min(1.0, (t - t0) / 0.35, (t1 - t) / 0.35) if t0 <= t <= t1 else 0.0
        k = max(0.0, k)
        if k > 0.01:
            s = '\n'.join(textwrap.fill(p_, 47) for p_ in s.split('\n'))
            ax.text(W / 2, 1105 + (1 - k) * 8, s, fontsize=20.5, color=rgba(WHITE, a * k), ha='center', va='center',
                    zorder=8, linespacing=1.55)


# ------------------------------------------------------------------ scenes
def intro(t, a):
    p = prog(t, 0.2, 1.2)
    for i, (x, y) in enumerate([(150, 330), (360, 300), (570, 330)]):
        node(x, y, 20, '', [BLUE, TEAL, GOLD][i], a=a * p, glow=1.0)
    for i, (x, y) in enumerate([(150, 330), (570, 330)]):
        line((x, y), (360, 300), MUTED, a=a * p * 0.5, lw=2)
    text(W / 2, 520, 'How does AI', size=50, color=WHITE, a=a * prog(t, 0.6, 1.5))
    text(W / 2, 590, 'like ChatGPT work?', size=50, color=GOLD, a=a * prog(t, 0.9, 1.8))
    text(W / 2, 720, 'A simple, visual explanation.', size=26, color=MUTED, a=a * prog(t, 1.8, 2.6))
    text(W / 2, 765, 'We start with one multiplication.', size=26, color=MUTED, a=a * prog(t, 2.4, 3.2))


def scene1(t, a):
    step_label(a, 'STEP 1  ·  ONE TINY NEURON')
    arr = prog(t, 4.2, 4.8)
    parts = [(r'$x$', BLUE), (r'$\,\cdot\,$', WHITE), (r'$w$', ORANGE)]
    if arr > 0:
        parts.append((r'$\,=\,0.4$', WHITE))
    rich(W / 2, 170, parts, size=58, a=a * prog(t, 0, 0.7))
    ix, ox, y = 120, 600, 520
    node(ix, y, 54, '0.8', BLUE, a=a * prog(t, 0.3, 1.0), size=32, glow=1.0)
    text(ix, y + 92, 'input', size=21, color=MUTED, a=a * prog(t, 0.8, 1.4))
    pl = prog(t, 1.0, 2.0)
    line((ix + 54, y), (ox - 48, y), ORANGE, a=a, lw=6, p=pl)
    text(W / 2, y - 60, 'w = 0.5', size=32, color=ORANGE, a=a * prog(t, 1.8, 2.5))
    text(W / 2, y + 62, 'the weight', size=21, color=MUTED, a=a * prog(t, 2.0, 2.7))
    node(ox, y, 48, '0.4' if arr > 0.5 else '', WHITE if arr > 0.5 else '#51606d', a=a * prog(t, 0.6, 1.2), size=30,
         glow=arr)
    text(ox, y + 92, 'output', size=21, color=MUTED, a=a * prog(t, 0.8, 1.4))
    pt = prog(t, 2.4, 4.2)
    if 0 < pt < 1:
        x = ix + 54 + (ox - 48 - ix - 54) * pt
        r = 15 - 7 * ease((pt - 0.5) * 3)
        dot(x, y, max(r, 7), BLUE, a=a)
    caption(t, a, [(0.4, 2.6, 'Inside every AI there are tiny units\ncalled "neurons". Each does very simple math.'),
                   (2.8, 6.4, 'A number comes in and is multiplied by a "weight":\na dial that says how much that input matters.')])


W_IN = [0.3, -0.6, 0.8, 0.5, 0.9]
W_W = [0.8, -0.5, 0.3, 0.7, 0.2]


def scene2(t, a):
    step_label(a, 'STEP 2  ·  MANY INPUTS')
    rich(W / 2, 170, [(r'$\sum_{i}$', WHITE), (r'$\,w_i$', ORANGE), (r'$x_i$', BLUE)], size=60, a=a * prog(t, 0, 0.7))
    ys = [330, 445, 560, 675, 790]; xin, sx, sy = 100, 580, 560
    text(xin, 250, 'inputs', size=21, color=MUTED, a=a * prog(t, 0.3, 1.0))
    for i, y in enumerate(ys):
        pi = prog(t, 0.4 + 0.25 * i, 1.0 + 0.25 * i)
        node(xin, y, 36, f'{W_IN[i]:g}'.replace('-', '\u2212'), BLUE, a=a * pi, size=24, glow=0.6)
        w = W_W[i]
        col = ORANGE if w > 0 else PURPLE
        line((xin + 36, y), (sx - 56, sy), col, a=a * 0.9, lw=2 + 9 * abs(w), p=prog(t, 1.4 + 0.25 * i, 2.2 + 0.25 * i))
        if t > 3.0:
            f = (t * 0.45 + i * 0.21) % 1.0
            x = xin + 36 + (sx - 56 - xin - 36) * f; y2 = y + (sy - y) * f
            dot(x, y2, 5, col, a=a * 0.9)
    node(sx, sy, 56, '\u03a3', GREEN, a=a * prog(t, 1.0, 1.8), size=40, glow=0.8)
    tot = 1.31 * prog(t, 3.6, 4.8)
    if t > 3.6:
        text(sx, sy + 98, f'total  {tot:.2f}', size=28, color=GREEN, a=a)
    # legend
    lx = 150
    line((lx, 900), (lx + 50, 900), ORANGE, a=a * prog(t, 2.5, 3.2), lw=8)
    text(lx + 62, 900, 'pushes the answer up', size=19, color=WHITE, a=a * prog(t, 2.5, 3.2), ha='left')
    line((lx, 940), (lx + 50, 940), PURPLE, a=a * prog(t, 2.8, 3.5), lw=8)
    text(lx + 62, 940, 'pushes the answer down', size=19, color=WHITE, a=a * prog(t, 2.8, 3.5), ha='left')
    caption(t, a, [(0.4, 3.4, 'A real neuron gets many inputs,\neach with its own weight (thicker line = bigger weight).'),
                   (3.5, 6.8, 'It adds everything up\ninto one total score.')])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def scene3(t, a):
    step_label(a, 'STEP 3  ·  THE DECISION')
    rich(W / 2, 170, [(r'$\sigma\,($', TEAL), (r'$\sum$', WHITE), (r'$w_i$', ORANGE), (r'$x_i$', BLUE),
                      (r'$\,+\,b$', GOLD), (r'$)$', TEAL)], size=50, a=a * prog(t, 0, 0.7))
    sx, sy = 280, 340
    pb = prog(t, 1.0, 1.8)
    val = '1.41' if t > 2.2 else '1.31'
    node(sx, sy, 52, val, GREEN, a=a * prog(t, 0.2, 0.9), size=30, glow=0.8)
    text(sx, sy - 82, 'total score', size=20, color=MUTED, a=a * prog(t, 0.5, 1.2))
    node(540, sy, 34, '+0.1', GOLD, a=a * pb, size=20, glow=0.8)
    text(540, sy - 62, 'bias', size=20, color=GOLD, a=a * pb)
    line((506, sy), (332, sy), GOLD, a=a * pb, lw=4, p=prog(t, 1.2, 2.0))
    # sigmoid graph
    gx0, gx1, gy0, gy1 = 130, 590, 470, 760
    pg = prog(t, 2.4, 3.2)
    box(gx0, gy0, gx1 - gx0, gy1 - gy0, '#2a3d50', fc='#0c1620', a=a * pg, lw=1.6, r=12)
    xs = np.linspace(-6, 6, 120)
    ys = sigmoid(xs)
    cx = gx0 + (xs + 6) / 12 * (gx1 - gx0); cy = gy1 - 20 - ys * (gy1 - gy0 - 40)
    k = max(2, int(120 * prog(t, 2.6, 3.8)))
    if pg > 0.05:
        ax.plot(cx[:k], cy[:k], color=rgba(TEAL, a), lw=4, zorder=4)
        text(gx0 + 8, gy0 + 22, '1', size=18, color=MUTED, a=a * pg, ha='left')
        text(gx0 + 8, gy1 - 18, '0', size=18, color=MUTED, a=a * pg, ha='left')
    pp = prog(t, 3.9, 4.7)
    if pp > 0:
        px = gx0 + (1.41 + 6) / 12 * (gx1 - gx0); py = gy1 - 20 - sigmoid(1.41) * (gy1 - gy0 - 40)
        dot(px, py, 8, GREEN, a=a * pp)
    line((sx, sy + 52), (sx, gy0 - 6), MUTED, a=a * prog(t, 2.2, 3.0), lw=3)
    text(sx + 80, (sy + 52 + gy0) / 2, 'squash', size=22, color=TEAL, a=a * prog(t, 2.6, 3.3))
    po = prog(t, 4.8, 5.6)
    node(360, 880, 50, '0.80', GREEN, a=a * po, size=28, glow=1.0)
    text(360, 955, 'the neuron\'s decision', size=21, color=MUTED, a=a * po)
    caption(t, a, [(0.4, 3.0, 'A small extra nudge, the "bias", is added.'),
                   (3.1, 6.8, 'Then the result is squashed into a smooth\nnumber between 0 and 1: the neuron\'s "decision".')])


def scene4(t, a):
    step_label(a, 'STEP 4  ·  A LAYER OF NEURONS')
    rich(W / 2, 170, [(r'$\sigma\,($', TEAL), (r'$W$', ORANGE), (r'$x$', BLUE), (r'$\,+\,b$', GOLD), (r'$)$', TEAL)],
         size=56, a=a * prog(t, 0, 0.7))
    xin, xout = 100, 610
    yi = [400, 500, 600, 700, 800]
    yo = np.linspace(330, 880, 14)
    r = np.random.RandomState(2)
    segs, cols = [], []
    for i in yi:
        for j in yo:
            segs.append([(xin + 28, i), (xout - 14, j)])
            c = ORANGE if r.rand() > 0.4 else PURPLE
            cols.append(rgba(c, 0.16 + 0.35 * r.rand()))
    pl = prog(t, 0.9, 2.4)
    if pl > 0.01:
        cols2 = [(c[0], c[1], c[2], c[3] * a * pl) for c in cols]
        ax.add_collection(LineCollection(segs, colors=cols2, linewidths=1.5, zorder=2))
    for k, y in enumerate(yi):
        node(xin, y, 28, '', BLUE, a=a * prog(t, 0.2 + 0.1 * k, 0.8 + 0.1 * k), glow=0.7)
    for k, y in enumerate(yo):
        pn = prog(t, 1.4 + 0.07 * k, 1.9 + 0.07 * k)
        pulse = 0.5 + 0.5 * np.sin(t * 3 + k * 1.3) if t > 3.0 else 0.4
        node(xout, y, 14, '', TEAL, a=a * pn, glow=pulse * 1.2, lw=2.5)
    text(xin, 300, 'inputs', size=21, color=MUTED, a=a * prog(t, 0.5, 1.2))
    text(xout, 280, 'one layer', size=24, color=TEAL, a=a * prog(t, 2.0, 2.8))
    caption(t, a, [(0.4, 3.0, 'Put many neurons side by side\nand you get a "layer".'),
                   (3.1, 6.0, 'Every neuron sees the same inputs,\nbut each learns to notice something different.')])


def scene5(t, a):
    step_label(a, 'STEP 5  ·  A NEURAL NETWORK')
    rich(W / 2, 170, [(r'$h_{\ell+1}$', BLUE), (r'$\,=\,\sigma\,($', WHITE), (r'$W_{\ell}$', ORANGE),
                      (r'$h_{\ell}$', BLUE), (r'$\,+\,b_{\ell}$', GOLD), (r'$)$', WHITE)], size=44, a=a * prog(t, 0, 0.7))
    cols_x = [90, 225, 360, 495, 630]
    counts = [5, 9, 9, 9, 3]
    pos = []
    for cx, n in zip(cols_x, counts):
        sp = min(60, 520 / n)
        ys = [600 + (i - (n - 1) / 2) * sp for i in range(n)]
        pos.append([(cx, y) for y in ys])
    r = np.random.RandomState(5)
    segs, cl = [], []
    for c in range(4):
        for p0 in pos[c]:
            for p1 in pos[c + 1]:
                segs.append([p0, p1]); cl.append(0.07 + 0.2 * r.rand())
    pl = prog(t, 0.6, 2.4)
    colr = [rgba(ORANGE if i % 3 else PURPLE, v * a * pl) for i, v in enumerate(cl)]
    ax.add_collection(LineCollection(segs, colors=colr, linewidths=1.2, zorder=2))
    phase = (t * 1.5) % 6.0 if t > 2.0 else -5
    for c, plist in enumerate(pos):
        for k, (x, y) in enumerate(plist):
            pn = prog(t, 0.2 + 0.15 * c, 0.8 + 0.15 * c)
            lit = max(0.0, 1 - abs(phase - c - 0.5) * 0.9)
            col = BLUE if c == 0 else (GREEN if c == 4 else TEAL)
            node(x, y, 18 if c in (0, 4) else 14, '', col, a=a * pn, glow=0.3 + 1.6 * lit, lw=2.5)
    text(90, 295, 'input', size=21, color=MUTED, a=a * prog(t, 0.8, 1.5))
    text(630, 360, 'output', size=21, color=MUTED, a=a * prog(t, 0.8, 1.5))
    text(360, 295, 'hidden layers', size=23, color=TEAL, a=a * prog(t, 1.4, 2.2))
    caption(t, a, [(0.4, 3.4, 'Stack many layers: simple patterns combine\ninto more complex ones. This is a neural network.'),
                   (3.5, 7.2, 'At first its weights are random.\nTraining on huge amounts of text slowly tunes them,\nuntil the answers get good.')])


WORDS = ['The', 'cat', 'sat', 'on', 'the', 'mat']
WX = [86, 190, 292, 383, 470, 575]
ATT = [[1],
       [.5, 1],
       [.15, 1, .45],
       [.1, .2, .9, .55],
       [.1, .1, .15, .75, .5],
       [.1, .2, .2, .7, .55, .95]]


def scene6(t, a):
    step_label(a, 'STEP 6  ·  ATTENTION')
    rich(W / 2, 170, [(r'$\mathrm{softmax}\,($', WHITE), (r'$\frac{QK^{\top}}{\sqrt{d}}$', GOLD), (r'$)\,$', WHITE),
                      (r'$V$', BLUE)], size=48, a=a * prog(t, 4.0, 4.8))
    y = 330
    for i, wd in enumerate(WORDS):
        pw = prog(t, 0.3 + 0.18 * i, 0.8 + 0.18 * i)
        hl = (i in (1, 2)) and 1.6 < t < 4.0
        text(WX[i], y, wd, size=34, color=GOLD if hl else WHITE, a=a * pw)
    # arcs from "sat" (2) to others
    pa = prog(t, 1.6, 2.6)
    curve((WX[2], y - 28), (WX[1], y - 28), -70, p=pa, color=GOLD, a=a, lw=7)
    curve((WX[2], y - 28), (WX[3], y - 28), -45, p=prog(t, 2.0, 2.9), color=GOLD, a=a * 0.55, lw=3.5)
    curve((WX[2], y - 28), (WX[0], y - 28), -95, p=prog(t, 2.2, 3.1), color=GOLD, a=a * 0.28, lw=2)
    text(W / 2, y - 140, 'who sat?  \u2192  the cat', size=24, color=GOLD, a=a * prog(t, 2.4, 3.2) * (1 - prog(t, 3.8, 4.2)))
    n, cell = 6, 54
    x0, y0 = W / 2 - n * cell / 2 + 24, 520
    pr = prog(t, 4.2, 6.8)
    for i in range(n):
        ri = min(max(pr * n - i, 0), 1)
        for j in range(n):
            if j > i:
                ax.add_patch(Rectangle((x0 + j * cell, y0 + i * cell), cell - 4, cell - 4, fc=rgba('#0d151c', a * ri), ec='none', zorder=3))
                continue
            v = ATT[i][j]
            c = np.array(matplotlib.colors.to_rgb(GOLD)) * (0.12 + 0.88 * v) + np.array([0.02, 0.03, 0.04]) * (1 - v)
            ax.add_patch(Rectangle((x0 + j * cell, y0 + i * cell), cell - 4, cell - 4, fc=rgba(c, a * ri), ec='none', zorder=3))
    pl = prog(t, 4.4, 5.4)
    for k, wd in enumerate(WORDS):
        text(x0 + k * cell + cell / 2 - 2, y0 - 20, wd, size=15, color=MUTED, a=a * pl)
        text(x0 - 12, y0 + k * cell + cell / 2 - 2, wd, size=15, color=MUTED, a=a * pl, ha='right')
    caption(t, a, [(0.4, 4.0, 'Words depend on each other.\nTo understand "sat", you must know WHO sat: the cat.'),
                   (4.1, 8.9, 'Attention lets every word look at the others\nand decide which ones matter most.\nBrighter square = stronger link.')])


def poly_pt(pts, f):
    pts = np.array(pts, float)
    seg = np.hypot(*(pts[1:] - pts[:-1]).T)
    d = f * seg.sum(); acc = 0
    for i, s in enumerate(seg):
        if d <= acc + s:
            u = (d - acc) / s
            return pts[i] + (pts[i + 1] - pts[i]) * u
        acc += s
    return pts[-1]


def scene7(t, a):
    step_label(a, 'STEP 7  ·  THE TRANSFORMER BLOCK')
    rich(W / 2, 150, [(r'$h\,=\,$', WHITE), (r'$x$', BLUE), (r'$\,+\,$', WHITE), (r'$\mathrm{Attn}(x)$', GOLD)],
         size=38, a=a * prog(t, 3.4, 4.2))
    rich(W / 2, 212, [(r'$y\,=\,h\,+\,$', WHITE), (r'$\mathrm{MLP}(h)$', TEAL)], size=38, a=a * prog(t, 3.8, 4.6))
    cx = 360
    main = [(cx, 930), (cx, 280)]
    pm = prog(t, 0.2, 1.4)
    line((cx, 930), (cx, 300), '#c8d2da', a=a * 0.55, lw=3, p=pm)
    arrow_head((cx, 288), (0, -1), '#c8d2da', a=a * pm * 0.8)
    text(cx, 962, 'words in', size=20, color=MUTED, a=a * pm)
    text(cx, 262, 'words out', size=20, color=MUTED, a=a * pm)
    # attention box
    pa = prog(t, 1.0, 1.9)
    box(205, 700, 310, 120, GOLD, fc='#241d0c', a=a * pa, lw=3, r=18)
    text(420, 746, 'Attention', size=29, color=GOLD, a=a * pa)
    text(420, 786, 'words talk to each other', size=17, color=WHITE, a=a * pa)
    for i in range(4):
        for j in range(4):
            if j <= i:
                v = 1 if i == j else 0.35 + 0.2 * ((i + j) % 3)
                ax.add_patch(Rectangle((228 + j * 15, 722 + i * 15), 13, 13, fc=rgba(GOLD, a * pa * v), ec='none', zorder=5))
    pm2 = prog(t, 1.6, 2.5)
    box(205, 450, 310, 120, TEAL, fc='#0c2224', a=a * pm2, lw=3, r=18)
    text(420, 496, 'Neural network', size=26, color=TEAL, a=a * pm2)
    text(420, 536, 'each word "thinks"', size=17, color=WHITE, a=a * pm2)
    for k, (x, n) in enumerate([(232, 3), (252, 4), (272, 3)]):
        for q in range(n):
            ax.add_patch(Circle((x, 480 + q * 15 + (4 - n) * 7), 4.5, fc=rgba(TEAL, a * pm2), ec='none', zorder=5))
    # plus nodes and skip lines
    pp = prog(t, 2.4, 3.2)
    node(cx, 650, 20, '+', WHITE, a=a * pp, size=24, lw=2.5)
    node(cx, 390, 20, '+', WHITE, a=a * pp, size=24, lw=2.5)
    sk1 = [(cx, 880), (140, 880), (140, 650), (cx - 20, 650)]
    sk2 = [(cx, 600), (105, 600), (105, 390), (cx - 20, 390)]
    for sk in (sk1, sk2):
        for i in range(len(sk) - 1):
            line(sk[i], sk[i + 1], WHITE, a=a * 0.35 * pp, lw=2.5, p=1)
    text(190, 905, 'skip path', size=17, color=MUTED, a=a * prog(t, 5.2, 5.9))
    # flowing dot
    if t > 3.0:
        f = ((t - 3.0) * 0.28) % 1.0
        x, y = poly_pt([(cx, 930), (cx, 300)], f)
        dot(x, y, 7, GOLD, a=a)
        for sk in (sk1, sk2):
            x, y = poly_pt(sk, ((t - 3.0) * 0.28 + 0.1) % 1.0)
            dot(x, y, 5, WHITE, a=a * 0.8)
    caption(t, a, [(0.4, 3.4, 'A transformer is built from repeated "blocks".\nEach block has two parts.'),
                   (3.5, 7.6, '1) Attention: words talk to each other.\n2) Neural network: each word "thinks" alone.\nSkip paths keep the original information.')])


def scene8(t, a):
    step_label(a, 'STEP 8  ·  STACK THEM')
    sent = ['the', 'cat', 'sat', 'on', 'the']
    for i, wd in enumerate(sent):
        pw = prog(t, 0.2 + 0.1 * i, 0.7 + 0.1 * i)
        box(120 + i * 98, 915, 84, 46, BLUE, a=a * pw, lw=2.2, r=10)
        text(162 + i * 98, 938, wd, size=22, color=WHITE, a=a * pw)
    ys = [790, 665, 540]
    for k, y in enumerate(ys):
        pb = prog(t, 0.6 + 0.35 * k, 1.3 + 0.35 * k)
        lit = max(0.0, 1 - abs(((t - 1.6) * 0.9) % 4 - (k + 0.5)) * 0.9) if t > 1.6 else 0
        box(135, y - 40, 450, 80, GOLD, fc='#241d0c', a=a * pb, lw=2.2 + 1.5 * lit, r=14)
        ax.add_patch(Rectangle((135, y - 40), 225, 80, fc=rgba(GOLD, a * pb * (0.10 + 0.25 * lit)), ec='none', zorder=3))
        ax.add_patch(Rectangle((360, y - 40), 225, 80, fc=rgba(TEAL, a * pb * (0.10 + 0.25 * lit)), ec='none', zorder=3))
        text(247, y, 'Attention', size=21, color=GOLD, a=a * pb)
        text(472, y, 'Neural net', size=21, color=TEAL, a=a * pb)
        if k < 2:
            arrow_head((360, y - 62), (0, -1), MUTED, a=a * pb, size=11)
    arrow_head((360, 848), (0, -1), MUTED, a=a, size=11)
    pd = prog(t, 2.0, 2.8)
    for q in range(3):
        node(360, 445 - q * 22, 4, '', MUTED, a=a * pd, fc=MUTED, lw=1)
    box(135, 330, 450, 80, GREEN, fc='#10261a', a=a * prog(t, 2.4, 3.2), lw=2.4, r=14)
    text(360, 370, 'Block N', size=21, color=GREEN, a=a * prog(t, 2.4, 3.2))
    text(600, 540, '\u00d7 dozens\nof blocks', size=19, color=GOLD, a=a * prog(t, 2.2, 3.0), ha='left', linespacing=1.4)
    if t > 1.0:
        f = ((t - 1.0) * 0.3) % 1.0
        dot(360, 940 - f * 580, 8, WHITE, a=a)
    text(360, 285, 'a deep "understanding" of the sentence', size=22, color=GREEN, a=a * prog(t, 3.8, 4.6))
    caption(t, a, [(0.4, 3.0, 'Now stack dozens of these blocks.'),
                   (3.1, 6.0, 'With every block, the model understands\nthe sentence a little more deeply.')])


def scene9(t, a):
    step_label(a, 'STEP 9  ·  PREDICT THE NEXT WORD')
    rich(W / 2, 170, [(r'$p($', WHITE), (r'$\mathrm{next\ word}$', GREEN), (r'$\,|\,\mathrm{words\ so\ far})$', WHITE)],
         size=44, a=a * prog(t, 0, 0.7))
    sy = 330
    fill = prog(t, 4.2, 5.0)
    base = 'the cat sat on the'
    text(W / 2 - 52, sy, base, size=34, color=WHITE, a=a * prog(t, 0.3, 1.0))
    # blank / filled word
    bx = W / 2 + 160
    if fill < 0.05:
        ax.plot([bx - 50, bx + 50], [sy + 24, sy + 24], color=rgba(GREEN, a * prog(t, 0.5, 1.2)), lw=3, zorder=4)
    else:
        text(bx, sy, 'mat', size=34, color=GREEN, a=a * fill)
    rows = [('floor', 0.08), ('idea', 0.01), ('moon', 0.03), ('mat', 0.86)]
    for k, (lab, p) in enumerate(rows):
        y = 470 + k * 72
        pr = prog(t, 1.2 + 0.2 * k, 2.4 + 0.2 * k)
        win = lab == 'mat'
        text(210, y, lab, size=27, color=GREEN if win else WHITE, a=a * pr, ha='right')
        ww = 340 * p * ease((t - 1.5 - 0.2 * k) / 1.6)
        box(235, y - 20, max(ww, 3), 40, GREEN if win else '#7d8a98', fc=GREEN if win else '#7d8a98', a=a * pr * (1 if win else 0.8), lw=0, r=8)
        if ease((t - 1.5 - 0.2 * k) / 1.6) > 0.9:
            text(235 + 340 * p + 14, y, f'{int(p * 100)}%', size=24, color=WHITE, a=a * pr, ha='left')
    text(W / 2, 800, 'then it repeats for the next word...', size=24, color=MUTED, a=a * prog(t, 5.0, 5.8))
    if t > 5.4:
        for q in range(3):
            node(300 + q * 60, 870, 9, '', GREEN, a=a * (0.3 + 0.7 * (0.5 + 0.5 * np.sin(t * 5 - q * 1.2))), fc=GREEN, lw=1)
    caption(t, a, [(0.4, 3.6, 'Finally, the model scores every possible next word.\nHere, "mat" looks the most likely.'),
                   (3.8, 7.3, 'It picks one, adds it to the sentence,\nand repeats: word by word.\nThat is how a chatbot writes.')])


def scene10(t, a):
    step_label(a, 'THE RESULT')
    box(50, 330, 620, 470, '#26394d', fc='#0b141d', a=a * prog(t, 0, 0.8), lw=2, r=28)
    node(110, 385, 24, '', TEAL, a=a * prog(t, 0.3, 1.0), glow=1.0, lw=3)
    text(150, 385, 'A chatbot like ChatGPT', size=23, color=WHITE, a=a * prog(t, 0.3, 1.0), ha='left')
    pu = prog(t, 0.9, 1.5)
    box(330, 440, 310, 70, BLUE, fc='#16304f', a=a * pu, lw=0, r=22)
    text(485, 475, 'Why is the sky blue?', size=21, color=WHITE, a=a * pu)
    msg = 'Sunlight is made of many colors.\nBlue light scatters the most in the\nair, so the sky looks blue.'
    n = int(max(0, (t - 1.9) * 26)); shown = msg[:n]
    pa = prog(t, 1.7, 2.3)
    box(70, 545, 480, 150, '#26394d', fc='#14202c', a=a * pa, lw=0, r=22)
    text(95, 620, shown + ('|' if int(t * 3) % 2 == 0 and n < len(msg) else ''), size=21, color=WHITE, a=a * pa, ha='left',
         linespacing=1.5)
    caption(t, a, [(0.4, 6.0, 'ChatGPT is this same idea, scaled up:\nbillions of weights, tuned on huge amounts of text.')])


def scene11(t, a):
    p = prog(t, 0.2, 1.2)
    for i, (x, y) in enumerate([(250, 420), (360, 395), (470, 420)]):
        node(x, y, 14, '', [BLUE, TEAL, GOLD][i], a=a * p, glow=1.0)
    line((250, 420), (360, 395), MUTED, a=a * p * 0.5, lw=2); line((470, 420), (360, 395), MUTED, a=a * p * 0.5, lw=2)
    text(W / 2, 540, 'for more projects', size=26, color=MUTED, a=a * prog(t, 0.7, 1.5))
    pb = prog(t, 1.2, 2.0)
    box(W / 2 - 235, 585, 470, 70, TEAL, fc='#0e2a2c', a=a * pb, lw=2, r=35)
    text(W / 2, 620, 'https://github.com/Hadifard', size=22.5, color=WHITE, a=a * pb)


SCENES = [(0.0, 4.5, intro), (4.5, 11.0, scene1), (11.0, 18.0, scene2), (18.0, 25.0, scene3), (25.0, 31.0, scene4),
          (31.0, 38.5, scene5), (38.5, 47.7, scene6), (47.7, 55.7, scene7), (55.7, 61.7, scene8), (61.7, 69.2, scene9),
          (69.2, 75.2, scene10), (75.2, 80.2, scene11)]
DUR = 80.2
FADE = 0.4


def frame(T):
    setup(T)
    for (s, e, fn) in SCENES:
        if s <= T < e or (T >= DUR - 1e-6 and e == DUR):
            t = T - s
            a = min(1.0, t / FADE, (e - T) / FADE if e != DUR else 1.0)
            fn(t, max(a, 0.0))
            break
    # top progress bar
    ax.add_patch(Rectangle((0, 0), W * min(T / DUR, 1), 5, fc=rgba(TEAL, 0.8), ec='none', zorder=9))


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'test':
        for T in [float(x) for x in sys.argv[2:]]:
            frame(T)
            os.makedirs('test_frames', exist_ok=True)
            fig.savefig(os.path.join('test_frames', f'test_{T:05.1f}.png'), facecolor=BG)
    else:
        f0, f1, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
        cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgba', '-s', f'{W}x{H}', '-r', str(FPS),
               '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '17', '-preset', 'medium', out]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for i in range(f0, f1):
            frame(i / FPS)
            fig.canvas.draw()
            pr.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
            if (i - f0) % 120 == 0:
                print(i, f1, flush=True)
        pr.stdin.close(); pr.wait()
        print('done')
