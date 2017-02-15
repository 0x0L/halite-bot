import numpy as np
from halite import STILL, NORTH, EAST, SOUTH, WEST


def dir2move(dx, dy):
    s = abs(dx) + abs(dy)
    if s == 0:
        return STILL, STILL, 0.0
    c = abs(dx) / s
    mx = WEST if dx < 0 else EAST
    my = NORTH if dy < 0 else SOUTH
    return (mx, my, c) if c > 0.5 else (my, mx, 1-c)


def neighborhood(M, point, n=1):
    y, x = point
    return M.take(range(y-n, y+n+1), mode='wrap', axis=0) \
            .take(range(x-n, x+n+1), mode='wrap', axis=1)


def adjacents(point, shape):
    y, x = point
    h, w = shape
    north = (y - 1) % h, x
    south = (y + 1) % h, x
    west = y, (x - 1) % w
    east = y, (x + 1) % w
    return [point, north, east, south, west]


def _transport(field, p):
    x, y = p
    return np.roll(np.roll(field, x, axis=1), y, axis=0)


def make_geom(shape):
    h, w = shape

    w = w if w % 2 == 1 else w - 1
    h = h if h % 2 == 1 else h - 1
    X, Y = np.meshgrid(np.arange(w, dtype=int), np.arange(h, dtype=int))

    dX = (X > w / 2) * w - X
    dY = (Y > h / 2) * h - Y

    distances = (dX**2 + dY**2)**0.5
    distances[0, 0] = 1e-6

    # distances = (abs(dX) + abs(dY)).astype(float)
    # distances[0, 0] = 1e-6

    p = (w // 2, h // 2)
    Ux = _transport(dX / distances, p)
    Uy = _transport(dY / distances, p)
    R = _transport(distances, p)
    return R, Ux, Uy
