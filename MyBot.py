from halite import connect, update, send
from geometry import *

from itertools import groupby

from scipy.ndimage.filters import laplace #, gaussian_filter1d, convolve1d
from scipy.signal import convolve2d


def conv(X, F):
    return convolve2d(X, F, mode='same', boundary='wrap')


## Connect bot and retrieve initial state
pid, shape, P, O, S = connect('ForceFieldBot7')


## Init computation
max_turn = int(10 * (shape[0] * shape[1])**0.5)
R, Ux, Uy = make_geom(shape)

attack = {k: False for k in range(1, np.max(O) + 1) if k != pid}

A = np.array(
    [[0.25, 0.5, 0.25],
     [0.50, 1.0, 0.50],
     [0.25, 0.5, 0.25]]
)

Q = np.ones((3, 3))

frame = 0
while True:
    O, S = update(shape)
    moves = []
    frame += 1

    background = (O == 0)
    owned = (O == pid)
    enemy = ~background & ~owned

    attack = {
        k: v | np.any(owned * conv(O == k, Q))
        for k, v in attack.items()
    }

    sm = (S * owned).sum() / owned.sum()
    pm = (P * owned).sum() / owned.sum()
    t = (S - sm) / (pm + 1e-3)
    t = np.maximum(t, 1)
    a = t + (S + pm) / (P + 1e-6)

    f = sum([conv(O == k, A) for k, v in attack.items() if v])
    f = f + (1 + P / (pm + 1e-3)) * background
    phi = 1 / a * f

    s = 2
    Fx = conv(phi, Ux / R**s)
    Fy = conv(phi, Uy / R**s)


    territory = list(zip(*owned.nonzero()))
    for p in territory:
        move, secondary_move, alpha = dir2move(Fx[p], Fy[p])
        adj = adjacents(p, shape)
        target, secondary_target = adj[move], adj[secondary_move]

        if ~owned[target] and S[target] >= S[p]:
            if (~owned[secondary_target] and S[secondary_target] < S[p]
                and (P[secondary_target] > P[target] or sum(attack.values()) > 0)
                and alpha < 0.9):
                move, secondary_move = secondary_move, move
                target, secondary_target = secondary_target, target
            else:
                continue

        if owned[target] and S[p] < 5 * P[p]:
            continue

        moves.append((p, target, move))

    S2 = S.copy()
    S2[~owned] *= -1
    final_moves = []
    for p, t, m in moves:
        if S2[t] + S[p] < 256+24:
            final_moves.append((p, m))
            S2[t] += S[p]
            S2[p] -= S[p]

    send(final_moves)
