import sys
import numpy as np

STILL, NORTH, EAST, SOUTH, WEST = range(5)


def _send_string(s):
    sys.stdout.write(s + '\n')
    sys.stdout.flush()


def _get_string():
    return sys.stdin.readline().rstrip('\n')


def _decode_array(string, shape):
    return np.fromstring(string, dtype=int, sep=' ').reshape(shape)


def send(moves):
    _send_string(' '.join(['{} {} {}'.format(p[1], p[0], m) for p, m in moves]))


def update(shape):
    split_string = _get_string().split()

    size = shape[0] * shape[1]
    owners = []
    while len(owners) < size:
        counter = int(split_string.pop(0))
        owner = int(split_string.pop(0))
        owners.extend([owner] * counter)
    assert len(owners) == size
    owner = np.array(owners).reshape(shape)

    assert len(split_string) == size
    strength = _decode_array(' ' .join(split_string), shape)

    return owner, strength


def connect(botname):
    pid = int(_get_string())

    w, h = tuple(map(int, _get_string().split()))
    shape = h, w

    production = _decode_array(_get_string(), shape)
    owner, strength = update(shape)

    _send_string(botname)
    return pid, shape, production, owner, strength
