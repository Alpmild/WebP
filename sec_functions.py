import numpy as np

from Constants import *


def c_dct(u):
    if isinstance(u, int):
        return 2 ** -0.5 if u == 0 else 1.0
    u = np.asarray(u)
    return np.where(u == 0, 2 ** -0.5, 1.0)


def category(delta):
    abs_val = abs(delta)
    if abs_val == 0:
        return 0
    return int(np.floor(np.log2(abs_val))) + 1


def convert(x, cat):
    if cat == 0:
        return ""
    if x < 0:
        x = ((1 << cat) - 1) ^ (abs(x))
    return bin(x)[2:].zfill(cat)


def iconvert(bits, cat):
    if cat == 0:
        return 0
    value = int(bits, 2)
    if bits[0] == '0':
        value = -((1 << cat) - 1 - value)
    return value


def quant_matrix(qmatrix, quality):
    if quality <= 0:
        quality = 0.01
    elif quality > 100:
        quality = 100

    if quality >= 50:
        scale = 1 - (quality - 50) / 50  # Диапазон 1.0 (50) → 0.5 (100)
    else:
        scale = 2 - quality / 50

    scaled_matrix = ((qmatrix * scale + 50) / 100).astype(np.uint8)
    scaled_matrix[scaled_matrix == 0] = 1
    return scaled_matrix


def hor_block(matrix: np.array, i, j, n):
    if j == 0:
        val = 128 if i == 0 else np.mean(matrix[i - n: i, n - 1])
        pred = np.full((n, n), val)
    else:
        col = matrix[i:i + n, j - 1:j]
        pred = np.tile(col, (1, n))
    return pred


def ver_block(matrix: np.array, i, j, n):
    if i == 0:
        val = 128 if j == 0 else np.mean(matrix[n - 1, j - n:j])
        pred = np.full((n, n), val)
    else:
        row = matrix[i - 1, j:j + n]
        pred = np.tile(row, (n, 1))
    return pred


def dc_block(matrix: np.array, i, j, n):
    val = 4
    val += 128 * 4 if i == 0 else sum(matrix[i - 1, j:j + n])
    val += 128 * 4 if j == 0 else sum(matrix[i:i + n, j - 1])

    return np.full((n, n), val // 8)


def tm_block(matrix: np.array, i, j, n):
    block = np.zeros((n, n))
    row = [128] * n if i == 0 else matrix[i - 1, j:j + n]
    col = [128] * n if j == 0 else matrix[i:i + n, j - 1]
    val = 128 if i == 0 or j == 0 else matrix[i - 1, j - 1]

    for y in range(n):
        for x in range(n):
            h_el = col[y] if x == 0 else matrix[i + y, j + x - 1]
            v_el = row[x] if y == 0 else matrix[i + y - 1, j + x]

            block[y, x] = val + (h_el - val) + (v_el - val)
    return block.astype(int)


def convert(matrix: np.array, pred_func, mode, n):
    """
    mode:
        0 - матрица разностей
        1 - восстановленная матрица
    """
    assert mode == 0 or mode == 1

    h, w = matrix.shape
    res_matrix = np.zeros((h, w)).astype(int)

    for i in range(0, h, n):
        for j in range(0, w, n):
            block = matrix[i:i + n, j:j + n]
            if mode == 0:
                res_block = block - pred_func(matrix, i, j, n)
            else:
                res_block = block + pred_func(res_matrix, i, j, n)
            res_matrix[i:i + n, j:j + n] = res_block

    return res_matrix


a = np.array([[i * j for j in range(1, 13)] for i in range(1, 13)])
res = convert(a, tm_block, 0, micro_N)
print(a, res, convert(res, tm_block, 1, micro_N), sep='\n' * 2)
