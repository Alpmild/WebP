import heapq
from Constants import *
from random import randint


class Huffman:
    class Node:
        def __init__(self, value=None, freq=0, left=None, right=None):
            self.value = value
            self.freq = freq
            self.left = left
            self.right = right

        def __lt__(self, other):
            return self.freq < other.freq

    @classmethod
    def build_tree(cls, freq_dict: dict):
        nodes = [cls.Node(value=i, freq=freq_dict[i]) for i in sorted(freq_dict.keys()) if freq_dict[i]]
        # nodes.sort(key=lambda x: x.value)
        # nodes.sort(key=lambda x: x.freq)
        heapq.heapify(nodes)

        while len(nodes) != 1:
            left = heapq.heappop(nodes)
            right = heapq.heappop(nodes)
            heapq.heappush(nodes, cls.Node(value=None, freq=left.freq + right.freq, left=left, right=right))
        return heapq.heappop(nodes)

    @classmethod
    def build_code(cls, node, prefix="", code_dict=None):
        if code_dict is None:
            code_dict = dict()
        if node:
            if node.value is not None:
                code_dict[node.value] = prefix
            else:
                cls.build_code(node.left, prefix + '0', code_dict)
                cls.build_code(node.right, prefix + '1', code_dict)
        return code_dict


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


def convert_num(x, cat):
    if cat == 0:
        return ""
    if x < 0:
        x = ((1 << cat) - 1) ^ (abs(x))
    return bin(x)[2:].zfill(cat)


def iconvert_num(bits, cat):
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

    if quality < 50:
        scale = 5000 / quality
    else:
        scale = 200 - quality * 2

    scaled_matrix = ((qmatrix * scale + 50) / 100).astype(np.uint8)
    scaled_matrix[scaled_matrix == 0] = 1
    return scaled_matrix


def h_pred(matrix: np.array, i, j, n):
    if j == 0:
        pred = np.full((n, n), 128)
    else:
        col = matrix[i:i + n, j - 1:j]
        pred = np.tile(col, (1, n))
    return pred.astype(int)


def hup_pred(matrix: np.array, i, j, n):
    block = np.zeros((n, n))
    if j == 0:
        val = 128
        col = np.array((val,) * 2 * n)
    else:
        col = list(matrix[i:i + n, j - 1])
        col += [col[-1]] * n

    k = 2 * n - 1
    for y in range(n):
        for x in range(n):
            pos = x + y + 1
            value = (col[pos] + 2 * col[min(pos + 1, k)] + col[min(pos + 2, k)] + 2) // 4
            block[y, x] = np.clip(value, 0, 255)
    return block


def v_pred(matrix: np.array, i, j, n):
    if i == 0:
        val = 128
        pred = np.full((n, n), val)
    else:
        row = matrix[i - 1, j:j + n]
        pred = np.tile(row, (n, 1))
    return pred


def vleft_pred(matrix: np.array, i, j, n):
    block = np.zeros((n, n))
    if i == 0:
        val = 128
        row = np.array((val,) * 2 * n)
    else:
        row = list(matrix[i - 1, j:j + 2 * n])
        if len(row) != 2 * n:
            row += [row[-1]] * (2 * n - len(row))

    k = 2 * n - 1
    for y in range(n):
        for x in range(n):
            pos = x + y + 1
            val = (row[pos] + 2 * row[min(pos + 1, k)] + row[min(pos + 2, k)] + 2) // 4
            block[y, x] = np.clip(val, 0, 255)
    return block


def dc_pred(matrix: np.array, i, j, n):
    matrix = matrix.copy().astype(int)

    val = 4
    val += 128 * 4 if i == 0 else sum(matrix[i - 1, j:j + n])
    val += 128 * 4 if j == 0 else sum(matrix[i:i + n, j - 1])

    return np.full((n, n), val // 8)


def tm_pred(matrix: np.array, i, j, n):
    block = np.zeros((n, n))
    row = [128] * n if i == 0 else matrix[i - 1, j:j + n].astype(int)
    col = [128] * n if j == 0 else matrix[i:i + n, j - 1].astype(int)
    val = 128 if i == 0 or j == 0 else int(matrix[i - 1, j - 1])

    for y in range(n):
        for x in range(n):
            block[y, x] = col[y] + row[x] - val
    return block.astype(int)


def d45_pred(matrix: np.array, i, j, n):
    block = np.zeros((n, n))

    k = 2 * n
    row = list(matrix[i - 1, j:j + k] if i != 0 else (128,) * k)
    if len(row) != k:
        row = [row[0]] * (k - len(row)) + row

    for y in range(n):
        for x in range(n):
            block[y, x] = row[x + y + 1]
    return block


PREDICTIONS = dict(H=h_pred, V=v_pred, DC=dc_pred, TM=tm_pred, HUP=hup_pred, VLEFT=vleft_pred, D45=d45_pred)
PREDS_KEYS = tuple(PREDICTIONS.keys())

def convert(matrix: np.array, pred, mode, n):
    assert mode == 0 or mode == 1

    pred_func = PREDICTIONS[pred]
    h, w = matrix.shape
    matrix = matrix.astype(int)
    res_matrix = np.zeros((h, w)).astype(int)

    size = 4 * n
    for i in range(0, h, size):
        for j in range(0, w, size):
            block_16x16 = matrix[i:i + size, j: j + size]
            block_h, block_w = block_16x16.shape

            res_block_16x16 = np.zeros((block_h, block_w)).astype(int)
            for k in range(0, block_h, n):
                for m in range(0, block_w, n):
                    block_4x4 = block_16x16[k:k + n, m: m + n]
                    if mode == 0:
                        res_block_4x4 = block_4x4 - pred_func(block_16x16, k, m, n)
                    else:
                        res_block_4x4 = block_4x4 + pred_func(res_block_16x16, k, m, n)
                    res_block_16x16[k:k + n, m:m + n] = res_block_4x4

            res_matrix[i:i + block_h, j: j + block_w] = res_block_16x16

    return res_matrix


# m = 'TM'
# # mat = np.array([[205, 242, 128, 68, 241, 196, 136, 194, 53, 64, 40, 182],
# #                 [84, 169, 191, 101, 82, 189, 52, 196, 146, 81, 185, 5],
# #                 [45, 253, 63, 83, 4, 178, 130, 179, 65, 45, 215, 163],
# #                 [121, 21, 249, 243, 113, 111, 189, 159, 2, 167, 152, 31],
# #                 [17, 1, 119, 199, 2, 105, 136, 97, 74, 109, 64, 244],
# #                 [82, 38, 31, 94, 248, 37, 71, 44, 6, 191, 149, 123],
# #                 [0, 78, 51, 251, 233, 76, 29, 156, 110, 172, 33, 87],
# #                 [104, 87, 148, 227, 71, 18, 212, 163, 32, 209, 118, 67]])
#
# mat = np.array([[randint(0, 255) for j in range(16)] for i in range(12)])
# print(mat, '\n')
#
# r1 = convert(mat, m, 0, N)
# print(r1, '\n')
#
# r2 = convert(r1, m, 1, N)
# print(r2, '\n')
#
# print(mat == r2)
