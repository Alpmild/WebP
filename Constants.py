import numpy as np

N = 4

CAT_FORM = '>BI'
RLE_CAT_FORM = '>IBI'
SEQ_LEN_FORM = '>I'
INFO_FORM = '>HHBBBB'
f = 0

modes = ['RGB', 'L', '1', 'RGBA']

# DCT_MAT = np.array([
#     [1, 1, 1, 1],
#     [2, 1, -1, -2],
#     [1, -1, -1, 1],
#     [1, -2, 2, -1]
# ])
#
# DCT_MAT_INV = np.linalg.inv(DCT_MAT)

Y_QUANT = np.array([
    [6, 4, 4, 6],
    [4, 3, 3, 4],
    [4, 3, 3, 4],
    [6, 4, 4, 6]
])

UV_QUANT = np.array([
    [10, 8, 8, 10],
    [8, 6, 6, 8],
    [8, 6, 6, 8],
    [10, 8, 8, 10]
])
