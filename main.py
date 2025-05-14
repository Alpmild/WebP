from PIL import Image
import struct
from collections import Counter

from Constants import *
from sec_functions import *

class WebPEncoder:
    @staticmethod
    def toYUV(rgb_image: np.array):
        R = rgb_image[:, :, 0].astype(np.float32)
        G = rgb_image[:, :, 1].astype(np.float32)
        B = rgb_image[:, :, 2].astype(np.float32)

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        U = -0.14713 * R - 0.28886 * G + 0.436 * B + 128
        V = 0.615 * R - 0.51499 * G - 0.10001 * B + 128

        yuv_image = np.stack([Y, U, V], axis=-1).clip(0, 255).astype(np.uint8)
        return yuv_image

    @staticmethod
    def downsampling(channel: np.array):
        h, w = channel.shape
        downsampled = np.zeros((h // 2, w // 2))

        for i in range(0, h, 2):
            for j in range(0, w, 2):
                block = channel[i:i + 2, j:j + 2]
                downsampled[i // 2, j // 2] = np.mean(block)

        return downsampled.astype(np.uint8)

    @staticmethod
    def extend_matrix(matrix: np.array, n):
        h, w = matrix.shape
        new_h, new_w = int(np.ceil(h / n) * n), int(np.ceil(w / n) * n)
        if new_h == h and new_w == w:
            return matrix

        new_matrix = np.zeros((new_h, new_w))
        new_matrix[:h, :w] = matrix
        if new_w != w:
            new_matrix[:h, w:] = matrix[:, -1:]
        if new_h != h:
            new_matrix[h:] = new_matrix[h - 1]
        return new_matrix

    @staticmethod
    def split_blocks(image_arr: np.array, n):
        h, w, *_ = image_arr.shape
        if h % n != 0 or w % n != 0:
            new_h = n * (h // n + bool(h % n))
            new_w = n * (w // n + bool(w % n))

            new_image = np.full((new_h, new_w), 0, dtype=object)
            new_image[:h, :w] = image_arr

            image_arr = new_image
            h, w = new_h, new_w

        blocks_array = np.array([image_arr[i:i + n, j: j + n] for i in range(0, h, n) for j in range(0, w, n)])
        return blocks_array

    @staticmethod
    def dct(block: np.ndarray, n):
        def F(u, v):
            x = np.arange(n)
            y = np.arange(n)

            cos1 = np.cos((2 * x[:, None] + 1) * u * np.pi / (2 * n))
            cos2 = np.cos((2 * y[None, :] + 1) * v * np.pi / (2 * n))

            cos_matrix = cos1 @ cos2
            res = np.sum(block * cos_matrix)

            res *= c_dct(u) * c_dct(v) * 2 / n
            return res

        return np.array([[F(i, j) for j in range(n)] for i in range(n)])

    @staticmethod
    def quantize_block(block, qmatrix):
        return np.round(block / qmatrix).astype(np.int32)

    @staticmethod
    def zigzag(matrix):
        n = len(matrix)
        result = []
        i, j = 0, 0

        for _ in range(n * n):
            result.append(matrix[i][j])

            if (i + j) % 2 == 0:  # Движение вверх-вправо
                if j == n - 1:
                    i += 1
                elif i == 0:
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:  # Движение вниз-влево
                if i == n - 1:
                    j += 1
                elif j == 0:
                    i += 1
                else:
                    i += 1
                    j -= 1

        return np.array(result)

    @staticmethod
    def encode_dc(arr: np.array, file):
        n = len(arr)
        arr = tuple(map(int, arr))

        arr_dc = (arr[0],) + tuple(map(lambda i: arr[i] - arr[i - 1], range(1, n)))
        categories = tuple(map(category, arr_dc))
        freq_dict = dict(Counter(categories))

        # Запись количества категорий
        file.write(struct.pack('>H', len(freq_dict)))
        # Запись словаря частотностей
        for i in sorted(freq_dict.keys()):
            file.write(struct.pack(CAT_FORM, i, freq_dict[i]))

        root = Huf.build_tree(freq_dict)
        codes = Huf.build_code(root)

        huf_str = ""
        encoded = bytearray()
        for i in range(n):
            dc = arr_dc[i]
            cat = category(dc)

            huf_str += codes[cat] + convert(dc, cat)
            while len(huf_str) >= 8:
                encoded.append(int(huf_str[:8], 2))
                huf_str = huf_str[8:]

        padding = 0
        if len(huf_str) != 0:
            padding = 8 - len(huf_str)
            encoded.append(int(huf_str.ljust(8, '0'), 2))

        # Запись длины байтового потока
        file.write(struct.pack(SEQ_LEN_FORM, len(encoded)))
        # Байтовый поток
        file.write(bytes(encoded))
        # Число нулей, добавленных в конец
        file.write(struct.pack('>B', padding))

    @staticmethod
    def encode_ac(arr: np.array, file):
        n = len(arr)
        arr = tuple(map(int, arr))
        arr_ac = (arr[0],) + tuple(map(lambda i: arr[i] - arr[i - 1], range(1, n)))
        ac, rle_ac = [], []

        zeros_cnt = 0
        for i in range(n):
            if arr_ac[i] != 0:
                ac.append(arr_ac[i])
                rle_ac.append((zeros_cnt, category(arr_ac[i])))
                zeros_cnt = 0
            else:
                zeros_cnt += 1
        if zeros_cnt != 0:
            ac.append(0)
            rle_ac.append((zeros_cnt, 0))

        freq_dict = dict(Counter(rle_ac))
        # Запись количества пар
        file.write(struct.pack('>H', len(freq_dict)))
        # Запись словаря частотностей
        for couple, value in sorted(freq_dict.items()):
            file.write(struct.pack(RLE_CAT_FORM, *couple, value))

        root = Huf.build_tree(freq_dict)
        codes = Huf.build_code(root)

        huf_str = ""
        encoded = bytearray()
        for i in range(len(rle_ac)):
            cat = rle_ac[i][1]
            huf_str += codes[rle_ac[i]] + convert(ac[i], cat)

            while len(huf_str) >= 8:
                encoded.append(int(huf_str[:8], 2))
                huf_str = huf_str[8:]

        padding = 0
        if len(huf_str) != 0:
            padding = 8 - len(huf_str)
            encoded.append(int(huf_str.ljust(8, '0'), 2))

        # Запись длины байтового потока
        file.write(struct.pack(SEQ_LEN_FORM, len(encoded)))
        # Байтовый поток
        file.write(bytes(encoded))
        # Число нулей, добавленных в конец
        file.write(struct.pack('>B', padding))


class WebPDecoder:
    @staticmethod
    def from_YUV(yuv_image: np.array):
        Y = yuv_image[:, :, 0].astype(np.float32)
        U = yuv_image[:, :, 1].astype(np.float32) - 128
        V = yuv_image[:, :, 2].astype(np.float32) - 128

        R = Y + 1.13983 * V
        G = Y - 0.39465 * U - 0.58060 * V
        B = Y + 2.03211 * U

        rgb_image = np.stack([R, G, B], axis=-1).clip(0, 255).astype(np.uint8)
        return rgb_image

    @staticmethod
    def idownsampling(channel: np.array):
        h, w = channel.shape
        restored = np.zeros((h * 2, w * 2))

        for i in range(h):
            for j in range(w):
                restored[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] = channel[i, j]

        return restored.astype(np.uint8)

    @staticmethod
    def join_blocks(blocks: np.array, h, w, n=macro_N):
        new_h, new_w = h, w
        if h % n != 0 or w % n != 0:
            new_h = n * int(np.ceil(h / n))
            new_w = n * int(np.ceil(w / n))

        image = np.full((new_h, new_w), 0, dtype=object)
        u, v = new_h // n, new_w // n
        for i in range(u):
            for j in range(v):
                for k in range(n):
                    image[i * n + k][j * n: j * n + n] = blocks[i * v + j][k]
        return image[:h, :w]

    @staticmethod
    def idct(block: np.ndarray, n):
        def F(x, y):
            u = np.arange(n)
            v = np.arange(n)

            cos1 = np.cos((2 * x + 1) * u[:, None] * np.pi / (2 * n))
            cos2 = np.cos((2 * y + 1) * v[:, None] * np.pi / (2 * n))

            cos_matrix = cos1 @ cos2.T

            c_u = c_dct(u).reshape(-1, 1)
            c_v = c_dct(v).reshape(1, -1)
            scale = c_u * c_v

            res = np.sum(scale * block * cos_matrix)
            res *= 2 / n
            return res

        return np.array([[F(i, j) for j in range(n)] for i in range(n)])

    @staticmethod
    def restore_block(block, qmatrix):
        return np.round(block * qmatrix).astype(np.int32)

    @staticmethod
    def inverse_zigzag(arr, n=8):
        matrix = [[0] * n for _ in range(n)]
        i, j = 0, 0

        for idx in range(n * n):
            matrix[i][j] = arr[idx]

            if (i + j) % 2 == 0:  # Движение вверх-вправо
                if j == n - 1:
                    i += 1
                elif i == 0:
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:  # Движение вниз-влево
                if i == n - 1:
                    j += 1
                elif j == 0:
                    i += 1
                else:
                    i += 1
                    j -= 1

        return np.array(matrix)
