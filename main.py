from PIL import Image
import struct
from collections import Counter

from sec_functions import *


def decode(data: bytes, root: Huffman.Node, padding: int, mode: str):
    assert mode == 'AC' or mode == 'DC'

    bits_buffer = ''.join(f'{byte:08b}' for byte in data)
    if padding > 0:
        bits_buffer = bits_buffer[:-padding]

    decoded = []
    cur_node = root
    i = 0

    while i < len(bits_buffer):
        bit = bits_buffer[i]
        i += 1
        cur_node = cur_node.left if bit == '0' else cur_node.right

        if cur_node.value is not None:
            if mode == 'DC':
                cat = cur_node.value
                if cat != 0:
                    dc = iconvert_num(bits_buffer[i:i + cat], cat)
                    decoded.append(dc)
                else:
                    decoded.append(0)
                i += cat
            else:
                run_len, cat = cur_node.value
                decoded.extend([0] * run_len)
                if cat != 0:
                    ac = iconvert_num(bits_buffer[i:i + cat], cat)
                    decoded.append(ac)
                else:
                    break
                i += cat
            cur_node = root

    for i in range(1, len(decoded)):
        decoded[i] += decoded[i - 1]

    return np.array(decoded)


class WebPEncoder:
    def __init__(self, path, scale):
        image = Image.open(path)
        self.mode = modes.index(image.mode)
        if image.mode in ('L', '1'):
            image = image.convert('RGB')
        self.image = np.array(image)

        self.scale = scale
        self.y_quant = quant_matrix(Y_QUANT, scale)
        self.uv_quant = quant_matrix(CbCr_QUANT, scale)

    @staticmethod
    def toYCbCr(rgb_image: np.array):
        R = rgb_image[:, :, 0].astype(np.float32)
        G = rgb_image[:, :, 1].astype(np.float32)
        B = rgb_image[:, :, 2].astype(np.float32)

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        Cb = -0.1687 * R - 0.3313 * G + 0.5 * B + 128
        Cr = 0.5 * R - 0.4187 * G - 0.0813 * B + 128

        ycbcr_image = np.stack([Y, Cb, Cr], axis=-1).clip(0, 255).astype(np.uint8)
        return ycbcr_image

    @staticmethod
    def downsampling(channel: np.array):
        h, w = channel.shape
        downsampled = np.zeros((h // 2, w // 2))

        for i in range(0, h, 2):
            for j in range(0, w, 2):
                block = channel[i:i + 2, j:j + 2]
                downsampled[i // 2, j // 2] = block[0, 0]

        return downsampled.astype(np.int32)

    @staticmethod
    def extend_matrix(matrix: np.array, n):
        h, w = matrix.shape
        new_h, new_w = (h + n - 1) // n * n, (w + n - 1) // n * n
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
        blocks_array = np.array([image_arr[i:i + n, j:j + n] for i in range(0, h, n) for j in range(0, w, n)])

        return blocks_array

    @staticmethod
    def dct(block: np.ndarray):
        n = block.shape[0]
        x = np.arange(n)
        u = np.arange(n).reshape(-1, 1)

        cos_u = np.cos((2 * x + 1) * u * np.pi / (2 * n))
        cos_u[0, :] *= np.sqrt(1 / n)
        cos_u[1:, :] *= np.sqrt(2 / n)

        return cos_u @ block @ cos_u.T

    @staticmethod
    def quantize_block(block, qmatrix):
        return np.round(block / qmatrix).astype(np.int32)

    @staticmethod
    def zigzag(matrix, n=N):
        result = []
        i, j = 0, 0

        for _ in range(n * n):
            result.append(matrix[i][j])

            if (i + j) % 2 == 0:
                if j == n - 1:
                    i += 1
                elif i == 0:
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:
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
        for i in freq_dict.keys():
            file.write(struct.pack(CAT_FORM, i, freq_dict[i]))

        root = Huffman.build_tree(freq_dict)
        codes = Huffman.build_code(root)

        huf_str = ""
        encoded = bytearray()
        for i in range(n):
            dc = arr_dc[i]
            cat = category(dc)

            huf_str += codes[cat] + convert_num(dc, cat)
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
        for couple in freq_dict.keys():
            file.write(struct.pack(RLE_CAT_FORM, *couple, freq_dict[couple]))

        root = Huffman.build_tree(freq_dict)
        codes = Huffman.build_code(root)

        huf_str = ""
        encoded = bytearray()
        for i in range(len(rle_ac)):
            cat = rle_ac[i][1]
            huf_str += codes[rle_ac[i]] + convert_num(ac[i], cat)

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

    def process(self, path, prediction, block_size=N):
        h, w = self.image.shape[:2]
        print(h, w)
        output_file = open(path, "wb")
        output_file.write(struct.pack(INFO_FORM, h, w, block_size, self.scale, PREDS_KEYS.index(prediction)))

        image_yuv = self.toYCbCr(self.image)
        for i in range(3):
            channel = image_yuv[:, :, i]
            if i != 0:
                channel = self.downsampling(channel)
            channel = self.extend_matrix(channel, block_size)
            channel = convert(channel, prediction, 0, block_size)

            blocks = self.split_blocks(channel, block_size)
            blocks = np.array(tuple(map(self.dct, blocks)))

            qmatrix = self.y_quant if i == 0 else self.uv_quant
            blocks = np.array(tuple(map(lambda x: self.quantize_block(x, qmatrix), blocks)))
            blocks = np.array(tuple(map(lambda x: self.zigzag(x), blocks)))

            dc = blocks[:, 0]
            print("DC", len(dc))
            self.encode_dc(dc, output_file)

            ac = np.hstack(blocks[:, 1:])
            print("AC", len(ac))
            self.encode_ac(ac, output_file)

        output_file.close()
        print()


class WebPDecoder:
    def __init__(self, path):
        self.image = open(path, "rb")

    @staticmethod
    def from_CbCr(ycbcr_image: np.array):
        Y = ycbcr_image[:, :, 0].astype(np.float32)
        Cb = ycbcr_image[:, :, 1].astype(np.float32) - 128
        Cr = ycbcr_image[:, :, 2].astype(np.float32) - 128

        R = Y + 1.402 * Cr
        G = Y - 0.34414 * Cb - 0.71414 * Cr
        B = Y + 1.772 * Cb

        rgb_image = np.stack([R, G, B], axis=-1).clip(0, 255).astype(np.uint8)
        return rgb_image

    @staticmethod
    def idownsampling(channel: np.array):
        h, w = channel.shape
        restored = np.zeros((h * 2, w * 2))

        for i in range(h):
            for j in range(w):
                restored[2 * i: 2 * i + 2, 2 * j: 2 * j + 2] = channel[i, j]

        return restored.astype(np.int32)

    @staticmethod
    def join_blocks(blocks: np.array, h, w, n):
        new_h, new_w = h, w
        if h % n != 0 or w % n != 0:
            new_h = (h + n - 1) // n * n
            new_w = (w + n - 1) // n * n

        image = np.full((new_h, new_w), 0)
        u, v = new_h // n, new_w // n
        for i in range(u):
            for j in range(v):
                image[i * n: i * n + n, j * n: j * n + n] = blocks[i * v + j]
        return image

    @staticmethod
    def idct(block: np.ndarray):
        n = block.shape[0]
        u = np.arange(n)
        x = np.arange(n).reshape(-1, 1)

        cos_x = np.cos((2 * x + 1) * u * np.pi / (2 * n))
        cos_x[:, 0] *= np.sqrt(1 / n)
        cos_x[:, 1:] *= np.sqrt(2 / n)

        return cos_x @ block @ cos_x.T

    @staticmethod
    def restore_block(block, qmatrix):
        return np.round(block * qmatrix).astype(np.int32)

    @staticmethod
    def inverse_zigzag(arr, n=N):
        matrix = [[0] * n for _ in range(n)]
        i, j = 0, 0

        for idx in range(n * n):
            matrix[i][j] = arr[idx]

            if (i + j) % 2 == 0:
                if j == n - 1:
                    i += 1
                elif i == 0:
                    j += 1
                else:
                    i -= 1
                    j += 1
            else:
                if i == n - 1:
                    j += 1
                elif j == 0:
                    i += 1
                else:
                    i += 1
                    j -= 1

        return np.array(matrix)

    @staticmethod
    def decode(file, mode):
        assert mode == 'AC' or mode == 'DC'

        freg_dict_len = struct.unpack('>H', file.read(2))[0]
        freq_dict = dict()
        for i in range(freg_dict_len):
            form = CAT_FORM if mode == 'DC' else RLE_CAT_FORM
            s = struct.calcsize(form)
            couple = struct.unpack(form, file.read(s))

            key, value = couple if mode == 'DC' else (couple[:2], couple[2])
            freq_dict[key] = value

        len_data = struct.unpack(SEQ_LEN_FORM, file.read(struct.calcsize(SEQ_LEN_FORM)))[0]
        data = file.read(len_data)
        padding = struct.unpack('>B', file.read(1))[0]

        root = Huffman.build_tree(freq_dict)
        return decode(data, root=root, padding=padding, mode=mode)

    def process(self, path):
        size = struct.calcsize(INFO_FORM)
        h, w, block_size, scale, pred_ind = struct.unpack(INFO_FORM, self.image.read(size))
        print(h, w)

        k = block_size ** 2 - 1
        y_cnt = int(np.ceil(h / block_size) * np.ceil(w / block_size))
        uv_cnt = int(np.ceil(h // 2 / block_size) * np.ceil(w // 2 / block_size))

        y_qmatrix = quant_matrix(Y_QUANT, scale)
        uv_qmatrix = quant_matrix(CbCr_QUANT, scale)

        channels = tuple()
        for i in range(3):
            if i == 0:
                b = y_cnt
                qmatrix = y_qmatrix
            else:
                b = uv_cnt
                qmatrix = uv_qmatrix
            dc = np.array(self.decode(self.image, 'DC'))
            print("DC", len(dc))
            dc.resize(b, 1)

            ac = np.array(self.decode(self.image, 'AC'))
            print("AC", len(ac))
            ac.resize(b, k)

            blocks =  tuple(map(lambda x: np.hstack((dc[x], ac[x])), range(b)))
            blocks = tuple(map(lambda x: self.inverse_zigzag(x, block_size), blocks))
            blocks = np.array(tuple(map(lambda x: self.restore_block(x, qmatrix), blocks)))

            blocks = np.array(tuple(map(self.idct, blocks)))

            # size = block_size
            # h_, w_ = (h + size - 1) // size * size, (w + size - 1) // size * size
            shape = (h // 2, w // 2) if i != 0 else (h, w)
            channel = self.join_blocks(blocks, *shape, block_size)

            channel = convert(channel, PREDS_KEYS[pred_ind], 1, block_size)
            if i != 0:
                channel = self.idownsampling(channel)
            channel = channel[:h, :w]

            channels += (channel,)

        new_image = np.stack(channels, axis=-1).clip(0, 255).astype(np.int16)
        new_image = Image.fromarray(self.from_CbCr(new_image))

        new_image.save(path)
        print()


# mode = "H"
# file = "HorGradient"
# temp_file = "Images\\test"
#
# # enc = WebPEncoder(f"Images\\Originals\\{file}.png", 70)
# # enc.process(temp_file, prediction=mode)
#
# dec = WebPDecoder(temp_file)
# dec.process(f"Images\\{file}_restored_{mode}.png")