import numpy as np
from Constants import DCT_MAT

from PIL import Image

# Загрузка изображения RGBA
image_rgba = Image.open("Images\\Originals\\Lenna.png")

# Конвертация в RGB
image_rgb = image_rgba.convert("RGB")

# Сохранение результата
image_rgb.save("Images\\Originals\\Lenna.png")