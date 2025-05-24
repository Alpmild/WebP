from PIL import Image
import os.path
from matplotlib import pyplot
import numpy as np

pred_colors= dict(H="red", V="blue", DC="green", TM="black", HUP="orange", VLEFT="purple", D45=(0, 255, 255))
files = (
    "Lenna.png",
    "Saber.png",
    "Saber_bw.png",
    "Saber_bw_dithered.png",
    "HorGradient.png",
    "VerGradient.png",
)

def graphics(file_dir: str, quality=tuple(range(0, 101, 5))):
    file_name, exp = os.path.splitext(file_dir)
    pyplot.figure(figsize=(8, 6))

    for pred in pred_colors:
        temp_fold = f"Images\\Temp\\{file_name}\\{pred}"
        sizes = []
        for q in quality:
            file = f"{temp_fold}\\{file_name}_{pred}_{q}"
            sizes.append(os.path.getsize(file) / 1024)

        pyplot.plot(quality, sizes, label=pred)

    pyplot.grid(True)
    pyplot.xlabel("Качество, %")
    pyplot.ylabel("Размер, Кб")
    pyplot.title(f"{file_name}.raw")

    pyplot.legend()
    pyplot.savefig(f"Images\\Графики\\{file_dir}")


def to_raw(file):
    file_name, exp = os.path.splitext(file)

    img = Image.open(f"Images\\Originals\\{file}")
    img_array = np.array(img)
    with open(f"Images\\RawOriginals\\{file_name}.raw", "wb") as f:
        f.write(img_array.tobytes())


for f in files:
    graphics(f)