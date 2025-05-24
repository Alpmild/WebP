import os
from main import *
from main import WebPEncoder

def func(file):
    input_folder = "Images\\Originals"
    temp_folder = "Images\\Temp"
    output_folder = "Images\\Restored"

    quality = {
        'H': range(0, 101, 5),
        'V': range(0, 101, 5),
        'DC': range(0, 101, 5),
        'TM': range(0, 101, 5),
        'HUP': range(0, 101, 5),
        'VLEFT': range(0, 101, 5),
        'D45': range(0, 101, 5)
    }

    for pred in quality:
        for q in quality[pred]:
            print(file, pred, q)
            file_name, exp = os.path.splitext(file)

            tfold = temp_folder
            for t in (file_name, pred):
                tfold = f"{tfold}\\{t}"
                if not os.path.exists(tfold):
                    os.mkdir(tfold)

            ofold = output_folder
            for t in (file_name, pred):
                ofold = f"{ofold}\\{t}"
                if not os.path.exists(ofold):
                    os.mkdir(ofold)

            orig = f"{input_folder}\\{file}"
            temp = f"{tfold}\\{file_name}_{pred}_{q}"
            out = f"{ofold}\\{file_name}_{pred}_{q}{exp}"

            enc = WebPEncoder(orig, q)
            enc.process(temp, pred)

            dec = WebPDecoder(temp)
            dec.process(out)


files = (
    # "Lenna.png",
    "HorGradient.png",
    "VerGradient.png",
    "Saber.png",
    "Saber_bw.png",
    "Saber_bw_dithered.png",
)
for f in files:
    func(f)