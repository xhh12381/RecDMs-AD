import os.path

from PIL import Image

imgs_path = "./datasets/"
with Image.open(os.path.join(imgs_path, "003.png")) as img:
    img_256 = img.resize((256, 256))
    img_256.save(os.path.join(imgs_path, "003_256.png"))
    img_128 = img.resize((128, 128))
    img_128.save(os.path.join(imgs_path, "003_128.png"))

