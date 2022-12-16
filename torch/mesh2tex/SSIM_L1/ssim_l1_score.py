import numpy as np
from skimage.measure import compare_ssim as ssim
import imageio
import os
from PIL import Image


def calculate_ssim_l1_given_paths(paths):
    file_list = os.listdir(paths[0])
    ssim_value = 0
    l1_value = 0
    for f in file_list:
        # assert(i[0] == i[1])
        fake = load_img(paths[0] + f)
        real = load_img(paths[1] + f)
        ssim_value += np.mean(
            ssim(fake, real, multichannel=True))
        l1_value += np.mean(abs(fake - real))
    
    ssim_value = ssim_value/float(len(file_list))
    l1_value = l1_value/float(len(file_list))

    return ssim_value, l1_value


def calculate_ssim_l1_given_tensor(images_fake, images_real):
    bs = images_fake.size(0)
    images_fake = images_fake.permute(0, 2, 3, 1).cpu().numpy()
    images_real = images_real.permute(0, 2, 3, 1).cpu().numpy()

    ssim_value = 0
    l1_value = 0
    for i in range(bs):
        # assert(i[0] == i[1])
        fake = images_fake[i]
        real = images_real[i]
        ssim_value += np.mean(
            ssim(fake, real, multichannel=True))
        l1_value += np.mean(abs(fake - real))
    ssim_value = ssim_value/float(bs)
    l1_value = l1_value/float(bs)

    return ssim_value, l1_value


def load_img(path):
    img = imageio.imread(path)
#     try:
#         img = Image.open(path) # open the image file
#         img.verify() # verify that it is, in fact an image
#         img = imageio.imread(path)
#     except (IOError, SyntaxError) as e:
#         print('Bad file:', path)
#         #os.remove(base_dir+"\\"+filename) (Maybe)
    img = img.astype(np.float64) / 255
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[2] == 1:
        img = np.concatenate([img, img, img], axis=-1)
    elif img.shape[2] == 4:
        img = img[:, :, :3]

    return img