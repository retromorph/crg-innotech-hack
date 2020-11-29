import dlib
import face_recognition as fc
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from preprocessor import preprocess


def encode(img):
    return fc.face_encodings(img)[0]


def compare_img_w_img(img1, img2):
    img1, img2 = np.array(img1), np.array(img2)
    return fc.compare_faces([encode(img1)], encode(img2))


def compare_imgs_w_img(imgs, img1):
    img1 = np.array(img1)
    return fc.compare_faces(map(lambda img: encode(np.array(img)), imgs), encode(img1))


def compare_imgs_w_imgs(imgs1, imgs2):
    return map(lambda img2: fc.compare_faces(map(lambda img1: encode(np.array(img1)), imgs1), encode(np.array(img2))),
               imgs2)


def authentication(proto, img):
    img = np.array(img)
    return fc.compare_faces(proto, encode(img))
