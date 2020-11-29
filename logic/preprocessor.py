import dlib
import numpy as np
from PIL import Image, ImageDraw
from math import *

from matplotlib import pyplot as plt


def normalize(img):
    return img.resize((256, int(img.size[1] * (256 / img.size[0]))), Image.ANTIALIAS)


def detect(img):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('../resources/face_landmarks_model.dat')

    img_grayscale = np.array(img.convert('L'))
    faces = detector(img_grayscale)
    if len(faces) == 1:
        face = np.array([[faces[0].left(), faces[0].top()],
                         [faces[0].right(), faces[0].bottom()],
                         [faces[0].right(), faces[0].top()],
                         [faces[0].left(), faces[0].bottom()]])

        points = predictor(image=img_grayscale, box=faces[0])
        face_parts = face_parts_mean(np.array(list(map(lambda p: [p.x, p.y], points.parts()))))

        return img, face, face_parts

    return None


def face_parts_mean(landmarks):
    nose = landmarks[27:36].mean(axis=0).astype('int')
    left_eye = landmarks[36:42].mean(axis=0).astype('int')
    right_eye = landmarks[42:48].mean(axis=0).astype('int')

    return {'nose': nose, 'left_eye': left_eye, 'right_eye': right_eye}


def rotation_z_coefficient(face, face_parts):
    eye_center = (face_parts['right_eye'] + face_parts['left_eye']) / 2
    nose = face_parts['nose']
    score = sqrt(((eye_center - nose)[0] ** 2 + (eye_center - nose)[1] ** 2) / (
            (face[1] - face[0])[0] ** 2 + (face[1] - face[0])[1] ** 2))

    return score


def warp_rotate(image, angle, center=None, expand=False):
    if center is None:
        return image.rotate(angle, expand)
    angle = -angle
    nx, ny = x, y = center
    cosine = cos(angle)
    sine = sin(angle)
    a = cosine
    b = sine
    c = x - nx * a - ny * b
    d = -sine
    e = cosine
    f = y - nx * d - ny * e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)


def align(img, face, face_parts):
    eye_vector = face_parts['right_eye'] - face_parts['left_eye']
    eye_center = (face_parts['right_eye'] + face_parts['left_eye']) / 2
    rotation_angle = acos(np.dot(eye_vector, [1, 0]) / sqrt(eye_vector[0] ** 2 + eye_vector[1] ** 2))

    img = warp_rotate(img, rotation_angle, eye_center)
    return img


def crop(img, face):
    img = np.array(img)
    top_left_x = min([face[0][0], face[1][0], face[2][0], face[3][0]])
    top_left_y = min([face[0][1], face[1][1], face[2][1], face[3][1]])
    bot_right_x = max([face[0][0], face[1][0], face[2][0], face[3][0]])
    bot_right_y = max([face[0][1], face[1][1], face[2][1], face[3][1]])

    return img[top_left_y:bot_right_y, top_left_x:bot_right_x]


def preprocess(img):
    # try:
    img = normalize(img)
    img, face, face_parts = detect(img)

    rotation_z = abs(rotation_z_coefficient(face, face_parts))

    if rotation_z < (0.7 / 12.5):
        raise BaseException
    img, face, face_parts = detect(align(img, face, face_parts))

    img = crop(img, face)

    return Image.fromarray(img).resize((256, 256), Image.ANTIALIAS), rotation_z
#
# except BaseException:
#     return None
