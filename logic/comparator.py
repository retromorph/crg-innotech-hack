import dlib
import face_recognition as fc
import numpy as np
from PIL import Image

from preprocessor import preprocess


def encode(face):
    face = np.array(face)
    return fc.face_encodings(face)[0]


def compare_face_w_face(face1, face2):
    return fc.compare_faces([encode(face1)], encode(face2))


def compare_faces_w_face(faces, face1):
    return fc.compare_faces(map(lambda face: encode(face), faces), encode(face1))


def compare_faces_w_faces(faces1, faces2):
    return map(lambda face2: fc.compare_faces(map(lambda face1: encode(face1), faces1), encode(face2)), faces2)


def compare_encode_w_encode(encode1, encode2):
    return fc.compare_faces([encode1], encode2)


def authentication(proto, face):
    return fc.compare_faces(proto, encode(face))
