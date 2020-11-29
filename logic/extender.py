import asyncio
import aiohttp
from PIL import Image
import numpy as np
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from comparator import encode, compare_encode_w_encode
from preprocessor import preprocess
from utils import load_image_from_url

vk_token = open('../resources/vk_token1.key', 'r').readline()
access_token = '73ec302a8e538bfd694f09a13c2939aff7f916fcf46c986bc4ccc5a03e003b078b24578c45f359c561c3c'


# async def authorisation(session):
#     async with session.get(
#             f'https://oauth.vk.com/authorize?client_id=7679929&client_secret=87SNTGe10uUFMi0F4Pdy&v=5.126&grant_type'
#             f'=client_credentials&response_type=token') as response:
#         return await response.json()

def preprocess_vk_images(images):
    result = []
    i = 0
    for photos in images:
        if i > 2:
            break
        try:
            for size in photos['sizes']:
                if size['type'] == 'x':
                    preprocessed = preprocess(load_image_from_url(size['url']))[0]
                    result.append(np.array(preprocessed))
        except BaseException:
            i -= 1
        i += 1
    return result


async def candidates_vk(name):
    session = aiohttp.ClientSession()

    fields_validate = 'photo_id'
    fields = 'photo_id, sex, bdate, city, country, home_town, photo_400_orig, lists, domain, has_mobile, contacts, site, education, universities, schools, status, followers_count, common_count, occupation, nickname, relatives, relation, personal, connections, exports, wall_comments, activities, interests, music, books, games, about, quotes, timezone, screen_name, maiden_name, crop_photo, career, military'

    async with session.get(
            f'https://api.vk.com/method/users.search?q={name}&has_photo=1&field={fields}&access_token={access_token}&v=5.126') as response:
        data = await response.json()
        candidates = list(map(lambda x: x['id'], data['response']['items']))

    candidates_faces = []

    for i, candidate in enumerate(candidates):
        if (i + 1) % 3:
            await asyncio.sleep(1)

        async with session.get(
                f'https://api.vk.com/method/photos.getUserPhotos?user_id={candidate}&album_id=profile&access_token={access_token}&sort=0&photo_sizes=1&type=x&count=10&v=5.126') as response:
            data = await response.json()
            candidates_faces.append([])
            if 'error' not in data:
                candidates_faces[-1] = preprocess_vk_images(data['response']['items'])
            if 'error' in data or len(candidates_faces[-1]) < 1:
                async with session.get(
                        f'https://api.vk.com/method/photos.get?owner_id={candidate}&album_id=profile&access_token={access_token}&rev=1&feed_type=photo&photo_sizes=0&count=100&v=5.126') as response:
                    data = await response.json()
                    candidates_faces[-1].extend(preprocess_vk_images(data['response']['items']))

    await session.close()

    return np.array(candidates), np.array(candidates_faces, dtype=object)


async def index_vk(url):
    session = aiohttp.ClientSession()
    async with session.get(
            f'https://api.vk.com/method/users.get?user_ids={url.split("/")[3]}&album_id=profile&access_token={access_token}&sort=0&photo_sizes=1&type=x&count=10&v=5.126') as response:
        data = await response.json()
        uid = data['response'][0]['id']
    faces = []

    async with session.get(
            f'https://api.vk.com/method/photos.getUserPhotos?user_id={uid}&album_id=profile&access_token={access_token}&sort=0&photo_sizes=1&type=x&count=10&v=5.126') as response:
        data = await response.json()

        if 'error' not in data:
            faces = preprocess_vk_images(data['response']['items'])
        if 'error' in data or len(faces) < 1:
            async with session.get(
                    f'https://api.vk.com/method/photos.get?owner_id={uid}&album_id=profile&access_token={access_token}&rev=1&feed_type=photo&photo_sizes=0&count=100&v=5.126') as response:
                data = await response.json()
                faces.extend(preprocess_vk_images(data['response']['items']))

        faces.append(np.array(Image.open("../resources/face2.jpg")))

        faces = list(map(lambda x: encode(x), faces))

        outlier_detection = DBSCAN(min_samples=2, eps=0.5)
        clusters = outlier_detection.fit_predict(faces)

        delta = 0
        for i in range(len(clusters)):
            if clusters[i] == -1:
                del faces[i - delta]
                delta += 1

    await session.close()
    return uid, faces[0]


async def user_get_info(uid):
    session = aiohttp.ClientSession()

    async with session.get(
            f'https://api.vk.com/method/account.getProfileInfo?user_id={uid}album_id=profile&access_token={access_token}&sort=0&photo_sizes=1&type=x&count=10&v=5.126') as response:
        data = (await response.json())['response']

    await session.close()
    return data['home_town'] if data['home_town'] != '' else None, \
           data['status'] if data['status'] != '' else None, \
           data['bdate'] if data['bdate'] != '' else None, data['city']['title'] if data['city'] else None,


async def authenticate_vk(face_encoding, name):
    candidates, candidates_faces = await candidates_vk(name)

    candidates_encodes = []
    for candidate_images in candidates_faces:
        candidates_encodes.append([])
        for image in candidate_images:
            try:
                encoding = encode(image)
                candidates_encodes[-1].append(encoding)
            except BaseException:
                pass
        candidates_encodes[-1] = np.array(candidates_encodes[-1])
    candidates_encodes = np.array(candidates_encodes, dtype=object)

    candidates_rate = []
    for encodes in candidates_encodes:
        true_sum = 0
        amount = 0
        for encoding in encodes:
            true_sum += int(compare_encode_w_encode(face_encoding, encoding)[0])
            amount += 1
        try:
            candidates_rate.append(true_sum / amount)
        except BaseException:
            candidates_rate.append(0)

    return candidates[candidates_rate.index(max(candidates_rate))], candidates_faces[
        candidates_rate.index(max(candidates_rate))]


# asyncio.run(authenticate(encode(Image.open("../resources/face0.jpg")), "Никита Мусалов"))
async def tmp():
    user_id, face_encoding = await index_vk("https://vk.com/mobynick_kaktys")
    print(await user_get_info(user_id))


asyncio.run(tmp())
