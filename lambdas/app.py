from flask import Flask
from flask import request

from PIL import Image
from io import BytesIO

app = Flask(__name__)

from logic.extender import authenticate_vk, index_vk, user_get_info
from logic.comparator import encode


@app.route('/find')
def find():
    import base64
    img = request.args['img']
    base64.b64decode(img)
    encoding = encode(Image.open(BytesIO(base64.b64decode(img))))

    # Теперь нужно поискать по базе данных совпадения по лицу и вернуть uid


@app.route('/index')
def index():
    vk = request.args['vk']
    uid, face_encoding = index_vk(vk)
    additional_info = user_get_info(uid)

    # сохранить uid, имя, face_encoding и additional_info


@app.route('/result')
def result():
    uid = request.args['uid']
    additional_info = user_get_info(uid)

    # гетаем результат нейронки антропыча
    # возвращаем additional_info и результат нейронки


if __name__ == '__main__':
    app.run()
