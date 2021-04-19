import aiohttp_jinja2
import jinja2
import secrets
import random

from aiohttp import web
from aiohttp.client import MultiDict

from source import fit_model, vectorize, model_predict


routes = web.RouteTableDef()

PORT = 8080
base_url = f"http://localhost:{PORT}"
IMGS, X, Y = 3, 5, 7
THRESH = 0.1
networks = {}


@routes.get('/')
async def setup(request: web.Request):
    raise web.HTTPFound(location=f"{base_url}/fit/{IMGS}:{X}-{Y}")


@routes.get('/fit/{imgs}:{x}-{y}')
@aiohttp_jinja2.template('fit.html')
async def web_fit(request: web.Request):
    imgs = request.match_info['imgs']
    x = request.match_info['x']
    y = request.match_info['y']

    return {
        'base_url': base_url,
        'imgs': int(imgs),
        'x': int(x),
        'y': int(y),
    }


@routes.post('/change')
async def change_dim(request: web.Request):
    data = await request.post()

    raise web.HTTPFound(location=f"{base_url}/fit/{data['imgs'] or IMGS}:{data['x'] or X}-{data['y'] or Y}")


def _create_images(x, y, imgs, data):
    images = [[[-1 for _ in range(x)]
               for _ in range(y)]
              for img in range(imgs)]

    for key in data.keys():
        k, key = key.split(':')
        i, j = key.split('-')
        i, j, k = int(i), int(j), int(k)

        images[k][j][i] = 1

    return [vectorize(im) for im in images]

def _create_image(x, y, data):
    image = [[-1 for _ in range(x)] for _ in range(y)]

    for key in data.keys():
        i, j = key.split('-')
        i, j = int(i), int(j)

        image[j][i] = 1
    
    return vectorize(image)


@routes.post('/fit')
async def fit(request: web.Request):
    data = await request.post()
    data = MultiDict(data)

    imgs = data.pop('imgs')
    imgs = int(imgs)

    # Проверяем количество образов
    if imgs < 0:
        raise web.HTTPBadRequest(text=f"Invalid images number: {imgs}")

    x = data.pop('x')
    x = int(x)

    y = data.pop('y')
    y = int(y)

    # Проверяем размеры образов
    if x * y < 15 or x * y > 50:
        raise web.HTTPBadRequest(text=f"Invalid XY combination: X={x}, Y={y}, XY={x * y}")

    # Создаем матрицы всех образов
    images = _create_images(x, y, imgs, data)

    token = secrets.token_hex(8)

    networks[token] = ((x, y, imgs), images, fit_model(images))

    raise web.HTTPFound(location=f"{base_url}/predict/{token}")


@routes.get('/weights/{token}')
async def get_weights(request: web.Request):
    token = request.match_info['token']

    raise web.HTTPOk(text=str(networks[token][-1]))


@routes.get('/predict/{token}')
@aiohttp_jinja2.template('predict.html')
async def web_predict(request: web.Request):
    token = request.match_info['token']
    randomize = request.url.query.get('randomize') or THRESH

    try:
        randomize = float(randomize)
    except ValueError:
        randomize = THRESH

    if randomize < 0. or randomize > 1.:
        raise web.HTTPBadRequest(text=f"Probability value exceeds [0, 1] limits: randomize={by}")

    (x, y, imgs), images, weights = networks[token]

    # Случайно изменяем образ, если опция была указана
    if request.url.query.get('randomize'):
        images = [[-val if random.random() < randomize else val
                   for val in im]
                  for im in images]

    im_idx = random.randrange(0, imgs)
    return {
        'x': x,
        'y': y,
        'image': images[im_idx],
        'token': token,
        'randomize': randomize
    }


@routes.post('/predict/{token}')
@aiohttp_jinja2.template('result.html')
async def predict(request: web.Request):
    token = request.match_info['token']
    data = await request.post()

    (x, y, _), images, weights = networks[token]
    image = _create_image(x, y, data)

    output, state = model_predict(weights, images, image)

    return {
        'x': x,
        'y': y,
        'input': image,
        'output': output,
        'state': int(state)
    }


def create():
    app = web.Application()
    app.add_routes(routes)

    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('lab07/templates'))

    web.run_app(app, port=PORT)


if __name__ == '__main__':
    create()
