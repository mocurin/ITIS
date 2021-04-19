import aiohttp_jinja2
import jinja2
import secrets

from aiohttp import web

from source import fit_model, vectorize


routes = web.RouteTableDef()

PORT = 8080
base_url = f"http://localhost:{PORT}"
IMGS, X, Y = 3, 5, 7
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


@routes.post('/fit')
async def fit(request: web.Request):
    data = await request.post()

    imgs = data['imgs']
    imgs = int(imgs)

    # Проверяем количество образов
    if imgs < 0:
        raise web.HTTPBadRequest(text=f"Invalid images number: {imgs}")

    x = data['x']
    x = int(x)

    y = data['y']
    y = int(y)
    
    # Проверяем размеры образов
    if x * y < 15 or x * y > 50:
        raise web.HTTPBadRequest(text=f"Invalid XY combination: X={x}, Y={y}, XY={x * y}")

    # Создаем матрицы всех образов
    imgs = [[[-1 for _ in range(x)] for _ in range(y)] for img in range(imgs)]

    for key in data.keys():
        if key in {'x', 'y', 'imgs'}:
            continue

        k, key = key.split(':')
        i, j = key.split('-')
        i, j, k = int(i), int(j), int(k)

        imgs[k][j][i] = 1

    import numpy as np
    print([np.array(im) for im in imgs])
    
    imgs = [vectorize(im) for im in imgs]

    token = secrets.token_hex(8)

    networks[token] = (imgs, fit_model(imgs))

    raise web.HTTPFound(location=f"{base_url}/predict/{token}")


@routes.get('/predict/{token}')
@aiohttp_jinja2.template('predict.html')
async def web_predict(request: web.Request):
    token = request.match_info['token']

    print(networks[token])

    return {} 


@routes.post('/predict/{token}')
@aiohttp_jinja2.template('result.html')
async def predict(request: web.Request):
    pass


def create():
    app = web.Application()
    app.add_routes(routes)
    
    aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader('lab07/templates'))

    web.run_app(app, port=PORT)


if __name__ == '__main__':
    create()
