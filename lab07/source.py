import numpy as np

from enum import IntEnum


np.set_printoptions(threshold=10000, linewidth=250)


# Функционирование в синхронном режиме
actF = lambda net, y: 1 if net > 0 else (-1 if net < 0 else y)


# Векторизация образов
def vectorize(img):
    return np.transpose(img).flatten()


# Сравнение векторов
def eq(l1, l2):
    return all(x == y for x, y in zip(l1, l2))


def fit_model(imgs):
    # Начальные веса как векторная сумма матричных произведений образов на транспонированных себя же
    W = sum(np.dot(np.transpose([img]), [img]) for img in imgs)

    # Зануляем диагональ
    np.fill_diagonal(W, 0)

    return W


# Перечисление для вариантов остановки
class StopState(IntEnum):
    NONE = 0
    LAST = 1
    ASYNC = 2
    SYNC = 3


def to_stop(Y, imgs):
    # Проверяем совпадение с одним из образов
    if any(eq(Y[-1], im) for im in imgs):
        return StopState.LAST
    
    # Проверяем зацикленность для асинхронного режима
    if len(Y) > 1 and eq(Y[-1], Y[-2]):
        return StopState.ASYNC

    # Проверяем зацикленность для синхронного режима
    if len(Y) > 3 and eq(Y[-1], Y[-3]) and eq(Y[-2], Y[-4]):
        return StopState.SYNC
    
    return StopState.NONE
    

def predict_once(W, img):
    # Векторное умножение
    Net = np.dot(W, img)

    # Применяем активацию
    return [actF(net, i) for net, i in zip(Net, img)]


def model_predict(W, imgs, img):
    Y = [img]
    while True:
        # Проверяем нужна ли остановка
        state = to_stop(Y, imgs)
        if state != StopState.NONE:
            return Y, state

        # Вычисляем результат итерации    
        y = predict_once(W, Y[-1])
        
        # Записываем результат итерации к остальным
        Y.append(y)
