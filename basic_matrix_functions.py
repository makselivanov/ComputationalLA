import numpy as np


def isMatrix(a: np.ndarray) -> bool:
    return a.ndim == 2


def isSquareMatrix(a: np.ndarray) -> bool:
    return isMatrix(a) and a.shape[0] == a.shape[1]


def isVector(b: np.ndarray) -> bool:
    return b.ndim == 1


def length(xs: np.array) -> float:
    return np.sqrt(abs(sum(map(lambda x: abs(x) ** 2, xs))))


# Return normalized vector
def norm(v: np.ndarray) -> np.ndarray:
    return np.array(list(map(lambda x: x / length(v), v)))


# A - matrix, return two array of radius and center of circle
def getGershgorinCircles(a: np.ndarray) -> (np.ndarray, np.ndarray):
    radius, center = [], []
    for i, row in enumerate(a):
        x, r = row[i], 0.
        for j, e in enumerate(row):
            if i != j:
                r += abs(e)
        radius.append(r)
        center.append(x)
    return np.array(radius), np.array(center)


# A - matrix
# return True if all gershorin circles is near a zero
def isEigenvaluesSmall(a: np.ndarray) -> bool:
    r, c = getGershgorinCircles(a)
    for i in range(r.size):
        if abs(r[i]) + abs(c[i]) > 1:
            return False
    return True


# A - matrix
# return True if all radius of gershorin circles is small
def isAllRadiusSmall(a: np.ndarray, eps: float) -> bool:
    r, c = getGershgorinCircles(a)
    return (r < eps).all()


def isSymmMatrix(a: np.ndarray, eps: float) -> bool:
    for i in range(a.shape[0]):
        for j in range(i):
            if abs(a[i, j] - a[j, i]) > eps:
                return False
    return True


def isTridiagMatrix(a: np.ndarray, eps: float) -> bool:
    for i in range(a.shape[0]):
        for j in range(i - 1):
            if abs(a[i, j]) > eps:
                return False
        for j in range(i + 2, a.shape[0]):
            if abs(a[i, j]) > eps:
                return False
    return True