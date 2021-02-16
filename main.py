import basic_matrix_functions as bmf
import iomatrix as iom
import time
import math
import numpy as np

MAX_COUNTER = 20
EPSILON = 1e-5


def firstStepSolution(a: np.ndarray, b: np.ndarray) -> np.array:
    return np.zeros(a.shape[1])


# Problem 1
# A - matrix, b - vector, eps - float
def simpleIteration(a: np.ndarray, b: np.ndarray, eps):
    if a.shape[0] != b.size:
        raise ValueError('Different numbers of rows in x = Ax + b')
    x = firstStepSolution(a, b)

    counter = MAX_COUNTER
    bigEigen = not bmf.isEigenvaluesSmall(a)

    while bmf.length(x - a.dot(x) - b) > eps and counter > 0:
        nx = a.dot(x) + b
        if bigEigen:
            if bmf.length(nx) < bmf.length(x) + 1:
                counter = MAX_COUNTER
            else:
                counter -= 1
            if counter == 0:
                return 0
        x = nx
    return x


# Problem 2
# A - matrix, b - vector, eps - float
def useGaussSeidel(a: np.ndarray, b: np.ndarray, eps):
    def f(a: np.ndarray, b: np.array, x: np.array):
        nx = np.array(b)
        for i in range(x.size):
            nx[i] = b[i]
            for j in range(x.size):
                if j < i:
                    nx[i] -= nx[j] * a[i, j]
                else:
                    if j > i:
                        nx[i] -= x[j] * a[i, j]
            nx[i] /= a[i, i]
        return nx

    x = firstStepSolution(a, b)
    flag = True
    for i in range(a.shape[0]):
        if abs(a[i, i]) > eps:
            flag = False
    if flag:
        return 0
    counter = MAX_COUNTER
    while bmf.length(a.dot(x) - b) > eps:
        nx = f(a, b, x)
        if bmf.length(nx) < bmf.length(x) + 1:
            counter = MAX_COUNTER
        else:
            counter -= 1
        if counter == 0:
            return 0
        x = nx
    return x


# A - matrix
def useGivensRotation(a: np.ndarray, i: int, j: int, cos: float, sin: float) -> np.ndarray:
    if abs(cos ** 2 + sin ** 2 - 1) > EPSILON:
        raise ValueError("cos^2 + sin^2 != 1")
    n = a.shape[0]
    buffer = a.copy()
    if i < 0 or i >= n or j < 0 or j >= n:
        raise ValueError("i or j not in range")
    rowi = np.array([0.0] * n)
    rowj = np.array([0.0] * n)
    for k in range(n):
        rowi[k] = cos * buffer[i, k] + sin * buffer[j, k]
        rowj[k] = -sin * buffer[i, k] + cos * buffer[j, k]
    for k in range(n):
        buffer[i, k] = rowi[k]
        buffer[j, k] = rowj[k]
    return buffer


def getCos(y: float, x: float):
    return x / np.sqrt(x ** 2 + y ** 2)


def getSin(y: float, x: float):
    return y / np.sqrt(x ** 2 + y ** 2)


# Problem 4
# A - matrix, return (Q, R)
# Q - Orthogonal matrix, R - upper triangular matrix
def findQR_useGivensRotation(a: np.ndarray) -> (np.ndarray, np.ndarray):
    n = a.shape[0]
    q = np.eye(n)   #E_n
    r = a.copy()

    for c in range(n):
        ind = -1
        for i in range(c, n):
            if ind == -1:
                if abs(r[i, c]) > EPSILON:
                    ind = i
            else:
                if abs(r[i, c]) > EPSILON:
                    cos = getCos(r[i, c], r[ind, c])
                    sin = getSin(r[i, c], r[ind, c])
                    r = useGivensRotation(r, ind, i, cos, sin)
                    q = useGivensRotation(q, ind, i, cos, sin)
                    ind = i
        if ind != -1 and ind != c:
            useGivensRotation(r, c, ind, 0, 1)
            useGivensRotation(q, c, ind, 0, 1)
    q = q.T
    return q, r


# Problem 5
# A - matrix, v - vector
def useHouseholderMultiplication(a: np.ndarray, v: np.ndarray) -> np.ndarray:
    v = bmf.norm(v)
    return a - (2 * v).reshape(v.size, 1) @ (v @ a).reshape(1, a.shape[1])


# Problem 6
# A - matrix, return (Q, R)
# Q - Orthogonal matrix, R - upper triangular matrix
def findQR_WithHouseholder(a: np.ndarray) -> (np.ndarray, np.ndarray):
    n = a.shape[0]
    q = np.eye(n)
    r = a.copy()

    for i, column in enumerate(r.T[:-1]):
        column = bmf.norm(column[i:])
        e = np.zeros_like(column)
        e[0] = 1
        u = bmf.norm(column - e)
        q[i:] = useHouseholderMultiplication(q[i:], u)
        r[i:] = useHouseholderMultiplication(r[i:], u)
    return q.T, r


# Problem 7
# A - matrix, x - vector, eps - float
def findMaxEigenvalue(a: np.ndarray, x: np.ndarray, eps: float) -> complex:
    start = time.time()
    while bmf.length(x) < eps:
        x = np.random.randn(x.size) + np.random.randn(x.size) * 1j
    x = x / bmf.length(x)
    lamb = x @ (a @ x)
    while bmf.length(a @ x - lamb * x) > eps:
        if time.time() > start + 5:
            return 0
        buffer = a @ x
        x = buffer / bmf.length(buffer)
        lamb = x @ (a @ x)
    return lamb


# Problem 8
# A - symmetric matrix, eps - float, return (L, Q)
# L - array of eigenvalue, Q - Orthogonal matrix
def findAllEigenvalue_Symmetric(a: np.ndarray, eps: float) -> (np.ndarray, np.ndarray):
    allQ = np.eye(a.shape[0])
    r = a.copy()
    while not bmf.isAllRadiusSmall(r, eps):
        buffer, r = findQR_WithHouseholder(r)
        allQ = allQ @ buffer
        r = r @ buffer
    return r.diagonal(), allQ


# Problem 9
# A - symmetric matrix, return (A', Q)
# A' - tridiagonalization of the matrix, Q - Orthogonal matrix
def findTridiagonalization(a: np.ndarray) -> (np.ndarray, np.ndarray):
    n = a.shape[0]
    q = np.eye(n)
    newA = a.copy()
    for i, row in enumerate(newA):
        if i + 2 == n:
            break
        row = row[i:] * np.concatenate((np.zeros(1), np.ones(n - i - 1)))
        row = bmf.norm(row)
        e = np.zeros_like(row)
        e[1] = 1
        u = row - e
        u = bmf.norm(u)
        q[i:] = useHouseholderMultiplication(q[i:], u)
        newA[i:, i:] = useHouseholderMultiplication(useHouseholderMultiplication(newA[i:, i:], u).T, u).T
    return newA, q


def useMultiplication_R_QT_fromTridiagonal(r: np.ndarray, q: np.ndarray) -> np.ndarray:
    buffer = np.ndarray((r.shape[0], q.shape[0]))
    for i, row in enumerate(r):         # считаем newA = RQ
        for j, column in enumerate(q):  # because right now we store Q.T = Q^-1
            buffer[i, j] = row[i] * column[i]
            if i + 1 < row.size:
                buffer[i, j] += row[i + 1] * column[i + 1]
            if i + 2 < row.size:
                buffer[i, j] += row[i + 2] * column[i + 2]
    return buffer


def useGivenRotation_fromTridiagonal(r: np.ndarray, q: np.ndarray, all_q: np.ndarray, index: int):
    len = math.sqrt(r[index, index] ** 2 + r[index + 1, index] ** 2)
    c = r[index, index] / len
    s = r[index + 1, index] / len
    allQ = useGivensRotation(all_q, index, index + 1, c, s)
    q = useGivensRotation(q, index, index + 1, c, s)
    r = useGivensRotation(r, index, index + 1, c, s)
    return r, q, allQ


# Problem 10
# A - Tridiagonal matrix, eps - float, return (L, Q)
# L - array of eigenvalue, Q - Orthogonal matrix
def findAllEigenvalue_Tridiagonal(a: np.ndarray, eps: float) -> (np.ndarray, np.ndarray):
    n = a.shape[0]
    allQ = np.eye(n)
    newA = a.copy()
    while not bmf.isAllRadiusSmall(newA, eps):
        q = np.eye(n)
        r = newA.copy()
        for i in range(n - 1):
            r, q, allQ = useGivenRotation_fromTridiagonal(r, q, allQ, i)
        newA = useMultiplication_R_QT_fromTridiagonal(r, q)
    allQ = allQ.T
    return newA.diagonal(), allQ


def findNearestEigenvalue_22(a: np.ndarray, eps) -> float:
    delta = (a[0, 0] - a[1, 1]) ** 2 + 4 * a[0, 1] * a[1, 0]
    if abs(delta) > eps:
        delta = math.sqrt(delta)
    else:
        delta = 0.
    x1 = (a[0, 0] + a[1, 1] - delta) / 2
    x2 = (a[0, 0] + a[1, 1] + delta) / 2
    if abs(a[1, 1] - x1) < abs(a[1, 1] - x2):
        return x1
    return x2


# Problem 11
# A - Tridiagonal matrix, eps - float, return (L, Q)
# L - array of eigenvalue, Q - Orthogonal matrix
def findAllEigenvalue_FastSmallApprox(a: np.ndarray, eps: float) -> (np.ndarray, np.ndarray):
    n = a.shape[0]
    m = n
    allQ = np.eye(n)
    newA = a.copy()
    LOOP = 1
    index = 0
    while not bmf.isAllRadiusSmall(newA, eps) and m > 1:
        approx = 0.
        index += 1
        if index == LOOP:
            approx = findNearestEigenvalue_22(newA[m-2:m, m-2:m], eps)
        q = np.eye(m)
        r = newA[:m, :m] - approx * np.eye(m)
        for i in range(m - 1):
            r, q, allQ = useGivenRotation_fromTridiagonal(r, q, allQ, i)
        newA[:m, :m] = useMultiplication_R_QT_fromTridiagonal(r, q)
        for i in range(m):
            newA[i, i] += approx
        if index == LOOP:
            if abs(newA[m - 1, m - 2]) < eps and abs(newA[m - 2, m - 1]) < eps:
                m -= 1
            index = 0
    allQ = allQ.T
    return newA.diagonal(), allQ


def graphIsomorphism(graph1: np.ndarray, graph2: np.ndarray) -> int:
    if graph1.shape != graph2.shape:
        return 0
    (tr1, q1) = findTridiagonalization(graph1)
    (tr2, q2) = findTridiagonalization(graph2)
    (vals1, vecs1) = findAllEigenvalue_FastSmallApprox(tr1, EPSILON)
    (vals2, vecs2) = findAllEigenvalue_FastSmallApprox(tr2, EPSILON)
    if not (sorted(vals1) == sorted(vals2)):
        return 0
    n = graph1.shape[0]
    degrees1 = [(0, 0)] * n
    degrees2 = [(0, 0)] * n
    for i, row in enumerate(graph1):
        for j, x in enumerate(row):
            if abs(x) > EPSILON:
                degrees1[i] += (0, 1)
                degrees1[j] += (1, 0)
    for i, row in enumerate(graph2):
        for j, x in enumerate(row):
            if abs(x) > EPSILON:
                degrees2[i] += (0, 1)
                degrees2[j] += (1, 0)
    if sorted(degrees1) != sorted(degrees2):
        return 0
    return 1


def findDegreeForRegular(graph: np.ndarray):
    d = round(sum(graph[0]))
    for row in graph:
        buffer = round(sum(row))
        if buffer != d:
            return -1
    return d


def findGraphExpansionForRegular(graph: np.ndarray) -> np.ndarray:
    d = findDegreeForRegular(graph)
    (tr, q) = findTridiagonalization(graph)
    #(vals, vecs) = findAllEigenvalue_Tridiagonal(tr, EPSILON) #It's way slower
    (vals, vecs) = findAllEigenvalue_FastSmallApprox(tr, EPSILON)
    sort_vals = sorted(vals)
    lamb = max((abs(sort_vals[0]), abs(sort_vals[-2]))) / d
    return lamb


def ext_gcd(a: int, b: int):
    if b == 0:
        return a, 1, 0
    (d, x, y) = ext_gcd(b, a % b)
    return d, y, x - y * (b // a)


def inv(x: int, p: int) -> int:
    if x % p == 0:
        return -1
    (_, a, _) = ext_gcd(x, p)
    return a


def genGraphWithPrime(p: int) -> np.ndarray:
    graph = np.zeros((p + 1, p + 1))
    graph[p + 1][0] += 1.
    graph[0][p + 1] += 1.
    graph[p + 1][p + 1] += 2.
    for i in range(p):
        graph[(i + 1) % p][i] += 1.
        graph[(i + p - 1) % p][i] += 1.
        if i != 0:
            graph[inv(i, p)][i] += 1.
    return graph


def genGraphWithN(n: int) -> np.ndarray:
    def getIndex(x, y):
        return x % n * n + y % n
    graph = np.zeros((n ** 2, n ** 2))
    for x in range(n):
        for y in range(n):
            graph[getIndex(x, y)][getIndex(x + 2 * y, y)] += 1.
            graph[getIndex(x, y)][getIndex(x + 2 * n - 2 * y, y)] += 1.
            graph[getIndex(x, y)][getIndex(x + 2 * y + 1, y)] += 1.
            graph[getIndex(x, y)][getIndex(x + 3 * n - 2 * y - 1, y)] += 1.

            graph[getIndex(x, y)][getIndex(x, y + 2 * x)] += 1.
            graph[getIndex(x, y)][getIndex(x, y + 2 * n - 2 * x)] += 1.
            graph[getIndex(x, y)][getIndex(x, y + 2 * x + 1)] += 1.
            graph[getIndex(x, y)][getIndex(x, y + 3 * n - 2 * x - 1)] += 1.
    return graph


# Main
if __name__ == '__main__':
    print("Ответа да/нет принимают еще yes/no или однобуквенные варианты, регистронезависимо")
    eps = float(input("Введите точность вычислений: "))
    digitsAfterComma = max(math.ceil(math.log10(1/eps)), 0)
    loopContinues = True
    A = None
    while loopContinues:
        problem = int(input("Напишите номер задачи, которую хотите проверить: "))
        if problem == 1:
            print("Метод простых итераций")
            A = iom.enterMatrix(A, eps, digitsAfterComma)
            b = iom.enterVector(A.shape[0])
            print("Ответ:", simpleIteration(A, b, eps))
        elif problem == 2:
            print("Метод Гаусса-Зейделя")
            A = iom.enterMatrix(A, eps, digitsAfterComma)
            b = iom.enterVector(A.shape[0])
            print("Ответ", useGaussSeidel(A, b, eps))
        elif problem == 3:
            print("Проверка вращений Гивенса")
            A = iom.enterMatrix(A, eps, digitsAfterComma)
            i, j = map(int, input("Введите координаты i, j вращения c нуля: ").split())
            phi = float(input("Введите угол в радианах: "))
            c = math.cos(phi)
            s = math.sin(phi)
            print("cos = {}, sin = {}".format(c, s))
            print("Вращение Гивенса: ")
            iom.printMatrix(useGivensRotation(A, i, j, c, s), eps, digitsAfterComma)
        elif problem == 4:
            print("QR-разложение с помощью вращений Гивенса")
            A = iom.enterMatrix(A, eps, digitsAfterComma)
            q, r = findQR_useGivensRotation(A)
            print("Ортогональная матрица Q")
            iom.printMatrix(q, eps, digitsAfterComma)
            print("Верхнетреугольная матрица R")
            iom.printMatrix(r, eps, digitsAfterComma)
        elif problem == 5:
            print("Умножение на матрицу Хаусхолдера")
            A = iom.enterMatrix(A, eps, digitsAfterComma)
            v = iom.enterVector(A.shape[0])
            print("Ответ: ")
            iom.printMatrix(useHouseholderMultiplication(A, v), eps, digitsAfterComma)
        elif problem == 6:
            print("QR-разложение с помощью матрица Хаусахолдера")
            A = iom.enterMatrix(A, eps, digitsAfterComma)
            q, r = findQR_WithHouseholder(A)
            print("Ортогональная матрица Q")
            iom.printMatrix(q, eps, digitsAfterComma)
            print("Верхнетреугольная матрица R")
            iom.printMatrix(r, eps, digitsAfterComma)
        elif problem == 7:
            # С комплексными максимальными собственными не работает, так как у вещественной матрицы все компл. собств
            # числа сопряжены и значит не бывает единств. макс. компл числа.
            print("Нахождение максимального собственного числа")
            A = iom.enterMatrix(A, eps, digitsAfterComma)
            x = iom.enterComplexVector(A.shape[0])
            print("Максимальное время выполнение работы - 5 секунд")
            print("Максимальное собственное число: {}".format(findMaxEigenvalue(A, x, eps)))
        elif problem == 8:
            print("Все собственные числа, наивное QR-разложение")
            A = iom.enterSymmMatrix(A, eps, digitsAfterComma)
            iom.printAllEigenvalue(findAllEigenvalue_Symmetric(A, eps), eps, digitsAfterComma)
        elif problem == 9:
            print("Тридигонализация матрицы")
            A = iom.enterSymmMatrix(A, eps, digitsAfterComma)
            print("Тридиагонализация")
            newA, Q = findTridiagonalization(A)
            iom.printMatrix(newA, eps, digitsAfterComma)
            print("Матрица преобразования")
            iom.printMatrix(Q, eps, digitsAfterComma)
        elif problem == 10:
            print("Все собственные числа, для трехдиагональной матрицы")
            A = iom.enterTridiagMatrix(A, eps, digitsAfterComma)
            iom.printAllEigenvalue(findAllEigenvalue_Tridiagonal(A, eps), eps, digitsAfterComma)
        elif problem == 11:
            print("Все собственные числа для трехдиагональной матрицы с оптимизацией")
            A = iom.enterTridiagMatrix(A, eps, digitsAfterComma)
            iom.printAllEigenvalue(findAllEigenvalue_FastSmallApprox(A, eps), eps, digitsAfterComma)
        elif problem == 12:
            print("Провека графов на неизоморфность")
            G1 = iom.enterMatrix(A, eps, digitsAfterComma)
            G2 = iom.enterMatrix(A, eps, digitsAfterComma)
            print(graphIsomorphism(G1, G2))
            pass
        elif problem == 13:
            print("Нахождения альфа для графа")
            n = int(input("Введите n для генерации графа Z/n x Z/n: "))
            alpha1 = findGraphExpansionForRegular(genGraphWithN(n))
            print("Ответ: ", alpha1)
            p = int(input("Введите простое p для генерации графа для Z/p + бесконечность: "))
            alpha2 = findGraphExpansionForRegular(genGraphWithN(p))
            print("Ответ: ", alpha2)
        else:
            print("Не получилось понять номер задачи, повторите ввод")
            break
        loopContinues = iom.readYesNo("Вернуться к выбору задачи?: ")
