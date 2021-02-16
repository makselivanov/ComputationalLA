import numpy as np
import basic_matrix_functions as bmf
import math


def isYes(ans: str) -> bool:
    if not ans.islower():
        ans.lower()
    return ans == "да" or ans == "д" or ans == "yes" or ans == "y"


def isNo(ans: str) -> bool:
    if not ans.islower():
        ans.lower()
    return ans == "нет" or ans == "н" or ans == "no" or ans == "n"


def readYesNo(question: str) -> bool:
    while True:
        ans = input(question)
        if isYes(ans):
            return True
        if isNo(ans):
            return False
        print("Не получилось понять ответ, повторите ввод")


def printMatrixWithOffset(a: np.ndarray, l: list, eps: float, digits: int) -> None:
    for i, row in enumerate(a):
        for j, x in enumerate(row):
            if abs(x) < eps:
                print('{{:{}}}'.format(l[j] + 1).format(0), end=' ')
            else:
                print('{{:{}.{}f}}'.format(l[j] + 1, digits).format(x), end=' ')
        print()
    pass


def printMatrix(a: np.ndarray, eps: float, digits: int) -> None:
    list = [0] * a.shape[0]
    for i, row in enumerate(a):
        for j, x in enumerate(row):
            list[j] = max(list[j], len(str(math.floor(x))) + digits)
    printMatrixWithOffset(a, list, eps, digits)


def printAllEigenvalue(t: tuple, eps: float, digits: int) -> None:
    eigen, a = t
    list = [0] * a.shape[0]
    for i, x in enumerate(eigen):
        list[i] = max(list[i], len(str(math.floor(x))) + digits)
    for i, row in enumerate(a):
        for j, x in enumerate(row):
            list[j] = max(list[j], len(str(math.floor(x))) + digits)
    print("Собственные числа: ")
    for i, x in enumerate(eigen):
        print('{{:{}.{}f}}'.format(list[i] + 1, digits).format(x), end=' ')
    print()
    print("Соответствующие собственные вектора в столбец: ")
    printMatrixWithOffset(a, list, eps, digits)


def enterMatrix(a, eps: float, digits: int) -> np.ndarray:
    useLast = None
    if a is not None:
        useLast = readYesNo("Использоватать предыдущую матрицу?: ")
    if useLast:
        print("Использована предыдущая матрица")
        printMatrix(a, eps, digits)
    else:
        n = int(input("Введите размер матрицы n: "))
        print("Введите матрицу A в n строк и n столбцов: ")
        a = np.ndarray((n, n))
        for i in range(n):
            for j, x in enumerate(map(float, input().split())):
                a[i, j] = x
    return a


def enterSymmMatrix(a, eps: float, digits: int) -> np.ndarray:
    useLast = False
    if a is not None and not bmf.isSymmMatrix(a, eps):
        print("Предыдущая матрица не симметричная, введите новую")
    elif a is not None:
        useLast = readYesNo("Использоватать предыдущую матрицу?: ")
    if useLast:
        print("Использована предыдущая матрица")
        printMatrix(a, eps, digits)
    else:
        isSymm = False
        while not isSymm:
            n = int(input("Введите размер матрицы n: "))
            print("Введите симметричную матрицу A в n строк и n столбцов: ")
            a = np.ndarray((n, n))
            for i in range(n):
                for j, x in enumerate(map(float, input().split())):
                    a[i, j] = x
            isSymm = bmf.isSymmMatrix(a, eps)
            if not isSymm:
                print("Матрица не симметричная, введите новую")
    return a


def enterTridiagMatrix(a, eps: float, digits) -> np.ndarray:
    useLast = False
    if a is not None and not bmf.isTridiagMatrix(a, eps):
        print("Предыдущая матрица не трехдиагональная, введите новую")
    elif a is not None:
        useLast = readYesNo("Использоватать предыдущую матрицу?: ")
    if useLast:
        print("Использована предыдущая матрица")
        printMatrix(a, eps, digits)
    else:
        isTridiag = False
        while not isTridiag:
            n = int(input("Введите размер матрицы n: "))
            print("Введите трехдиагональную матрицу A в n строк и n столбцов: ")
            a = np.ndarray((n, n))
            for i in range(n):
                for j, x in enumerate(map(float, input().split())):
                    a[i, j] = x
            isTridiag = bmf.isTridiagMatrix(a, eps)
            if not isTridiag:
                print("Матрица не трехдиагональная, введите новую")
    return a


def enterVector(n: int) -> np.ndarray:
    print("Введите вектор в одну строку из {} элементов".format(n))
    while True:
        v = np.array(list(map(float, input().split())))
        if v.shape == (n,):
            break
        print("Размер не совпал, повторите ввод")
    return v


def enterComplexVector(n: int) -> np.ndarray:
    print("Введите комплексный вектор в одну строку из {} элементов, элементы вида 1+1j".format(n))
    while True:
        v = np.array(list(map(complex, input().split())))
        if v.shape == (n,):
            break
        print("Размер не совпал, повторите ввод")
    return v
