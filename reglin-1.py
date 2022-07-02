from math import *
from random import *

def inv(mat):
    a = mat[0][0]
    b = mat[0][1]
    c = mat[1][0]
    d = mat[1][1]
    
    det = 1.0 / (a * d - b * c)
    return [[det * d, -det * b], [-det * c, det * a]]

def linearRegression(values):
    a1 = 0
    b1 = 0
    c1 = 0
    a2 = 0
    b2 = 0
    c2 = 0

    for p in points:
        a1 += 2 * p[0] * p[0]
        b1 += 2 * p[0]
        c1 += 2 * p[0] * p[1]

        a2 += 2 * p[0]
        b2 += 2
        c2 += 2 * p[1]

    mat = inv([[a1, b1],[a2, b2]])
    mat = prod(mat, [[c1], [c2]])
    a = mat[0][0]
    b = mat[1][0]

    return a, b

def prod(mat1, mat2):
    assert(len(mat1[0]) == len(mat2))
    res = []
    for i in range(len(mat1)):
        res.append([])
        for j in range(len(mat2[0])):
            res[i].append(0)
            for k in range(len(mat2)):
                res[i][j] += mat1[i][k] * mat2[k][j]
    return res

if __name__ == '__main__':
    points = []
