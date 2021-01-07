import numpy as np
import pandas as pd
from scipy.io import loadmat
from sklearn import preprocessing

from scipy.io import loadmat

import numpy as np
import sklearn.preprocessing as preprocessing
import pandas as pd
from numpy import array
from numpy.linalg import norm
from sklearn.preprocessing import StandardScaler
from copy import copy, deepcopy


def OMP(b, A, s):
    # size dictionary row sayısını alıyoruz
    m = 400
    S = []
    # OMP iterations
    k = 0
    r = b
    A=A.to_numpy()

    while k < s:
        # step1 sweep
        innerProds = np.zeros((400, 1))
        # innerProds=pd.DataFrame(innerProds)

        for tt in range(0, 400):
            e = A[:, tt].T

            e = e.reshape(1, 200)

            innerProds[tt] = np.dot(e, r)

        print(innerProds)
        # step 2 choose atom
        atom = np.argmax(innerProds, axis=0, out=None)

        # step 3 update support
        S.append(atom)

        # Step 4: Find approx. sparse x (least squares)

        # max değerinin sutünunu alıyor s nin bulundugu sutunu almam gerek buraya

        A_S = A[:, S[k]]
        A_S = A_S.reshape(200, 1)
        A_S_T = A_S.T
        A_S_T = A_S_T.reshape(1, 200)
        print(A_S)
        deneme = A_S
        # s nin indisini koymam gerek ki oradaki tüm değeri alsın
        first = A_S_T.dot(A_S)

        second = A_S_T.dot(b)
        A_S = np.zeros((200, 1))
        x_S = np.divide(first, second)
        print("X_S", x_S)

        # Step 5: update residual
        secondPart = A_S * x_S
        r = np.subtract(b, secondPart)
        print("Residual", r.shape)
        k = k + 1
    S = S.sort()


X = loadmat('C:/Users/Basak/Desktop/indian_pines/Indian_pines_corrected.mat')['indian_pines_corrected']
y = loadmat('C:/Users/Basak/Desktop/indian_pines/Indian_pines_gt.mat')['indian_pines_gt']


M = np.zeros((21025, 200))
c = 0
for i in range(145):
    for j in range(145):
        M[:][c] = (X[i][j][:])
        c = c + 1
M = M.transpose()

# step 2

Mnorm = np.zeros((200, 21025))
Mnorm = M
l2_norm = preprocessing.normalize(M, norm='l2', axis=1)
l2_norm = pd.DataFrame(l2_norm)
s = 0
Mnorm = pd.DataFrame(Mnorm)

for a in range(21025):
    Mnorm[:][a] = Mnorm[:][a] / (norm(Mnorm[:][a], 2))

# step 3 200* 21025 matriste 21025lik kısımdan 400 tane secip matris.i küçültüyoruz ve bu bizim dictionaryimiz oluyor

df = pd.DataFrame(Mnorm)
# D=pd.DataFrame(np.zeros((200,400)))

D = df.sample(n=400, axis=1, replace="false")
columnsIndex = D.columns

dictionary = np.zeros((200, 400))
D.columns = [i for i in range(0, 400)]

signal = np.random.uniform(0, 1, size=(200, 1))
s = 3
A = np.random.uniform(0, 1, size=(200, 400))
OMP(signal, D, s)

