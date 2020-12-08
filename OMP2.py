# Orthogonal Matching Pursuit algorithm
# by Dr Suha Tuna
# 15.06.2019

# b  : signal (RHS of the linear system)
# A  : normalized dictionary
# s  : sparsity level
# e  : error tolerance (Defined but not used in this code)
# S  : support
# x  : approx. sol'n of min ||x||_0  s.t.  Ax=b

# % Size of the dictionary A
from numpy import size
from numpy.ma import zeros
from numpy.linalg import norm
import pandas as pd
import numpy as np

S = []
def OMP(b, A, s):
    # Size of the dictionary A
    global S
    m = 0  # 512 olacak değeri
    m = A.shape[0]

    print("m", m)
    # m  : # of atoms (columns of A)

    # % OMP Iterations
    k = 0
    r = b


    while (k < s):
        # k = k + 1
        # Step 1: Sweep
        innerProds = np.zeros((256,1))  # tamamen sıfır matrisle dolduruyoruz
        # A matrix i (256,512) lik bir matrix

        print("innerprods", innerProds)
        tt = 0
        while (tt < m):
            # exx1 = A.transpose()
            # exx3 = exx1[:][tt]
            # print(exx1.shape)
            # print(exx3)
            # exx4 = A[:][tt].transpose()
            exx10 = A.transpose()[:][tt]
            ex1=np.transpose(A[:][tt])
            exx7 = r

            # exx2 = innerProds[tt] #innerprodsta
            # alternatif matrix multiplication çözümü innerProds[:][tt]= np.matmul((A[:][tt].transpose()),r) #burada transpoze almamız gerekiyor

            innerProds[tt] = abs(np.matmul(ex1, r)) # 256, dönüyor
            aNormal = A[:][tt]
            aDeneme = A[tt][:]
            transposeDeneme = aDeneme.T
            aTranspose = A[:][tt]

            # innerProds[tt]=abs((A[:][tt].transpose())*(r))
            tt = tt + 1

        print(innerProds)

        # Step 2: Choose atom

       # atom = max(innerProds) # maksimum atomu seçiyoruz bu sutün olacak. Burada matlabdeki max kodunu
       #implemente etmemiz gerekiyor. matlab max kodu sutunlara göre bakıp en büyüğü alıyor
        atom=np.amax(innerProds,axis=0)




        S = []
        # Step 3: Update support
        S.append(atom)
        exx4 = atom
        exxS = S
        print("s", s)
        # Step 4: Find approx. sparse x (least squares)

        A_S =[[]]

        A_S =(A[:][s])  # max değerinin sutünunu alıyor s nin bulundugu sutunu almam gerek buraya
        #s nin indisini koymam gerek ki oradaki tüm değeri alsın
        print(type(A_S))
        print(A_S)
        first=(np.matmul((np.transpose(A_S)), A_S))
        dd=b
        second=np.matmul((np.transpose(A_S)), b)
        x_S = first/second


        # Step 5: update residual
        secondPart=A_S*x_S
        r = np.subtract(b,secondPart )
        k = k + 1
        print("RESIDUAL",r.shape[0])
        print("S",S.sort())
#S.sort()
S.sort(key = abs)
# test
# b  : signal (RHS of the linear system
# A  : normalized dictionary
# s  : sparsity level


# sparsity level
# sözlüğü l2 normdan gelen katsayıya göre yeniden böleceğiz


A = (np.random.uniform(0, 1, size=(256, 512)))  # dictionary

A_signal = (np.random.uniform(0, 1, size=(1, 512)))

normalized_array = (A - A.min(axis=0)) / (A.max(axis=0) - A.min(axis=0))
normalized_signal = (A_signal - A_signal.min(axis=0)) / (A_signal.max(axis=0) - A_signal.min(axis=0))
#normalized dictionary oluşturmak için l2 formuna göre normalizasyon alarak bölme işlemi gerçektirilicektir.
l2_norm=np.sqrt((A* A).sum(axis=0))
#l2 değerinin sıfırıncı sayısının sözlüğün a[0][i] ye bölünecek
dictionary_normalized=np.zeros((256,512))
sinyal=A[:][3]
normalized_signal=np.sqrt((sinyal* sinyal).sum(axis=0))
for i in range(256):
    for j in range(512):
        dictionary_normalized[i][j]=np.divide(A[i][j],l2_norm[i])
print(dictionary_normalized)
print(normalized_signal)

OMP(sinyal, dictionary_normalized, 5)
