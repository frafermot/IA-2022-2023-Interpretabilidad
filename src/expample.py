from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
import pandas as pd
import random

data = pd.read_csv('hour.csv')
atributos = data.loc[:, 'instant':'registered']
objetivo = data['cnt']


def metodo_lime(N,x,k):
    Xprima=[]
    R=[]
    W=[]
    Yprima=[]
    for i in range (1,N):
        atributosAPerturbar=seleccionar_atributos_aleatorios(x,k)
        xprima = perturbar_muestra(x,atributosAPerturbar)
        w = distancia_entre_muestras(x,xprima)
        r = representación_muestra(x,xprima)
        Xprima.append(xprima)
        R.append(r)
        W.append(w)
        yprima = predicción_de_modelo(xprima)
        Yprima.append(yprima)
    G = entrenar_modelo_ridge(R,Yprima,W)
    return G.getParams()

def distancia_entre_muestras(x,xprima):
    return cosine_similarity(x.reshape(1,-1),xprima.reshape(1,-1))

def entrenar_modelo_ridge(R, Yprima, W):
    modelo_entrenado = Ridge(alpha=1.0) 
    modelo_entrenado.fit(R, Yprima, sample_weight=W)
    return modelo_entrenado

def seleccionar_atributos_aleatorios(x,k):
    return random.sample(range(len(x)), k)

def perturbar_muestra(x,atributosAPerturbar):
    xprima = x
    for atributo in atributosAPerturbar:
        xprima[atributo] = xprima[atributo] + random.randint(-1,1)
    
    return xprima


