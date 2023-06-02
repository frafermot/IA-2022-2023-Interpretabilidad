from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import random
from sklearn import model_selection

data = pd.read_csv('hour.csv')
atributos = data.loc[:, 'instant':'registered']
objetivo = data['cnt']
(atributos_entrenamiento, atributos_prueba,
 objetivo_entrenamiento, objetivo_prueba) = model_selection.train_test_split(
    atributos, objetivo, test_size=.33)

def metodo_lime(N,x,k):
    Xprima=[]
    R=[]
    W=[]
    Yprima=[]
    for i in range (1,N):
        atributosAPerturbar=seleccionar_atributos_aleatorios(x,k)
        xprima = perturbar_muestra(x,atributosAPerturbar)
        w = distancia_entre_muestras(x,xprima)
        r = representacion_muestra(atributosAPerturbar)
        Xprima.append(xprima)
        R.append(r)
        W.append(w)
        yprima = prediccion_de_modelo(xprima)
        Yprima.append(yprima)
    G = entrenar_modelo_ridge(R,Yprima,W)
    return G.getParams()

def distancia_entre_muestras(x,xprima):
    return cosine_similarity(x.reshape(1,-1),xprima.reshape(1,-1))

def representacion_muestra(atributosAPerturbar):
    return np.array([1 if i in atributosAPerturbar else 0 for i in range(len(x))])

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

def prediccion_de_modelo(xprima):
    model = RandomForestRegressor()
    model.fit(atributos_entrenamiento, objetivo_entrenamiento)
    y_pred = model.predict(xprima)
    return y_pred




