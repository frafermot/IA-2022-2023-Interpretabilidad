from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import random
from sklearn import model_selection

data = pd.read_csv('hour.csv')
data = data.drop("dteday", axis=1)
atributos = data.loc[:, 'season':'registered']
objetivo = data['cnt']

(atributos_entrenamiento, atributos_prueba,
 objetivo_entrenamiento, objetivo_prueba) = model_selection.train_test_split(
    atributos, objetivo, test_size=.33)

Iterac=2
atributosAcambiar=7

def metodo_lime(N,x,k):
    Xprima=[]
    R=[]
    W=[]
    Yprima= []
    for i in range (0,N):
        atributosAPerturbar=seleccionar_atributos_aleatorios(x,k)
        xprima = perturbar_muestra(x,atributosAPerturbar)
        w = distancia_entre_muestras(x,xprima)
        #r = representacion_muestra(atributosAPerturbar)
        r=atributosAPerturbar
        Xprima.append(xprima)
        R.append(r)
        W.append(w)
        yprima = prediccion_de_modelo(xprima)        
        Yprima.append(yprima)
        
    G = entrenar_modelo_ridge(R,Yprima,W)
    return G.getParams()

def distancia_entre_muestras(x,xprima):
    distancia = cosine_distances(x.values, xprima.values)
    print(distancia)
    return distancia

#def representacion_muestra(atributosAPerturbar):
    #return np.array([1 if i in atributosAPerturbar else 0 for i in range(len(x))])

def entrenar_modelo_ridge(R, Yprima, W):
    modelo_entrenado = Ridge(alpha=1.0) 
    modelo_entrenado.fit(R, Yprima, sample_weight=W)
    return modelo_entrenado


def perturbar_muestra(x,atributosAPerturbar):
    xprima = x.copy()
    for i in range(0,len(atributosAPerturbar)):
        if atributosAPerturbar[i] == 1:
            xprima[i] = xprima[i] + random.randint(-1,1)
    print(xprima.values)
    return xprima

def prediccion_de_modelo(xprima):
    model = RandomForestRegressor()
    model.fit(atributos_entrenamiento, objetivo_entrenamiento)
    y_pred = model.predict(xprima)
    return y_pred

def seleccionar_atributos_aleatorios(dataframe, k):
    seleccionados = random.sample(list(dataframe.columns), k)
    lista = [0 for i in range(len(dataframe.columns))]
    
    for columna in dataframe.columns:
        if columna in seleccionados:
            lista[columna] = 1
        else:
            lista[columna] = 0
    print(lista)
    return lista


nuevos_ejemplos = pd.DataFrame([[1,0,1,8,0,6,0,1,0.24,0.2879,0.75,0,1,7]])

metodo_lime(Iterac,nuevos_ejemplos,atributosAcambiar)