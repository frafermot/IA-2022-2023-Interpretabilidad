#Importación de paquetes y lectura de csv:
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity,cosine_distances
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import random
from sklearn import model_selection
from scipy.spatial.distance import euclidean
from scipy.stats import spearmanr

data = pd.read_csv('hour.csv')
data = data.drop("dteday", axis=1)
data = data.drop("instant", axis=1)
atributos = data.loc[:, 'season':'registered']
objetivo = data['cnt']

(atributos_entrenamiento, atributos_prueba,
 objetivo_entrenamiento, objetivo_prueba) = model_selection.train_test_split(
    atributos, objetivo, test_size=.33)

atributos_entrenamiento = atributos_entrenamiento.values

rangos = {}

for columna in data.columns:
    valor_min = data[columna].min()
    valor_max = data[columna].max()
    rangos[columna] = {'min': valor_min, 'max': valor_max}


def metodo_lime(N,x,k,model):
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
        yprima = prediccion_de_modelo(xprima,model)        
        Yprima.append(yprima)
    W=np.array(W)
    W = W.reshape(W.shape[0])
    G = entrenar_modelo_ridge(R,Yprima,W)
    return G.coef_

#Se seleccional los k atributos aleatorios a modificar
def seleccionar_atributos_aleatorios(dataframe, k):
    seleccionados = random.sample(list(dataframe.columns), k)
    lista = [0 for i in range(len(dataframe.columns))]
    
    for columna in dataframe.columns:
        if columna in seleccionados:
            lista[columna] = 1
        else:
            lista[columna] = 0
    return lista

#def perturbar_muestra(x,atributosAPerturbar):
#    xprima = x.copy()
#    for i in range(0,len(atributosAPerturbar)):
#        if atributosAPerturbar[i] == 1:
#            xprima[i] = xprima[i] + random.randint(-1,1)
#    return xprima

#Se perturba la muestra original de forma que los atributos seleccionados pueden variar en un rango en función del valor máximo y mínimo de dicho atributo
def perturbar_muestra(x, atributosAPerturbar):
    xprima = x.copy()
    for i in range(0,len(atributosAPerturbar)):
        if atributosAPerturbar[i] == 1:
            rango_min = xprima[i] - rangos[list(rangos.keys())[i]]['min']
            rango_max = rangos[list(rangos.keys())[i]]['max'] - xprima[i]
            xprima[i] = random.uniform(rango_min, rango_max)
            #xprima[i] = xprima[i] + perturbacion
    return xprima

# Se calcula la distancia coseno entre la muestra original y la perturbada
def distancia_entre_muestras(x,xprima):
    return cosine_distances(x.values, xprima.values)

#def representacion_muestra(atributosAPerturbar):
#    return np.array([1 if i in atributosAPerturbar else 0 for i in range(len(x))])

#Se predice el resultado de la muestra con el modelo elegido
def prediccion_de_modelo(xprima,model):
    y_pred = model.predict(xprima)
    return y_pred

#Se entrena el modelo de regresión Ridge con las muestras perturbadas y los pesos calculados
def entrenar_modelo_ridge(R, Yprima, W):
    modelo_entrenado = Ridge(alpha=1.0) 
    modelo_entrenado.fit(R, Yprima, sample_weight=W)
    return modelo_entrenado


#Probar el método LIME
nuevos_ejemplos = pd.DataFrame([[1,0,1,8,0,6,0,1,0.24,0.2879,0.75,0,1,7]])

Iterac=100
atributosAcambiar=5
modelRandomForest = RandomForestRegressor(n_estimators=100)
modelRandomForest.fit(atributos_entrenamiento, objetivo_entrenamiento)
print("Método LIME con Random Forest: ",metodo_lime(Iterac,nuevos_ejemplos,atributosAcambiar,modelRandomForest))


# Identidad

def identidad(a, b):
    epsilon_a = np.finfo(a.dtype).eps
    epsilon_b = np.finfo(b.dtype).eps
    return euclidean(a, b) == 0 and euclidean(epsilon_a, epsilon_b) == 0

# Separabilidad

def separabilidad(a, b):
    epsilon_a = np.finfo(a.dtype).eps
    epsilon_b = np.finfo(b.dtype).eps
    return euclidean(a, b) != 0 and euclidean(epsilon_a, epsilon_b) > 0

# Estabilidad

def estabilidad(x):
    d_x = np.array([euclidean(z - y) for y in x for z in x])
    d_epsilon = np.array([euclidean(np.finfo(z.dtype).eps - np.finfo(y.dtype).eps for y in x) for z in x])
    return spearmanr(d_x, d_epsilon) > 0

# Selectividad

def selectividad():
    return null

# Coherencia

def coherencia():
    return null

# Completitud

def completitud():
    return null

# Congruencia

def congruencia():
    return null