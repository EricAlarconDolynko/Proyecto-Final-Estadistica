import numpy as np
from mpmath import mp
import math
import matplotlib.pyplot as plt
from scipy.stats import chi2, f, binom
from scipy.special import eval_legendre

""" 
A continuación encontrará todas las funciones necesarias par la pregunta
número 2
"""

#### SIMULAR PPPH ####

def obtener_n(lambda_actual: float, t:int) -> float:
    """ 
    Obtiene el n de la distribución Poisson(lambda T)
    """
    return np.random.poisson(t*lambda_actual)

def obtener_uniforme_aleatorio(t: int) -> float:
    """ 
    Obtiene un número aleatorio de la distribución Uniforme(0,T)
    """
    return np.random.uniform(0,t)

def simular_PPH(lambda_actual: float, t:int) -> list:
    """ 
    Simula un PPH con parametro lambda y retorna los tiempos de llegada.
    """
    n = obtener_n(lambda_actual, t)
    tiempos_llegada = []
    
    for i in range(n):
        tiempo = obtener_uniforme_aleatorio(t)
        tiempos_llegada.append(tiempo)
        
    return sorted(tiempos_llegada)

def graficar_PPH(tiempos_llegada:list, t: int) -> float:
    """ 
    Grafica un proceso de Poisson con parámetro lambda
    """
    x = [0] + tiempos_llegada + [t]
    y = [0] + list(range(1, len(tiempos_llegada)+1)) + [len(tiempos_llegada)]
    
    plt.step(x, y, where='post', linewidth=2)
    plt.xlabel('Tiempo')
    plt.ylabel('Número de ocurrencias')
    plt.title('Proceso de Poisson')
    plt.xlim(0, t)
    plt.ylim(0, len(tiempos_llegada)+1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

#### SIMULAR PPPH ####

#### PEARSON #### 

def dividir_intervalo(n: int, t:int) -> dict:
    """
    Retorna un diccionario con los distintos intervalos
    """
    k = int(math.sqrt(n))
    intervalos = {}
    pasos = t//k
    
    inicio = 0
    final = pasos
    for i in range(k):
        if i == k-1:
            intervalos[(inicio, t)] = 0
        else:
            intervalos[(inicio, final)] = 0
            inicio += pasos
            final += pasos
    return intervalos


def actualizar_info_observada(observacion: float,intervalos: dict) -> dict:
    """ 
    Actualiza cuantas observaciones cae en cada intervalo
    """
    for intervalo in intervalos:
        if intervalo[0] <= observacion <= intervalo[1]:
            intervalos[intervalo] += 1
    pass

def calcular_estadistico_pearson(intervalos: dict, n:int) -> float:
    """ 
    Retorna el resultado del estadístico de Pearson
    """
    pearson = 0
    k = int(math.sqrt(n))
    esperanza = n/k
    
    for intervalo in intervalos:
        pearson += ((intervalos[intervalo]-esperanza)**2)/(esperanza)
    
    return pearson

def valor_critico_chi(n: int) -> float:
    """ 
    Retorna el valor critico de una chi cuadrado con n grados de 
    libertad y percentil 0.95.
    """
    return chi2.ppf(0.95, n)

def prueba_pearson(lambda_actual: float, t: int, proceso_poison: list) -> bool:
    """ 
    Retorna si se acepta o no la hipotesis nula con pearson
    """
    
    n = len(proceso_poison)
    k = int(math.sqrt(n))
    intervalos = dividir_intervalo(n,t)
    
    for valor in proceso_poison:
        actualizar_info_observada(valor, intervalos)
    
    pearson = calcular_estadistico_pearson(intervalos, n)    
    chi = valor_critico_chi(k-1)
    
    if pearson < chi:
        print(f"Se ACEPTA la Hipótesis nula con un pearson de {pearson} y chi de {chi}")
        return True
    else:
        print(f"Se RECHAZA la Hipótesis nula con un pearson de {pearson} y chi de {chi}")
        return False

#### FIN PEARSON ####

#### PRUEBA SUAVE ####

def polinomio_de_legendre(k: int, u: float) -> float:
    """ 
    Retorna el resultado de Phi_k (u) para evaluar desviaciones de la uniformidad
    """
    return np.sqrt(2*k + 1) * eval_legendre(k, 2*u - 1)

def caluclar_lista_u(datos_pph: list, t:int) -> list:
    """ 
    Retorna la lista de U_i = Ti/T
    """
    lista_u = []
    for ti in datos_pph:
        u = ti/t
        lista_u.append(u)
    return lista_u

def calcular_componente_ortogonal(k: int, n: int, datos_u: list) -> float:
    """ 
    Calcula el V_k usando polinomios de legendre
    """
    vk = 0
    for i in range(n):
        vk += polinomio_de_legendre(k, datos_u[i])
    return vk*(1/math.sqrt(n))

def calcular_estadístico_prueba(lambda_actual: float, t: int, profundidad: int, proceso_poisson: list) -> float:
    """ 
    Retorna el resultado del estadístico de prueba
    """
    valor_estadistico = 0
    lista_u = caluclar_lista_u(proceso_poisson, t)
    n = len(lista_u)
    for k in range(1, profundidad+1):
        valor_estadistico += (calcular_componente_ortogonal(k,n,lista_u)**2)
        
    chi2 = valor_critico_chi(profundidad)
    if valor_estadistico < chi2:
        print(f"La hipótesis se ACEPTA con un valor del estadístico de {valor_estadistico} y un chi de {chi2}")
        return True
    else:
        print(f"La hipótesis se RECHAZA con un valor del estadístico de {valor_estadistico} y un chi de {chi2}")
        return False
        

#### FIN PRUEBA SUAVE ####

#### PROCESO NO HOMOGENEA ####

def determinar_lambda_estrella(intensidad: int, t: int) -> float:
    """ 
    Retorna el lambda estrella de acuerdo a la intensidad
    """
    if intensidad == 1:
        return 1 + 0.02*t
    else:
        if t >= 20:
            return 1.2
        else:
            return 1    
        
def calcular_intensidad(intensidad: int, t: int) -> float:
    """ 
    Calcula lambda(t) dependiendo de la intensidad
    """
    if intensidad == 1:
        return 1 + 0.02*t
    else:
        if (0 <= t < 20) or (40 <= t < 60) or (80 <= t < 100):
            return 1
        elif (20 <= t < 40) or (60 <= t < 80):
            return 1.2
    
def obtener_N(t:int, lambda_estrella: float) -> float:
    """ 
    Se obtiene el número de eventos potenciales 
    """
    return np.random.poisson(t*lambda_estrella)

def generar_ui_potencial(t: int, n: float) -> list:
    """ 
    Se simulan los tiempos potenciales U_i con distribución Uniforme(0,T)
    """
    lista_u = []
    for i in range(n):
        u = obtener_uniforme_aleatorio(t)
        lista_u.append(u)
    return lista_u
    
def generar_hi_potencial(lambda_estrella: int, n: float) -> list:
    """ 
    Se simulan los umbrales de aceptación h_i con distribución Uniforme(0, lambda_estrella)
    """
    lista_h = []
    for i in range(n):
        h = obtener_uniforme_aleatorio(lambda_estrella)
        lista_h.append(h)
    return lista_h

def ajustar_mt(t: float, intensidad: int) -> list:
    """ 
    Esta función ajsuta los ui con el m(t)
    """
    if intensidad == 1:
        return t + (0.01 * ((t)**2))
    else:
        if 0 <= t < 20:
            return t
        elif 20 <= t < 40:
            return 20 + (1.2*(t-20))
        elif 40 <= t < 60:
            return 44 + (t-40)
        elif 60 <= t < 80:
            return 64 + (1.2*(t-60))
        elif 80 <= t <= 100:
            return 88 + (t-80)
        
    
def simular_ppnh(t: int, intensidad: int) -> list:
    """ 
    Simula un proceso de poisson no Homogeneo
    """
    lambda_estrella = determinar_lambda_estrella(intensidad, t)
    n = obtener_N(t, lambda_estrella)
    lista_u = generar_ui_potencial(t, n)
    lista_h = generar_hi_potencial(lambda_estrella, n)
    
    lista_ppnh = []
    
    for i in range(n):
        u = lista_u[i]
        h = lista_h[i]
        intensidad_t = calcular_intensidad(intensidad, u)
        if h < intensidad_t:
            lista_ppnh.append(u)
    
    lista_ppnh_ordenada = sorted(lista_ppnh)
        
    return lista_ppnh_ordenada

#### FIN PROCESO NO HOMOGENEO ####
    
#### POTENCIA ####

def potencia_estadistico(prueba: str, t:int, intensidad: int) -> float:
    """ 
    Calcula la potencia
    """
    rechazos_correctos = 0
    for i in range(3000):
        poison_sucio = simular_ppnh(t, intensidad)
        poison = []
        for poi in poison_sucio:
            poi = ajustar_mt(t, intensidad)
            poison.append(poi)
            
        if prueba == "Pearson":
            decision = prueba_pearson(1, t, poison)
            if decision is False:
                rechazos_correctos += 1
        
        elif prueba == "Suave":
            decision = calcular_estadístico_prueba(1, t, 4, poison)
            if decision is False:
                rechazos_correctos += 1
                
        print(f"Iteración {i}")
    
    potencia = rechazos_correctos/3000
    print(f"La potencia fue de {potencia} con {rechazos_correctos} rechazos correctos")
    
    return potencia
    
#### FIN DE POTENCIA ####

#### PRUEBAS HOMOGENEAS ####

def obtener_exponencial_aleatorio(lambda_estrella: float) -> float:
    """ 
    Retorna un valor de Exponencial(lambda)
    """
    return np.random.exponential(lambda_estrella)

def obtener_pph_exponencial(lambda_acual: float, n: int) -> float:
    """ 
    Retorna el pph con la escogencia exponencial, y los tiempos entre llegada
    """
    lista_s = []
    for i in range(n):
        s = obtener_exponencial_aleatorio(lambda_acual)
        lista_s.append(s)
    
    tiempos_de_llegada = []
    for i in range(n):
        ti = 0
        for j in range(i):
            ti += lista_s[j]        
        tiempos_de_llegada.append(ti)
    
    return lista_s

def valor_critico_F(critico: float, grado1: int, grado2: int) -> float:
    """ 
    Retorna el valor de una Fcritico,grado1,grado2
    """
    return f.ppf(critico,grado1,grado2)

def prueba_r(tiempo_llegad1:list, tiempo_llegada2: list):
    """ 
    Realiza el estadístico de prueba para aceptar o rechazar la hipótesis
    """
    n = len(tiempo_llegad1)
    m = len(tiempo_llegada2)
    sumax = 0
    for tiempo in tiempo_llegad1:
        sumax += tiempo
        
    sumay = 0
    for tiempo in tiempo_llegada2:
        sumay += tiempo
        
    r = sumax / sumay
    
    valor_critico_lower = valor_critico_F(0.025,2*n,2*m)
    valor_critico_upper = valor_critico_F(0.975,2*n,2*m)
    if r < valor_critico_lower or r > valor_critico_upper:
        print(f"Se RECHAZA la hipótesis nula con un r de {r} y lower de {valor_critico_lower} y upper de {valor_critico_upper}")
        return False
    else:
        print(f"Se ACEPTA la hipótesis nula con un r de {r} y lower de {valor_critico_lower} y upper de {valor_critico_upper}")
        return True
    

#### FIN PRUEBAS HOMOGENEAS ####

#### PRUEBA BINOMIAL ####

def valor_critico_binomial(alpha:float, n: int, p: float) -> float:
    """ 
    Retorna el valor crítico para una Binom_alpha,n,p
    """
    return binom.ppf(alpha,n,p)

def prueba_comparacion(proceso1:list, proceso2:list) -> bool:
    """ 
    Realiza el estadístico para aceptar o rechazar la hipótesis
    """
    n = len(proceso1)
    m = len(proceso2)
    
    s = 0
    for tiempo in proceso1:
        s += tiempo
        
    t = 0
    for tiempo in proceso2:
        t += tiempo
        
    p = m/(m+n)
    k_lower = valor_critico_binomial(0.025, int(s+t), p)
    k_upper = valor_critico_binomial(0.975, int(s+t), p)
    
    if s <= k_lower or s >= k_upper:
        print(f"Se RECHAZA la hipótesis nula con s {s} y con lower {k_lower} y upper {k_upper}")
        return False
    else:
        print(f"Se ACEPTA la hipótesis nula con s {s} y con lower {k_lower} y upper {k_upper}")
        return True
             
#### FIN PRUEBA BINOMIAL ####

#### SOLUCIÓN #### 

#Punto 1
lamda = 1
t = 5   
intensidad = 1
#tiempos_llegada = simular_PPH(lamda, t)
#print(tiempos_llegada)
#graficar_PPH(tiempos_llegada, t)

#Punto 2.1
#prueba = prueba_pearson(lamda, t)

#Punto 2.2
#prueba = calcular_estadístico_prueba(lamda, t, 4)

#Punto 3
#tiempo_llegada = simular_ppnh(t, intensidad)
#graficar_PPH(tiempo_llegada, t)

#Punto 4.1
#a = potencia_estadistico("Pearson", t, intensidad)

#Punto 4.2
#b = potencia_estadistico("Suave", t, intensidad)
#print(f"La potencia de Pearson fue: {a}, la potencia del suave fue {b}")

#Punto 5
#primer_pph = obtener_pph_exponencial(1, 1000)
#segundo_pph = obtener_pph_exponencial(2, 1000)
#prueba_r(primer_pph, segundo_pph)

#Punto 6
primer_pph = obtener_pph_exponencial(1, 1000)
segundo_pph = obtener_pph_exponencial(1, 1000)
prueba_comparacion(primer_pph, segundo_pph)