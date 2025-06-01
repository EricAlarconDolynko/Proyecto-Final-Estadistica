import numpy as np
from mpmath import mp
import math
import matplotlib.pyplot as plt
import random

""" 
A continuación encontrará todas las funciones necesarias para
la pregunta número 1, donde los Datos son la representación de la tabla
proporcionada en el enunciado.
"""

datos_originales = [[53,1], [57,1], [58,1], [63,1], [66,0], [67,0], [67,0], [67,0], [68,0], [69,0], [70,0], [70,0], [70,1], [70,1], [72,0], [73,0], [75,0], [75,1], [76,0], [76,0], [78,0], [79,0], [81,0]]

#### EMV ####

def p_de_x(indice: int, theta_k: list, muestra: list) -> float:
    """ 
    Calcula p(x_i)
    """
    alfa = theta_k[0]
    beta = theta_k[1]
    numerador = math.exp(alfa + (beta*muestra[indice][0]))
    denominador = 1 + math.exp(alfa + (beta*muestra[indice][0]))
    return numerador/denominador

def S_de_theta(theta_k: list, muestra: list) -> list:
    """ 
    Calcula la matriz 2x1 de S(teta)
    """
    S_alfa = 0
    S_beta = 0
    for i in range(len(muestra)):
        S_alfa += (muestra[i][1] - p_de_x(i, theta_k, muestra))
        S_beta += (muestra[i][0]) * (muestra[i][1] - p_de_x(i, theta_k, muestra))
        
    return [S_alfa, S_beta]

def H_de_theta(theta_k: list, muestra: list) -> list:
    """ 
    Calcula la matriz de 2x2 que representa H(teta)
    """
    derivada_alfa = 0
    derivada_beta = 0
    derivada_alfa_beta = 0
    for i in range(len(muestra)):
        derivada_alfa += p_de_x(i, theta_k, muestra) * (1-p_de_x(i, theta_k, muestra))
        derivada_alfa_beta += (muestra[i][0]*p_de_x(i, theta_k, muestra))*(1-p_de_x(i, theta_k, muestra))
        derivada_beta += (muestra[i][0]**2)*(p_de_x(i,theta_k, muestra))*(1-p_de_x(i,theta_k, muestra))
    
    return [[derivada_alfa,derivada_alfa_beta],[derivada_alfa_beta,derivada_beta]]

def inversa_H_de_theta(h:list) -> list:
    """ 
    Calcula la inversa de H(teta)
    """
    
    a = h[0][0]
    b = h[0][1]
    c = h[1][0]
    d = h[1][1]
    
    determinante = a*d - b*c
    if determinante == 0:
        determinante = 1
    return [[d/determinante, -b/determinante], [-c/determinante, a/determinante]]

def multiplcar_h_inversa_s_theta(h_inv: list, s: list) -> list:
    """ 
    Calcula H^-1(theta) * S(theta)
    """
    primer_componente = (h_inv[0][0] * s[0]) + (h_inv[0][1] * s[1])
    segunda_componente = (h_inv[1][0] * s[0]) + (h_inv[1][1] * s[1])
    return [primer_componente, segunda_componente]

def sumar_matrices(matrizA: list, matrizB: list) -> list:
    return [matrizA[0]+matrizB[0], matrizA[1]+matrizB[1]]
    
def newton_raphson(iteraciones: int, alfa_0: float, beta_0:float, muestra: list) -> list:
    """ 
    Calcula el emv con newton-raphson dado un número de iteraciones
    """
    theta_actual = [alfa_0, beta_0]
    lista_theta = []
    
    for i in range(iteraciones):
        h_theta = H_de_theta(theta_actual, muestra)
        s_theta = S_de_theta(theta_actual, muestra)
        theta_next = sumar_matrices(theta_actual , multiplcar_h_inversa_s_theta(inversa_H_de_theta(h_theta), s_theta))
        lista_theta.append(theta_next)
        theta_actual = theta_next
        
    
    return lista_theta[-1]

#### FIN EMV #### 

#### PROBABILIDADES EMV ####


def probabilidad_fallo(x:int, alfa:float, beta: float) -> float:
    """ 
    Retorna la probabilidad de fallo de acuerdo a la formual 4.58 del enunciado
    usando el emv de alfa y beta.
    """
    numerador = math.exp(alfa + (beta*x))
    denominador = 1 + math.exp(alfa + (beta*x))
    return numerador/denominador


#### FIN PROBABILIDADES EMV ####

#### BOOTSTRAPING ####

def generar_muestras_bootstrap(iteraciones: int, muestra: list) -> list:
    """ 
    Genera las muestras bootstrap con n iteraciones
    """
    lista_bootstrap = []
    
    for i in range(iteraciones):
        bootstrap = []
        for j in range(len(muestra)):
            indice_random = random.randint(0, len(muestra)-1)
            bootstrap.append(muestra[indice_random])
        lista_bootstrap.append(bootstrap)
    
    return lista_bootstrap

def metodo_de_efron(muestras: list, temperatura: int, iteraciones: int) -> list:
    """ 
    Encuentra los 1000 qb ordenados ascendentemente para encontrar 
    el percentil 10% usando el método proporcionado en el anexo
    """
    lista_q = []
    i = 1
    for muestra in muestras:
        emv_muestra = newton_raphson(iteraciones, -0.85, -0.001, muestra)
        alfa = emv_muestra[0]
        beta = emv_muestra[1]
        qb = alfa + (beta*temperatura)
        lista_q.append(qb)
        print(i)
        i += 1
    
    return sorted(lista_q)
    
def calcular_limite(q: float):
    numerador = math.exp(q)
    denominador = 1 + math.exp(q)
    return numerador/denominador

#### FIN BOOTSTRAPING ####

### SEGUNDA PARTE ###

#### MONTE CARLO ####

def beta_de_xi(xi: float) -> float:
    """ 
    Calucla el beta dado el xi
    """
    return -math.exp(xi)

def prior_alfa(alfa_estrella: float) -> float:
    """ 
    Calcula el prior de alfa evaluado en alfa estrella
    """
    numerador = math.exp(  (-(alfa_estrella-15.04)**2) / 2*10 )
    denominador = math.sqrt(2*math.pi *10)
    return numerador / denominador
    
def prior_xi(xi_estrella: float) -> float:
    """ 
    Calcula el prior de xi evaluado en xi estrella
    """
    numerador = math.exp(  (-(xi_estrella+1.46)**2) / 2*0.5 )
    denominador = math.sqrt(2*math.pi *0.5)
    return numerador / denominador

def conditional_likelihood(datos, alfa, beta):
    """ 
    Calcula el l(y |x , alfa, beta)
    """    
    multi_lieklihood = 1
    for i in range(len(datos)):
        x = datos[i][0]
        y = datos[i][1]
        multi_lieklihood *= (pow(probabilidad_fallo(x,alfa,beta),y) * pow((1-probabilidad_fallo(x,alfa,beta)),(1-y)))
    return multi_lieklihood

def generar_propuesta_alfa(alfa_anterior):
    """ 
    Calcula el alfa* propuesto de acuerdo a la Normal(alfa_anterior, 10)
    """
    return np.random.normal(alfa_anterior, 10)

def generar_propuesta_xi(xi_anterior):
    """ 
    Calcula el xi* propuesto de acuerdo a la Normal(xi_anterior, 0.5)
    """
    return np.random.normal(xi_anterior, 0.5)

def generar_u_aleatorio():
    """
    Retorna un valor aleatorio de la Uniforme(0,1) 
    """
    return np.random.uniform(0,1)


def ratio_aceptacion_alfa(datos,alfa_estrella,alfa_anterior,beta_anterior):
    """
    Retorna el ratio de aceptación para alfa 
    """  
    pi_alfa_estrella = conditional_likelihood(datos, alfa_estrella, beta_anterior) * prior_alfa(alfa_estrella)
    pi_alfa_anterior = conditional_likelihood(datos, alfa_anterior, beta_anterior)
    ratio = pi_alfa_estrella/pi_alfa_anterior
    return min(ratio, 1)

def ratio_aceptacion_xi(datos, xi_estrella, xi_anterior, alfa_anterior):
    """ 
    Retorna el ratio de aceptación para xi
    """
    pi_xi_estrella = (conditional_likelihood(datos, alfa_anterior, beta_de_xi(xi_estrella)) * prior_xi(xi_estrella))
    pi_xi_anterior = (conditional_likelihood(datos, alfa_anterior, beta_de_xi(xi_anterior)) * prior_xi(xi_anterior))
    ratio = pi_xi_estrella/pi_xi_anterior
    return min(ratio,1)
    
    
def metodo_mcmc(datos, alfa_inicial, xi_inicial, iteraciones, burn_in):
    """ 
    Calcula el metodo mcmc para la distribución prior
    """
    alfa_samples = []
    beta_samples = []
    alfa_anterior = alfa_inicial
    xi_anterior = xi_inicial
    
    for i in range(iteraciones):
        alfa_estrella = generar_propuesta_alfa(alfa_inicial)
        a_de_alfa = ratio_aceptacion_alfa(datos, alfa_estrella, alfa_anterior, beta_de_xi(xi_anterior))
        u_aleatorio = generar_u_aleatorio()
        
        if u_aleatorio <= a_de_alfa:
            alfa_anterior = alfa_estrella
            
        if iteraciones > burn_in:
            alfa_samples.append(alfa_anterior)
        
        xi_estrella = generar_propuesta_xi(xi_anterior)
        a_de_xi = ratio_aceptacion_xi(datos, xi_estrella, xi_anterior, alfa_anterior)
        u_aleatorio = generar_u_aleatorio()
        
        if u_aleatorio <= a_de_xi:
            xi_anterior = xi_estrella
            
        if iteraciones > burn_in:
            beta_samples.append(beta_de_xi(xi_anterior))
        
        print(f"Iteración {i}")
    return [alfa_samples, beta_samples]

def mostrar_histograma_prior_alfa(alfa_samples: list) -> list:
    """ 
    Grafica la distribución posterior de alfa
    """
    plt.hist(alfa_samples, bins=50, density=True)
    plt.title("Distribución Posterior de alpha")
    plt.show()
    
def mostrar_histograma_prior_beta(beta_samples: list) -> list:
    """ 
    Graficar la distribución posterior de beta
    """
    plt.hist(beta_samples, bins=50, density=True)
    plt.title("Distribución Posterior de beta")
    plt.show()

#### FIN MONTECARLO ####

#### ANALISIS MONTECARLO ####

def distribucion_probabilidad_fallo(montecarlo: list, temperatura: int) -> list:
    """ 
    Grafica la distribución de la probabilidad de fallo a temperatura x.
    """
    lista_alfas = montecarlo[0]
    lista_betas = montecarlo[1]
    
    lista_temperatura = []
    
    for i in range(len(lista_alfas)):
        lista_temperatura.append(probabilidad_fallo(temperatura, lista_alfas[i], lista_betas[i]))
    
    plt.hist(lista_temperatura, bins=50, density=True)
    plt.title(f"Distribución Posterior de Probabilidad de Fallo en Temperatura {temperatura} F")
    plt.show()
    
    return lista_temperatura
    
def encontrar_delta(lista_temperatura: list) -> list:
    """ 
    Calcula el percentil 10% para probabilidad de fallo de una temperatura dada
    """
    ordenada = sorted(lista_temperatura)
    percentil_10 = int(len(lista_temperatura) * 0.1)
    return ordenada[percentil_10-1]
    
#### FIN ANALISIS MONTECARLO ####    

### SOLUCIÓN ###

#PARTE 1

#Punto 1.1.1
#emv = newton_raphson(10000, -0.85, -0.001, datos_originales)
#print(f"El emv es: {emv}")

#Punto 1.1.2
#temperatura = 65
#probabilidad = probabilidad_fallo(temperatura, emv[0], emv[1])
#print(f"La probabilidad de fallo en temperatura {temperatura} F es de {probabilidad*100}%")

#Punto 1.3
#muestra_bootstrap = generar_muestras_bootstrap(1000, datos_originales)
#efron = metodo_de_efron(muestra_bootstrap, 65, 1000)
#limite_inferior = calcular_limite(efron[899])
#print(f"El limite inferior es {limite_inferior}")

#PARTE 2

#Punto 2.1
montecarlo = metodo_mcmc(datos_originales, 15.04, -1.46, 1000000, 2000)
mostrar_histograma_prior_alfa(montecarlo[0])
mostrar_histograma_prior_beta(montecarlo[1])
fallo_31 = distribucion_probabilidad_fallo(montecarlo, 31)
fallo_65 = distribucion_probabilidad_fallo(montecarlo, 65)
delta_31 = encontrar_delta(fallo_31)
delta_65 = encontrar_delta(fallo_65)
print(f"El percentil en temperatura 31 es de {delta_31}")
print(f"El percentil en temperatura 65 es de {delta_65}")


