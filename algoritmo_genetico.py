import random
import math
import matplotlib.pyplot as plt
from sympy import lambdify, simplify, sin, cos, symbols
from statistics import mean


class Individuo:
    def __init__(self, cromosoma, fenotipo, fitness):
        self.cromosoma = cromosoma
        self.fenotipo = fenotipo
        self.fitness = fitness

class AlgoritmoGenetico:
    def __init__(self, precision, rango, limiteGeneraciones, limitePoblacion, tamanioPoblacionInicial,
                 probabilidadMutIndiv, probabilidadMutGen):
        x = symbols('x')
        expresion = ((x ** 3) / 100) * sin(x) + (x ** 2) * cos(x)
        self.function = lambdify(x, simplify(expresion))
        self.precision = precision
        self.rango = rango
        self.limiteGeneraciones = limiteGeneraciones
        self.limitePoblacion = limitePoblacion
        self.tamanioPoblacionInicial = tamanioPoblacionInicial
        self.Rx = self.rango[1] - self.rango[0]
        self.nPx = math.ceil(self.Rx / self.precision) + 1
        self.nBx = len(bin(self.nPx)) - 2
        self.poblacion = []
        self.mejoresCasos = []
        self.peoresCasos = []
        self.promedioCasos = []
        self.probabilidadMutIndiv = probabilidadMutIndiv
        self.probabilidadMutGen = probabilidadMutGen
        self.puntoCruzaFijo = self.nBx // 2  # Punto de cruza fijo

    def mutacion(self, individual):
        if random.random() <= self.probabilidadMutIndiv:
            for i in range(self.nBx):
                if random.random() < self.probabilidadMutGen:
                    # Negación del bit
                    individual[0][i] = 1 - individual[0][i]
            return self.generarIndividuo(individual[0])
        return individual

    def generarIndividuo(self, genotipo):
        i = int("".join(map(str, genotipo)), 2)
        fenotipo = self.rango[0] + i * self.precision
        fenotipo = min(fenotipo, self.rango[1])
        aptitud = self.function(fenotipo)
        return [genotipo, i, fenotipo, aptitud]

    def poda(self):
        # Ordenar la población por fitness en orden descendente
        self.poblacion.sort(key=lambda individuo: individuo[3], reverse=True)

        # Mantener al mejor individuo
        mejor_individuo = self.poblacion[0]
        nueva_poblacion = [mejor_individuo]

        # Eliminación aleatoria del resto de la población
        poblacion_restante = self.poblacion[1:]
        poblacion_restante = random.sample(poblacion_restante, min(len(poblacion_restante), self.limitePoblacion - 1))

        # Agregar los individuos seleccionados a la nueva población
        nueva_poblacion.extend(poblacion_restante)

        # Actualizar la población
        self.poblacion = nueva_poblacion

    def cruza(self, a, b, n):
        # Utilizar el punto de cruza fijo
        genotipoa = a[0][:self.puntoCruzaFijo] + b[0][self.puntoCruzaFijo:]
        genotipob = b[0][:self.puntoCruzaFijo] + a[0][self.puntoCruzaFijo:]

        return self.generarIndividuo(genotipoa), self.generarIndividuo(genotipob)

    @staticmethod
    def seleccionarPadre(poblacion):
        return random.choice(poblacion)

    def generarPoblacionInicial(self):
        for _ in range(self.tamanioPoblacionInicial):
            genotipo = random.choices([0, 1], k=self.nBx)
            individual = self.generarIndividuo(genotipo)
            self.poblacion.append(individual)

    def iniciar(self, minimize):
        self.generarPoblacionInicial()
        for _ in range(self.limiteGeneraciones):
            nuevaPoblacion = []
            for _ in range(len(self.poblacion) // 2):
                padre1 = self.seleccionarPadre(self.poblacion)
                padre2 = self.seleccionarPadre(self.poblacion)
                hijo1, hijo2 = self.cruza(padre1, padre2, 3)  # 3 es un ejemplo, ajusta según tus necesidades
                nuevaPoblacion.extend([self.mutacion(hijo1), self.mutacion(hijo2)])
            self.poblacion.extend(nuevaPoblacion)
            self.poblacion.sort(key=lambda x: x[3], reverse=minimize)
            self.mejoresCasos.append(self.poblacion[0])
            self.peoresCasos.append(self.poblacion[-1])
            self.promedioCasos.append(mean(x[3] for x in self.poblacion))
            self.poda()



        



    
        

        


        

   
