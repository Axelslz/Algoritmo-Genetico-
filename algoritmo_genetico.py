import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import math
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
        self.poblacion.sort(key=lambda individuo: individuo[3], reverse=True)
        self.poblacion = self.poblacion[:self.limitePoblacion]

    def cruza(self, a, b):
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
                hijo1, hijo2 = self.cruza(padre1, padre2)
                nuevaPoblacion.extend([self.mutacion(hijo1), self.mutacion(hijo2)])
            self.poblacion.extend(nuevaPoblacion)
            self.poblacion.sort(key=lambda x: x[3], reverse=minimize)
            self.mejoresCasos.append(self.poblacion[0])
            self.peoresCasos.append(self.poblacion[-1])
            self.promedioCasos.append(mean(x[3] for x in self.poblacion))
            self.poda()

class InterfazGrafica:
    def __init__(self, root):
        self.root = root
        self.root.title("Algoritmo Genético")
        self.root.geometry("650x400+0+0")
        self.root.configure(bg="white")
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=7)

        self.ag = None

        # Variables de la interfaz
        self.precision_var = tk.DoubleVar(value=0.05)
        self.min_x_var = tk.DoubleVar(value=-4)
        self.max_x_var = tk.DoubleVar(value=4)
        self.max_generations_var = tk.IntVar(value=10)
        self.initial_population_var = tk.IntVar(value=3)
        self.max_population_var = tk.IntVar(value=7)
        self.individual_mutation_prob_var = tk.DoubleVar(value=0.80)
        self.gen_mutation_prob_var = tk.DoubleVar(value=0.20)

        # Componentes de la interfaz
        self.crear_componentes()

    def crear_componentes(self):
        etiquetas_entradas = [
            ("Precisión para X:", self.precision_var),
            ("Valor mínimo para X:", self.min_x_var),
            ("Valor máximo para X:", self.max_x_var),
            ("Límite de generaciones:", self.max_generations_var),
            ("Población inicial:", self.initial_population_var),
            ("Población máxima:", self.max_population_var),
            ("PMI:", self.individual_mutation_prob_var),
            ("PMG:", self.gen_mutation_prob_var)
        ]

        for i, (label_text, variable) in enumerate(etiquetas_entradas, start=1):
            ttk.Label(self.root, text=label_text).grid(row=i, column=0, sticky=tk.W)
            entry = ttk.Entry(self.root, textvariable=variable)
            entry.grid(row=i, column=1, sticky=tk.W)

        ttk.Button(self.root, text="Maximizar", command=lambda: self.ejecutar_algoritmo(False)).grid(row=i+1, column=0, padx=(0, 3))
        ttk.Button(self.root, text="Minimizar", command=lambda: self.ejecutar_algoritmo(True)).grid(row=i+1, column=1, padx=(3, 0))

    def ejecutar_algoritmo(self, minimize):
        try:
            precision = float(self.precision_var.get())
            rango = [float(self.min_x_var.get()), float(self.max_x_var.get())]
            limiteGeneraciones = int(self.max_generations_var.get())
            limitePoblacion = int(self.max_population_var.get())
            tamanioPoblacionInicial = int(self.initial_population_var.get())
            probabilidadMutIndiv = float(self.individual_mutation_prob_var.get())
            probabilidadMutGen = float(self.gen_mutation_prob_var.get())

            self.ag = AlgoritmoGenetico(precision, rango, limiteGeneraciones, limitePoblacion, tamanioPoblacionInicial,
                                        probabilidadMutIndiv, probabilidadMutGen)
            self.ag.iniciar(minimize)
            self.visualizar_generaciones()

            messagebox.showinfo("Resultado", "Optimización completada")
        except ValueError as e:
            messagebox.showerror("Error", f"Error de entrada: {str(e)}")

    def visualizar_generaciones(self, pausa_entre_generaciones=0.5):
        for i in range(len(self.ag.mejoresCasos)):
            self.mostrar_estadisticas(i)
            self.root.update()
            self.root.after(int(pausa_entre_generaciones * 1000))

    def mostrar_estadisticas(self, generacion):
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))

        # Graficar el mejor caso
        axs[0, 0].plot([i[3] for i in self.ag.mejoresCasos[:generacion + 1]])
        axs[0, 0].set_title('Mejor Caso')

        # Graficar el peor caso
        axs[0, 1].plot([i[3] for i in self.ag.peoresCasos[:generacion + 1]])
        axs[0, 1].set_title('Peor Caso')

        # Graficar el promedio
        axs[1, 0].plot(self.ag.promedioCasos[:generacion + 1])
        axs[1, 0].set_title('Promedio')

        # Mostrar la población
        poblacion_fitness = [i[3] for i in self.ag.poblacion]
        axs[1, 1].hist(poblacion_fitness, bins=20, edgecolor='black')
        axs[1, 1].set_title('Distribución de Población')

        for ax in axs.flat:
            ax.set(xlabel='Generación', ylabel='Fitness')

        plt.tight_layout()

        # Mostrar el gráfico en la interfaz
        canvas = FigureCanvasTkAgg(fig, master=self.root)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.grid(row=0, column=2, rowspan=10)
        canvas.draw()

if __name__ == "__main__":
    root = tk.Tk()
    interfaz = InterfazGrafica(root)
    root.mainloop()
