import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class Individuo:
    def __init__(self, cromosoma, fenotipo, fitness):
        self.cromosoma = cromosoma
        self.fenotipo = fenotipo
        self.fitness = fitness

class AlgoritmoGenetico:
    def __init__(self, formula, tamano_poblacion_inicial, tamano_poblacion_maximo, prob_mutacion_individuo, prob_mutacion_genetica,
                 rango_x_min, rango_x_max, rango_y_min, rango_y_max, max_iteraciones, porcentaje_convergencia):
        self.formula = formula
        self.tamano_poblacion_inicial = tamano_poblacion_inicial
        self.tamano_poblacion_maximo = tamano_poblacion_maximo
        self.prob_mutacion_individuo = prob_mutacion_individuo
        self.prob_mutacion_genetica = prob_mutacion_genetica
        self.rango_x_min = rango_x_min
        self.rango_x_max = rango_x_max
        self.rango_y_min = rango_y_min
        self.rango_y_max = rango_y_max
        self.max_iteraciones = max_iteraciones
        self.porcentaje_convergencia = porcentaje_convergencia
        self.individuos = []

    def evaluar_formula(self, x, y):
        try:
            resultado = eval(self.formula, {'x': x, 'y': y, 'np': np})
            return resultado
        except Exception as e:
            print(f"Error al evaluar la fórmula: {str(e)}")
            return None

    def generar_poblacion_inicial(self):
        # Genera una población inicial dentro de los rangos especificados
        x = np.random.uniform(self.rango_x_min, self.rango_x_max, self.tamano_poblacion_inicial)
        y = np.random.uniform(self.rango_y_min, self.rango_y_max, self.tamano_poblacion_inicial)
        return x, y

    def calcular_fitness(self, x, y):
        # Calcula el fitness para un individuo
        z = self.evaluar_formula(x, y)
        return z

    def cruzar_individuos(self, padre1, padre2):
        # Cruce de un punto
        punto_cruce = np.random.randint(len(padre1.cromosoma))
        hijo1_cromosoma = np.concatenate((padre1.cromosoma[:punto_cruce], padre2.cromosoma[punto_cruce:]))
        hijo2_cromosoma = np.concatenate((padre2.cromosoma[:punto_cruce], padre1.cromosoma[punto_cruce:]))
        return hijo1_cromosoma, hijo2_cromosoma

    def mutar_individuo(self, cromosoma):
        # Mutación de cambio de bit
        mascara_mutacion = np.random.rand(len(cromosoma)) < self.prob_mutacion_individuo
        cromosoma_mutado = cromosoma.copy()
        cromosoma_mutado[mascara_mutacion] = np.random.uniform(-1, 1, np.sum(mascara_mutacion))
        return cromosoma_mutado

    def ejecutar_algoritmo_genetico(self):
        # Genera población inicial
        x, y = self.generar_poblacion_inicial()

        # Criterios de paro
        iteracion = 0
        porcentaje_convergencia_actual = 0

        while iteracion < self.max_iteraciones and porcentaje_convergencia_actual < self.porcentaje_convergencia:
            # Calcula el fitness para cada individuo en la población actual
            fitness = [self.calcular_fitness(xi, yi) for xi, yi in zip(x, y)]

            # Almacena información sobre los individuos en esta iteración
            poblacion_actual = [Individuo((xi, yi), (xi, yi), fi) for xi, yi, fi in zip(x, y, fitness)]
            self.individuos.append(poblacion_actual)

            # Realiza operaciones del algoritmo genético (cruce, mutación, etc.)
            nueva_poblacion = []

            # Selección por torneo
            for _ in range(self.tamano_poblacion_inicial):
                torneo = np.random.choice(len(poblacion_actual), size=2, replace=False)
                padre1 = poblacion_actual[torneo[0]]
                padre2 = poblacion_actual[torneo[1]]

                # Cruce
                hijo1_cromosoma, hijo2_cromosoma = self.cruzar_individuos(padre1, padre2)

                # Mutación
                hijo1_cromosoma = self.mutar_individuo(hijo1_cromosoma)
                hijo2_cromosoma = self.mutar_individuo(hijo2_cromosoma)

                # Agrega los nuevos individuos a la nueva población
                nueva_poblacion.append(Individuo(hijo1_cromosoma, (hijo1_cromosoma[0], hijo1_cromosoma[1]),
                                                 self.calcular_fitness(hijo1_cromosoma[0], hijo1_cromosoma[1])))

                nueva_poblacion.append(Individuo(hijo2_cromosoma, (hijo2_cromosoma[0], hijo2_cromosoma[1]),
                                                 self.calcular_fitness(hijo2_cromosoma[0], hijo2_cromosoma[1])))

            # Reemplaza la población anterior con la nueva
            poblacion_actual = nueva_poblacion

            # Actualiza convergencia (simulada aquí con un valor fijo del 10%)
            porcentaje_convergencia_actual += 10

            iteracion += 1

        # Encuentra el individuo más apto al finalizar el algoritmo
        mejor_individuo = max(poblacion_actual, key=lambda ind: ind.fitness)

        # Muestra el mensaje de resultado
        resultado = f"¡Algoritmo Genético ejecutado con éxito!\n"
        resultado += f"Individuo más apto:\n"
        resultado += f"Cromosoma: {mejor_individuo.cromosoma}\n"
        resultado += f"Fenotipo: {mejor_individuo.fenotipo}\n"
        resultado += f"Fitness: {mejor_individuo.fitness}\n"

        print(resultado)
        return resultado

class InterfazGrafica:
    def __init__(self, root):
        self.root = root
        self.root.title("Algoritmo Genético")

        # Variables de la interfaz
        self.formula_var = tk.StringVar()
        self.tamano_poblacion_var = tk.IntVar()
        self.tamano_poblacion_maximo_var = tk.IntVar()
        self.prob_mutacion_individuo_var = tk.DoubleVar()
        self.prob_mutacion_genetica_var = tk.DoubleVar()
        self.rango_x_min_var = tk.DoubleVar()
        self.rango_x_max_var = tk.DoubleVar()
        self.rango_y_min_var = tk.DoubleVar()
        self.rango_y_max_var = tk.DoubleVar()
        self.max_iteraciones_var = tk.IntVar()
        self.porcentaje_convergencia_var = tk.DoubleVar()

        # Componentes de la interfaz
        self.crear_componentes()

    def crear_componentes(self):
        # Etiquetas
        ttk.Label(self.root, text="Ayuda (Fórmula):").grid(row=0, column=0, sticky=tk.W)
        formula_entry = ttk.Entry(self.root, textvariable=self.formula_var)
        formula_entry.grid(row=0, column=1, columnspan=2, sticky=tk.W + tk.E)

        ttk.Label(self.root, text="Tamaño de Población Inicial:").grid(row=1, column=0, sticky=tk.W)
        tamano_poblacion_entry = ttk.Entry(self.root, textvariable=self.tamano_poblacion_var)
        tamano_poblacion_entry.grid(row=1, column=1, sticky=tk.W)

        ttk.Label(self.root, text="Tamaño de Población Máximo:").grid(row=2, column=0, sticky=tk.W)
        tamano_poblacion_maximo_entry = ttk.Entry(self.root, textvariable=self.tamano_poblacion_maximo_var)
        tamano_poblacion_maximo_entry.grid(row=2, column=1, sticky=tk.W)

        ttk.Label(self.root, text="Prob. Mutación por Individuo:").grid(row=3, column=0, sticky=tk.W)
        prob_mutacion_individuo_entry = ttk.Entry(self.root, textvariable=self.prob_mutacion_individuo_var)
        prob_mutacion_individuo_entry.grid(row=3, column=1, sticky=tk.W)

        ttk.Label(self.root, text="Prob. Mutación Genética:").grid(row=4, column=0, sticky=tk.W)
        prob_mutacion_genetica_entry = ttk.Entry(self.root, textvariable=self.prob_mutacion_genetica_var)
        prob_mutacion_genetica_entry.grid(row=4, column=1, sticky=tk.W)

        ttk.Label(self.root, text="Rango X (Mínimo):").grid(row=5, column=0, sticky=tk.W)
        rango_x_min_entry = ttk.Entry(self.root, textvariable=self.rango_x_min_var)
        rango_x_min_entry.grid(row=5, column=1, sticky=tk.W)

        ttk.Label(self.root, text="Rango X (Máximo):").grid(row=6, column=0, sticky=tk.W)
        rango_x_max_entry = ttk.Entry(self.root, textvariable=self.rango_x_max_var)
        rango_x_max_entry.grid(row=6, column=1, sticky=tk.W)

        ttk.Label(self.root, text="Rango Y (Mínimo):").grid(row=7, column=0, sticky=tk.W)
        rango_y_min_entry = ttk.Entry(self.root, textvariable=self.rango_y_min_var)
        rango_y_min_entry.grid(row=7, column=1, sticky=tk.W)

        ttk.Label(self.root, text="Rango Y (Máximo):").grid(row=8, column=0, sticky=tk.W)
        rango_y_max_entry = ttk.Entry(self.root, textvariable=self.rango_y_max_var)
        rango_y_max_entry.grid(row=8, column=1, sticky=tk.W)

        ttk.Label(self.root, text="Máx. Iteraciones:").grid(row=9, column=0, sticky=tk.W)
        max_iteraciones_entry = ttk.Entry(self.root, textvariable=self.max_iteraciones_var)
        max_iteraciones_entry.grid(row=9, column=1, sticky=tk.W)

        ttk.Label(self.root, text="% Convergencia:").grid(row=10, column=0, sticky=tk.W)
        porcentaje_convergencia_entry = ttk.Entry(self.root, textvariable=self.porcentaje_convergencia_var)
        porcentaje_convergencia_entry.grid(row=10, column=1, sticky=tk.W)

        # Botón de ejecución
        ttk.Button(self.root, text="Ejecutar", command=self.ejecutar_algoritmo).grid(row=11, column=0, columnspan=3)

    def ejecutar_algoritmo(self):
        # Obtener valores de la interfaz
        formula = self.formula_var.get()
        tamano_poblacion = self.tamano_poblacion_var.get()
        tamano_poblacion_maximo = self.tamano_poblacion_maximo_var.get()
        prob_mutacion_individuo = self.prob_mutacion_individuo_var.get()
        prob_mutacion_genetica = self.prob_mutacion_genetica_var.get()
        rango_x_min = self.rango_x_min_var.get()
        rango_x_max = self.rango_x_max_var.get()
        rango_y_min = self.rango_y_min_var.get()
        rango_y_max = self.rango_y_max_var.get()
        max_iteraciones = self.max_iteraciones_var.get()
        porcentaje_convergencia = self.porcentaje_convergencia_var.get()

        # Crear instancia de AlgoritmoGenetico y ejecutar
        ag = AlgoritmoGenetico(formula, tamano_poblacion, tamano_poblacion_maximo, prob_mutacion_individuo,
                               prob_mutacion_genetica, rango_x_min, rango_x_max, rango_y_min, rango_y_max,
                               max_iteraciones, porcentaje_convergencia)
        resultado = ag.ejecutar_algoritmo_genetico()

        # Mostrar el resultado en un cuadro de mensaje
        messagebox.showinfo("Resultado", resultado)

if __name__ == "__main__":
    root = tk.Tk()
    interfaz = InterfazGrafica(root)
    root.mainloop()