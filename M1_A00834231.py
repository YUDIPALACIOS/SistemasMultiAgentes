# ----------------------------------------------------------------------------------------------
# Modelación de sistemas multiagentes con gráficas computacionales
# Alumna: Yudith Korina Hernández Palacios, A00834231
# Fecha de última actualización: 24 de febrero de 2024
# ----------------------------------------------------------------------------------------------
#                                   DESARROLLO DE LA SOLUCIÓN
# ----------------------------------------------------------------------------------------------
# ______________________________________________________________________________________________
#                                            LIBRERIAS
# ----------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.lines as mlines
from mesa import Agent, Model
from mesa.space import MultiGrid
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from timeit import default_timer as timer

valueLimit = 5000

class MultiAgentSystem(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.steps = 0  # Inicializa el contador de pasos del agente

    def clean(self):
        # Verifica si la celda actual está sucia y la limpia
        if self.model.floor[self.pos[0]][self.pos[1]] == 1:
            self.model.floor[self.pos[0]][self.pos[1]] = 0

    def move_randomly(self):
        # Define las posibles direcciones de movimiento
        options = np.array([[-1, -1], [-1, 0], [-1, 1],
                             [0, -1], [0, 1],
                             [1, -1], [1, 0], [1, 1]])
        # Elige una dirección aleatoria para moverse
        i = int(np.random.rand() * valueLimit) % len(options)
        x = self.pos[0] + options[i][0]
        y = self.pos[1] + options[i][1]
        # Se mueve si la celda vecina está dentro de los límites del grid
        if self.can_move(x, y):
            self.model.grid.move_agent(self, (x, y))
            self.model.stepCounter += 1
            self.steps += 1

    def can_move(self, x, y):
        # Verifica si la celda vecina está dentro de los límites del grid
        return (x >= 0 and x < self.model.grid.width and
                y >= 0 and y < self.model.grid.height)

    def step(self):
        # Realiza el proceso de limpieza y movimiento en cada paso
        self.clean()
        self.move_randomly()

def get_grid(model):
    # Obtiene el estado de las celdas (limpias, sucias)
    grid = np.zeros((model.grid.width, model.grid.height))
    for x in range(model.grid.width):
        for y in range(model.grid.height):
            if model.grid.is_cell_empty((x, y)):
                grid[x][y] = model.floor[x][y] * 2
            else:
                grid[x][y] = 1
    return grid

class MultiAgentModel(Model):
    def __init__(self, width, height, num_agents, dirty_cells_percentage=0.5):
        super().__init__()
        self.num_agents = num_agents
        self.dirty_cells_percentage = dirty_cells_percentage
        self.grid = MultiGrid(width, height, True)
        self.schedule = RandomActivation(self)
        self.floor = np.zeros((width, height))
        self.width = width
        self.height = height
        self.stepCounter = 0
        self.timer = 0

        # Creación de agentes y colocación en el grid
        for i in range(self.num_agents):
            a = MultiAgentSystem(i, self)
            self.grid.place_agent(a, (0, 0))
            self.schedule.add(a)

        # Inicialización de las celdas sucias de manera aleatoria
        amount = int((width * height) * dirty_cells_percentage)
        for i in range(amount):
            finished = False
            while not finished:
                x = int(np.random.rand() * valueLimit) % width
                y = int(np.random.rand() * valueLimit) % height
                if self.floor[x][y] == 0:
                    self.floor[x][y] = 1
                    finished = True

        self.datacollector = DataCollector(model_reporters={"Grid": get_grid})

    def is_all_clean(self):
        return np.all(self.floor == 0)

    def calculate_clean_cells(self):
        # Calcula el porcentaje de celdas limpias
        total_cells = self.width * self.height
        clean_cells = int((total_cells) - np.count_nonzero(self.floor == 1))
        clean_percentage = (clean_cells * 100) / total_cells
        return int(clean_percentage)

    def total_steps(self):
        return self.stepCounter

    def total_time(self):
        return self.timer

    def step(self):
        start = timer()
        # Recopila información del modelo
        self.datacollector.collect(self)
        # Ejecuta un paso de la simulación para cada agente
        self.schedule.step()
        end = timer()
        delta_time = end - start
        self.timer += delta_time

# Configuración inicial
heightGrid = 15
widthGrid = 15
dirtyCellsPercentage = 0.3
agentsAmount = 5
generationLimit = 500

# Crear el modelo y ejecutar la simulación
model = MultiAgentModel(heightGrid, widthGrid, agentsAmount, dirtyCellsPercentage)
agentsTotalAmount = []
agentsTotalSteps = []
executionTime = []

for k in range(30):
    model = MultiAgentModel(heightGrid, widthGrid, agentsAmount, dirtyCellsPercentage)
    i = 1
    lastTime = 0
    while not model.is_all_clean():
        model.step()
        i += 1
        lastTime += model.total_time()

    print("-----------------------------------------------------------")
    print(" Sistema multiagente de limpieza se está ejecutando con éxito")
    print(f"El porcentaje de celdas limpias es: {model.calculate_clean_cells()}%")
    print(f"La cantidad total de pasos hechos por los agentes de limpieza es: {model.total_steps()}")
    print("El tiempo total de ejecución es: %.2f s" % lastTime)
    print(f" La cantidad de agentes es: {agentsAmount}")
    print(" Fin del informe")
    print("-----------------------------------------------------------")
    agentsTotalAmount.append(agentsAmount)
    agentsTotalSteps.append(model.total_steps())
    executionTime.append(float(f'{lastTime:.2f}'))

    agentsAmount += 10

all_grid = model.datacollector.get_model_vars_dataframe()

# Animación
fig, axs = plt.subplots(figsize=(10, 10))
axs.set_xticks([])
axs.set_yticks([])
patch = plt.imshow(all_grid.iloc[0][0], cmap=plt.cm.binary)

def animate_func(i):
    patch.set_data(all_grid.iloc[i][0])

anim = animation.FuncAnimation(fig, animate_func, frames=generationLimit)
plt.show()

# ----------------------------------------------------------------------------------------------
#                                      FIN DE LA SOLUCIÓN
# ----------------------------------------------------------------------------------------------
