
import os
import sys

# Para mantener el código propio y el tomado externamente separados, rutamos los directorios.
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTIAGENT_DIR = os.path.join(ROOT_DIR, "multiagent")

# Añadimos multiagent al path.
sys.path.insert(0, MULTIAGENT_DIR)
os.chdir(MULTIAGENT_DIR)


import numpy as np
from game import Agent
import pacman

# Para adaptar el repositorio a nuestro uso, debemos de adaptar dos comandos.
read_command = getattr(pacman, 'read_command', getattr(pacman, 'readCommand', None))
run_games = getattr(pacman, 'run_games', getattr(pacman, 'runGames', None))


# Clase del agente de markov de Pacman.
class MarkovAgent(Agent):

    # Hereda de Agent que viene en game.py de UC Berkeley.

    # Definimos la función de inicialización (instanción).
    def __init__(self, **kwargs):

        # Pasamos los argumentos requeridos.
        super().__init__(**kwargs)

        # Matriz de transición NxN.
        self.P = None

        # Espacio de Estados S. Las casillas son coordenadas x, y.
        self.pos_to_idx = {}  # Diccionario que traduce (x, y) a núm de estado i (0, ..., N-1).
        self.idx_to_pos = {}  # Diccionario que traduce núm de estado i a (x, y).

    # Función auxiliar que analiza las paredes del mapa y crea los diccionarios de estados y llena P.
    def build_transition_matrix(self, gameState):

        walls = gameState.get_walls()  # Obtenemos las posiciones de las paredes.

        height = walls.height
        width = walls.width

        # Encontramos los espacios vacios.

        idx = 0  # Índice auxiliar para pasillos.

        # Por cada coordenada en x.
        for x in range(width):

            # Por cada coordenada en y.
            for y in range(height):

                # Si no hay una pared en (x, y).
                if not walls[x][y]:

                    # Almacenamos las coordenadas y el índice.
                    self.pos_to_idx[(x, y)] = idx
                    self.idx_to_pos[idx] = (x, y)
                    idx += 1

        N = len(self.pos_to_idx)  # N va a ser la longitud del diccionario de posiciones vacías.


        # Inicializamos una matriz P de ceros de tamaño N x N.

        self.P = np.zeros((N, N), dtype=float)


        # Para cada casilla, calculamos sus vecinos libres y repartimos la probabilidad de 1/k.

        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # (1, 0): Este, (-1, 0): Oeste, (0, 1): Norte, (0, -1): Sur.

        # Iteramos por todas las casillas:

        for (x, y), i in self.pos_to_idx.items():

            neighbors = []  # Inicializamos vecinos en 0.

            # Iteramos por las 4 direcciones cardinales.
            for dirx, diry in directions:

                # Coordenadas de los vecinos al Norte, Sur, Este y Oeste.
                n_x, n_y = x + dirx, y + diry

                # Si existe en el diccionario de casillas libres, es un vecino válido (no pared).

                if (n_x, n_y) in self.pos_to_idx:

                    # Lo añadimos a la lista devecinos válidos para la coordenada.
                    neighbors.append(self.pos_to_idx[(n_x, n_y)])

            # Calculamos la cantidad de vecinos de la coordenada.

            k = len(neighbors)

            # Si existe al menos un vecino:
            if k > 0:

                # La probabilidad es 1/k.
                prob = 1/k

                # Añadimos la probabilidad a todos los vecinos.

                for j in neighbors:
                    self.P[i, j] = prob
            else:

                # No hay vecinos por lo que no existe probabilidad de transición (aislada en sí misma).
                self.P[i, i] = 1


        # Todos los renglones deben de sumar 1.
        rsum = self.P.sum(axis=1)
        assert np.allclose(rsum, 1), "La matriz P no cumple con ser estocástica"
        print(f"Espacio de estados S = {N}. Matriz P creada con forma {self.P.shape}.")

        self.P2 = np.linalg.matrix_power(self.P, 2)
        self.P3 = np.linalg.matrix_power(self.P, 3)


    def get_action(self, gameState):

        # Si aún no se ha generado la matriz de transiciones, la generamos.
        if self.P is None:
            self.build_transition_matrix(gameState)

        # Para decidir el movimiento del jugador (pacman), se deben de analizar los posibles movimientos de 
        # legal_actions (N, S, E, O o Stop). Descartamos Stop para evitar que el algoritmo se quede quieto.

        legal_actions = gameState.get_legal_actions(0)

        actions = [a for a in legal_actions if a != "Stop"]

        # Si no hubieran acciones posibles, dejamos que se pueda quedar quieto.
        if not actions:
            actions = legal_actions

        
        # Obtenemos las posiciones actuales de los fantasmas enemigos.
        # Si un fantasma está en (xf, yf) encontramos su estado usando i = self.pos_to_idx[(int(xf), int(yf))]-
        
        normal_ghost_index = []  # Declaramos una lista para almacenar los índices de los fantasmas.
        weak_ghost_index = []  # Declaramos una lista para almacenar los índices de los fantasmas vulnerables.

        # Por cada fantasma.
        for ghost_state in gameState.get_ghost_states():

            # Sus coordenadas (x, y).
            xf, yf = ghost_state.get_position()

            # Juntamos las coordenadas en un solo vector.
            pos = (int(xf), int(yf))

            # Si se encuentra dentro de las casillas libres.
            if pos in self.pos_to_idx:

                # El índice donde se debe de añadir.
                index = self.pos_to_idx[pos]

                # Si el fantasma es vulnerable (con mínimo tres turnos de tiempo restante):
                if ghost_state.scared_timer > 3:

                    # Lo añadimos al vector de fantasmas asustados.
                    weak_ghost_index.append(index)

                else:

                    # Lo añadimos al vector de fantasmas normales.
                    normal_ghost_index.append(index)


        
        # Utilizando la teoría de Chapman-Kolmogorov, sabemos que la probabilidad de que un fantasma pase de su casilla
        # actual i, a una libre j en m pasos es exactamente (i, j) de la m-ésima potencia de la matríz de transición.

        # Es decir: (P_ij)⌃(m) = (P⌃m)ij, donde si m = 1 es la matriz directa de P y el fantasma estará estará en el siguiente
        # turno, y si m = 2 es P⌃2 que es donde podría estar en dos turnos. Como debemos de utilizar muchas operaciones de potencia,
        # podemos calcularlas antes de la simulación del juego (es decir, cuando llenamos P). Hacemos lo mismo con m = 3.

        # Debemos asimismo establecer ciertos hiperparámetros para dirigir el comportamiento del agente correctamente.

        best_action = actions[0]  # Inicializamos la mejor acción como la primera dada.
        best_expected_val = -float("inf")  # Inicializamos el mejor valor esperado como -inf para que cualquiera mejor lo reemplace.

        # Matriz de pagos (se penaliza la muerte).

        death_penalty = -500  # Un valor lo suficientemente grande para forzar al modelo, pero que no aproxima a -inf.

        # Para cada acción legal a vamos a calcular su viabilidad.
        for a in actions:

            successor = gameState.generate_successor(0, a)  # Successor permite simular el movimiento de pacman.
            new_pos = successor.get_pacman_position()  # Posible nueva posición de pacman.
            new_idx = self.pos_to_idx[new_pos]  # Posible nueva posición en índice de pacman.

            # Calculamos la probabilidad de colisión dada por Chapman (riesgo aleatorio).
            prob_col_1 = sum(self.P[g_i, new_idx] for g_i in normal_ghost_index)
            prob_col_2 = sum(self.P2[g_i, new_idx] for g_i in normal_ghost_index)
            prob_col_3 = sum(self.P3[g_i, new_idx] for g_i in normal_ghost_index)

            # Calculamos la probabilidad de matar a un fantasma débil.
            prob_scared_1 = sum(self.P[g_i, new_idx] for g_i in weak_ghost_index)
            prob_scared_2 = sum(self.P2[g_i, new_idx] for g_i in weak_ghost_index)
            prob_scared_3 = sum(self.P3[g_i, new_idx] for g_i in weak_ghost_index)

            # Riesgo total en corto plazo.
            kill_reward = (prob_scared_1 * 200) + (0.5 * prob_scared_2 * 100) + (0.25 * prob_scared_3 * 50)
            expected_risk = (prob_col_1 * death_penalty) + (0.5 * prob_col_2 * death_penalty) + (0.25 * prob_col_3 * death_penalty) + kill_reward
            # Riesgo en 2 turnos tiene la mitad de peso, riesgo en 3 turnos un 25%. Igual para las recompensas de caza.

            # Recompensa inmediata.
            reward_score = successor.get_score() - gameState.get_score()  # Calculamos la diff entre el score anterior y el nuevo posible.

            # Para fomentar que el agente vaya hacia direcciones donde hay comida, podemos calcularla distancia Manhattan a la 
            # comida más cercana (no se puede usar euclidiana al ser un entorno cuadriculado).

            food_list = successor.get_food().as_list()  # Obtenemos una lista con toda la comida.

            # Usamos fórmula  Manhattan.
            minimum_dist = min((abs(new_pos[0] - fx) + abs(new_pos[1] - fy) for fx, fy in food_list), default=0)

            # Evaluamos si hay power-ups.
            powers = successor.get_capsules()
            obtained_powers = len(gameState.get_capsules()) - len(powers)  # Determinamos si se obtuvo un power-up.

            # Si se tomó un poder se le da una recompensa.
            reward_power = obtained_powers * 150

            # Si hay fantasmas normales y hay un poder cercano, incentivamos que use el poder.
            if powers and normal_ghost_index:

                # Distancia Manhattan mínima de los poderes.
                minimum_power_dist = min(abs(new_pos[0] - px) + abs(new_pos[1] - py) for px, py in powers)

                reward_power -= 0.8*minimum_power_dist  # Se la restamos a la recompensa para que busque minimizarla.

            reward = reward_score + reward_power - (0.5 * minimum_dist if food_list else 0)  
            # Le restamos la distancia a la comida para que fomente reducirla (si no hay comida no se le resta nada).
            
       
            # Valor esperado.
            ev = reward + expected_risk

            # Si el valor esperado es el mejor hasta el momento.
            if ev > best_expected_val:

                # Lo volvemos el mejor.
                best_expected_val = ev

                # Y guardamos la acción.
                best_action = a

        # Devolvemos la mejor acción encontrada en el espacio de búsqueda.
        return best_action

            


if __name__ == '__main__':

    # Debemos de registrar el agente sin tocar multiagent.
    import multi_agents
    multi_agents.MarkovAgent = MarkovAgent


    # Leemos los comandos y argumentos de la terminal.
    args = read_command(sys.argv[1:])

    # Asignamos el agente a la partida

    args['pacman'] = MarkovAgent()

    # Arrancamos el simulador gráfico
    run_games(**args)