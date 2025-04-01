from collections import deque
import heapq
import time
import copy
import pygame
from ui_module import gui

class Solver:
    def __init__(self, game):
        self.game = game
        self.best_moves = []
        self.number_moves = float('+inf')
        iteration = 0

    # cria um hash com informaçao do state da board(board, hand, goal, queue) para depois comparar se o state ja foi visitado ou nao
    def hash_game_state(self, game):
        board_str = str(game.board.grid)  
        hand_str = str([str(p) for p in game.hand.pieces])
        goal_str = str(game.goal.goal)
        # queue_str = str([str(p) for p in game.queue.pieces]) 
        return hash(board_str + hand_str + goal_str )
    
    def dfs(self, game_state, move_sequence, depth):
        if depth == 0 or game_state.is_goal_met():
            if len(move_sequence)<self.number_moves:
                self.number_moves = len(move_sequence)
                self.best_moves = move_sequence[:]
                # if game_state.is_goal_met() == True:
                #     print("Goal met")
                #     print(self.best_moves)
            return
        
        for row in range(game_state.board.rows):
            for col in range(game_state.board.cols):
                for hand_index in range(len(game_state.hand.pieces)):
                    new_game_state = copy.deepcopy(game_state)
                    piece = new_game_state.hand.get_piece(hand_index)
                    if piece:
                        try:
                            new_game_state.board.place_piece(row, col, piece)
                            new_game_state.board.pop_clusters()
                            new_game_state.refill_hand()
                            move_sequence.append((row, col, hand_index))
                            self.iteration += 1
                            self.dfs(new_game_state, move_sequence, depth - 1)
                            move_sequence.pop()
                        except (IndexError, ValueError):
                            continue
    
    def find_best_moves(self, max_depth=23):
        self.iteration = 0
        self.dfs(self.game, [], max_depth)
        print("iteration:", self.iteration)
        return self.best_moves
    
    def bfs(self, game):
        visited = set()
        queu = deque()
        queu.append((game, []))
        while queu:
            game, move_sequence = queu.popleft()
            # if(len(move_sequence) == 1):
            #     print(move_sequence, game.is_goal_met())
            #     # print(game)

            current_hash = self.hash_game_state(game)
            if current_hash in visited:
                continue
            visited.add(current_hash)

            if(game.is_goal_met()):
                self.best_moves = move_sequence
                return
            
            for move in self.possible_moves(game):  
                new_game = copy.deepcopy(game)
                piece = new_game.hand.get_piece(move[2])
                new_game.board.place_piece(move[0],move[1], piece)
                new_game.board.pop_clusters()
                new_game.refill_hand()
                
                state_hash = self.hash_game_state(new_game)
                self.iteration += 1
                if state_hash not in visited:
                    new_sequence = move_sequence + [move] 
                    queu.append((new_game, new_sequence))

    def possible_moves(self, game):
        possible_moves = []
        for row in range(game.board.rows):
            for cols in range(game.board.cols):
                for hand in range(len(game.hand.pieces)):
                    new_game = copy.deepcopy(game)
                    piece = new_game.hand.get_piece(hand)
                    if piece:
                        try:
                            new_game.board.place_piece(row, cols, piece)
                            possible_moves.append((row,cols,hand))
                        except (IndexError, ValueError):
                            continue 
        return possible_moves
    
    def find_best_moves_bfs(self):
        self.iteration = 0
        self.bfs(self.game)
        print("iteration:", self.iteration)
        return self.best_moves
    
    def a_star(self):
        visited = {} 
        priority_queue = []
        entry_count = 0

        initial_state = copy.deepcopy(self.game)
        initial_g = 0
        initial_h = self.heuristic2(initial_state)
        heapq.heappush(priority_queue, (initial_g + initial_h, entry_count, initial_state, []))
        entry_count += 1

        while priority_queue:
            f, _, current_game, move_sequence = heapq.heappop(priority_queue)
            current_hash = self.hash_game_state(current_game)

            # print(move_sequence, game.is_goal_met())
            # print(game)

            if current_hash in visited and visited[current_hash] <= f:
                continue
            visited[current_hash] = f

            if current_game.is_goal_met():
                self.best_moves = move_sequence
                print("iteration:", entry_count)
                return self.best_moves

            for move in self.possible_moves(current_game):
                new_game = copy.deepcopy(current_game)
                piece = new_game.hand.get_piece(move[2])
                try:
                    new_game.board.place_piece(move[0], move[1], piece) 
                    new_game.board.pop_clusters()
                    new_game.refill_hand()
                except (IndexError, ValueError):
                    continue

                new_g = len(move_sequence) + 1  
                new_h = self.heuristic2(new_game)
                new_f = new_g + new_h
                new_hash = self.hash_game_state(new_game)
                
                if new_hash not in visited or new_f < visited.get(new_hash, float('+inf')):
                    heapq.heappush(priority_queue, (new_f, entry_count, new_game, move_sequence + [move]))  
                    entry_count += 1

    def greedy(self):
        visited = set()
        priority_queue = []
        entry_count = 0
        initial_state = copy.deepcopy(self.game)
        initial_h = self.heuristic1(initial_state)
        heapq.heappush(priority_queue, (initial_h, entry_count, initial_state, []))
        entry_count += 1

        while priority_queue:
            h, _, current_game, move_sequence = heapq.heappop(priority_queue)
            current_hash = self.hash_game_state(current_game)

            if current_hash in visited:
                continue
            visited.add(current_hash)

            if current_game.is_goal_met():
                self.best_moves = move_sequence
                print("iterations:", entry_count)
                return self.best_moves

            for move in self.possible_moves(current_game):
                new_game = copy.deepcopy(current_game)
                piece = new_game.hand.get_piece(move[2])
                try:
                    new_game.board.place_piece(move[0], move[1], piece)
                    new_game.board.pop_clusters()
                    new_game.refill_hand()
                except (IndexError, ValueError):
                    continue

                new_h = self.heuristic1(new_game)
                new_hash = self.hash_game_state(new_game)

                if new_hash not in visited:
                    heapq.heappush(priority_queue, (new_h, entry_count, new_game, move_sequence + [move]))
                    entry_count += 1

    # Heuristica simples calcula soma total do objetivo
    def heuristic1(self, game):
        return game.goal.goal_sum() 
        
    # Heuristica assume que qualquer move elimina o maximo de cores possiveis
    def heuristic2(self, game):
        total_remaining = game.goal.goal_sum()
        return (total_remaining + 3) // 4  




class Color:
    BLUE = "B"
    GREEN = "G"
    RED = "R"
    YELLOW = "Y"
    NULL = "N" #estado de transição, entre um bloco ser rebentado e preenchido




class Piece:
    def __init__(self, c1, c2, c3, c4):
        self.colors = [c1, c2, c3, c4]

    def get_colors(self):
        return self.colors

    # Remove todas as ocorrências de uma cor na peça e preenche os espaços vazios
    # Retorna o número de ocorrências da cor removida
    def pop_color(self, color):
        count = self.colors.count(color)
        self.colors = [Color.NULL if c == color else c for c in self.colors]
        self.fill_nulls()
        return count
    
    
    def pop_connected(self, pos, color):        
        popped_indices = set()
    
        adjacency = {
            0: [1, 2],  # Top-Left → Top-Right, Bottom-Left
            1: [0, 3],  # Top-Right → Top-Left, Bottom-Right
            2: [0, 3],  # Bottom-Left → Top-Left, Bottom-Right
            3: [1, 2]   # Bottom-Right → Top-Right, Bottom-Left
        }
        
        for start_index in range(4):
            if self.colors[start_index] != color or start_index in popped_indices:
                continue
            
            stack = [start_index]
            cluster = []
            
            while stack:
                current = stack.pop()
                if current in popped_indices or self.colors[current] != color:
                    continue
                
                popped_indices.add(current)
                cluster.append(current)
                
            
                for neighbor in adjacency[current]:
                    if neighbor not in popped_indices and self.colors[neighbor] == color:
                        stack.append(neighbor)
            
        
            # print("pop_color cluster", cluster)
            # print("pop_color neighbor", neighbor)
            if len(cluster) >= 2:
                for idx in cluster: 
                    self.colors[idx] = Color.NULL
            if len(cluster) == 1:
                self.colors[pos] = Color.NULL
        
        self.fill_nulls()  # Your existing fill method
        return len(popped_indices)
        

        

    # Preenche os espaços vazios com cores baseadas nas cores dos vizinhos
    def fill_nulls(self):
        for i in range(4):
            if self.colors[i] == Color.NULL:
                neighbors = self.get_neighbors(i)
                if neighbors:
                    self.colors[i] = self.choose_fill_color(neighbors, i)

    # Retorna as cores dos vizinhos de uma dada posição dentro da peça
    def get_neighbors(self, index):
        if index == 0:
            return [self.colors[1], self.colors[2]]
        elif index == 1:
            return [self.colors[0], self.colors[3]]
        elif index == 2:
            return [self.colors[0], self.colors[3]]
        elif index == 3:
            return [self.colors[1], self.colors[2]]

    # Escolhe a cor menos popular entre os vizinhos para preencher um espaço vazio
    # Se houver um empate, prioriza o vizinho horizontal
    def choose_fill_color(self, neighbors, index):
        color_counts = {Color.BLUE: 0, Color.GREEN: 0, Color.RED: 0, Color.YELLOW: 0}
        for color in self.colors:
            if color != Color.NULL:
                color_counts[color] += 1

        # Filtra Color.NULL dos vizinhos
        valid_neighbors = [color for color in neighbors if color != Color.NULL]
        neighbor_counts = {color: color_counts[color] for color in valid_neighbors}
        if not neighbor_counts:
            return Color.NULL  # se todos os vizinhos forem NULL, retorn NULL

        least_popular_color = min(neighbor_counts, key=neighbor_counts.get)

        if len(valid_neighbors) == 2 and neighbor_counts[valid_neighbors[0]] == neighbor_counts[valid_neighbors[1]]:
            # Se houver um empate, prioriza o vizinho horizontal
            if index in [0, 1]:  # Horizontal neighbors for index 0 and 2 are 1 and 3 respectively
                return neighbors[0]
            else:  # Horizontal neighbors for index 1 and 3 are 0 and 2 respectively
                return neighbors[1]
        else:
            return least_popular_color

    # Retorna True se todas as cores da peça forem Color.NULL
    def is_fully_null(self):
        return all(color == Color.NULL for color in self.colors)

    # Retorna uma representação da peça como uma string
    def __str__(self):
        return f"{self.colors[0]} {self.colors[1]} | {self.colors[2]} {self.colors[3]}"





class Board:
    def __init__(self, rows, cols, goal):
        self.rows = rows
        self.cols = cols
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]
        self.goal = goal

    # Coloca uma peça na posição (row, col) do tabuleiro
    # Lança uma exceção IndexError se a posição estiver fora dos limites
    def place_piece(self, row, col, piece):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            if self.grid[row][col] is None:
                # new_piece = copy.deepcopy(piece)
                # self.grid[row][col] = new_piece
                self.grid[row][col] = piece
                
            else:
                raise ValueError("Position already occupied")
        else:
            raise IndexError("Position out of bounds")

    # Retorna a peça na posição (row, col) do tabuleiro
    def get_piece(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        else:
            raise IndexError("Position out of bounds")

    # Remove todas as ocorrências de uma cor em uma peça na posição (row, col)
    def pop_color_at(self, row, col, color):
        piece = self.get_piece(row, col)
        if piece:
            count = piece.pop_color(color)
            if piece.is_fully_null():
                self.grid[row][col] = None
            return count
        return 0
    
    def pop_color_at2(self, row, col, pos, color):
        piece = self.get_piece(row, col)
        if piece:
            count = piece.pop_connected(pos, color)
            if piece.is_fully_null():
                self.grid[row][col] = None
            return count
        return 0

    # Encontra clusters com a mesma cor e remove-as
    # Cluster: conjunto de partes de peças adjacentes com a mesma cor
    def find_clusters(self):
        clusters = []
        visited = set()
        
        # # DFS para encontrar clusters
        # def dfs(r, c, color, direction):
        #     stack = [(r, c, direction)]
        #     cluster = []
        #     while stack:
        #         row, col, direction = stack.pop()
        #         if (row, col, direction) not in visited and 0 <= row < self.rows and 0 <= col < self.cols:
        #             piece = self.get_piece(row, col)
        #             if piece and color in piece.get_colors():
        #                 visited.add((row, col, direction))
        #                 cluster.append((row, col))
        #                 # Check adjacent cells based on direction
        #                 if direction == "top-left":
        #                     if row > 0 and piece.colors[0] == color:  # Top
        #                         top_piece = self.get_piece(row - 1, col)
        #                         if top_piece and top_piece.colors[2] == color:
        #                             stack.append((row - 1, col, "bottom-left"))
        #                     if col > 0 and piece.colors[0] == color:  # Left
        #                         left_piece = self.get_piece(row, col - 1)
        #                         if left_piece and left_piece.colors[1] == color:
        #                             stack.append((row, col - 1, "top-right"))
        #                 if direction == "top-right":
        #                     if row > 0 and piece.colors[1] == color:  # Top
        #                         top_piece = self.get_piece(row - 1, col)
        #                         if top_piece and top_piece.colors[3] == color:
        #                             stack.append((row - 1, col, "bottom-right"))
        #                     if col < self.cols - 1 and piece.colors[1] == color:  # Right
        #                         right_piece = self.get_piece(row, col + 1)
        #                         if right_piece and right_piece.colors[0] == color:
        #                             stack.append((row, col + 1, "top-left"))
        #                 if direction == "bottom-left":
        #                     if row < self.rows - 1 and piece.colors[2] == color:  # Bottom
        #                         bottom_piece = self.get_piece(row + 1, col)
        #                         if bottom_piece and bottom_piece.colors[0] == color:
        #                             stack.append((row + 1, col, "top-left"))
        #                     if col > 0 and piece.colors[2] == color:  # Left
        #                         left_piece = self.get_piece(row, col - 1)
        #                         if left_piece and left_piece.colors[3] == color:
        #                             stack.append((row, col - 1, "bottom-right"))
        #                 if direction == "bottom-right":
        #                     if row < self.rows - 1 and piece.colors[3] == color:  # Bottom
        #                         bottom_piece = self.get_piece(row + 1, col)
        #                         if bottom_piece and bottom_piece.colors[1] == color:
        #                             stack.append((row + 1, col, "top-right"))
        #                     if col < self.cols - 1 and piece.colors[3] == color:  # Right
        #                         right_piece = self.get_piece(row, col + 1)
        #                         if right_piece and right_piece.colors[2] == color:
        #                             stack.append((row, col + 1, "bottom-left"))
        #     return cluster

        # for row in range(self.rows):
        #     for col in range(self.cols):
        #         piece = self.get_piece(row, col)
        #         if piece:
        #             for i, color in enumerate(piece.get_colors()):
        #                 if color != Color.NULL:
        #                     direction = ["top-left", "top-right", "bottom-left", "bottom-right"][i]
        #                     cluster = dfs(row, col, color, direction)
        #                     if len(cluster) > 1:
        #                         clusters.append((color, cluster))
        # return clusters

        def dfs(row, col, quadrant, color):
            stack = [(row, col, quadrant)]
            cluster = []
            while stack:
                r, c, q = stack.pop()
                if (r, c, q) in visited:
                    continue
                piece = self.get_piece(r, c)
                if not piece or piece.colors[q] != color:
                    continue
                visited.add((r, c, q))
                cluster.append((r, c, q))
                # Check adjacent quadrants in neighboring pieces
                if q == 0:  # Top-left quadrant
                    # Check piece above (its bottom-left quadrant, q=2)
                    if r > 0:
                        stack.append((r - 1, c, 2))
                    # Check left piece (its top-right quadrant, q=1)
                    if c > 0:
                        stack.append((r, c - 1, 1))
                elif q == 1:  # Top-right quadrant
                    # Check piece above (its bottom-right quadrant, q=3)
                    if r > 0:
                        stack.append((r - 1, c, 3))
                    # Check right piece (its top-left quadrant, q=0)
                    if c < self.cols - 1:
                        stack.append((r, c + 1, 0))
                elif q == 2:  # Bottom-left quadrant
                    # Check piece below (its top-left quadrant, q=0)
                    if r < self.rows - 1:
                        stack.append((r + 1, c, 0))
                    # Check left piece (its bottom-right quadrant, q=3)
                    if c > 0:
                        stack.append((r, c - 1, 3))
                elif q == 3:  # Bottom-right quadrant
                    # Check piece below (its top-right quadrant, q=1)
                    if r < self.rows - 1:
                        stack.append((r + 1, c, 1))
                    # Check right piece (its bottom-left quadrant, q=2)
                    if c < self.cols - 1:
                        stack.append((r, c + 1, 2))
            return cluster

        for row in range(self.rows):
            for col in range(self.cols):
                piece = self.get_piece(row, col)
                if piece:
                    for q in range(4):
                        color = piece.colors[q]
                        if color != Color.NULL and (row, col, q) not in visited:
                            cluster = dfs(row, col, q, color)
                            if len(cluster) >= 2:  # Minimum cluster size
                                clusters.append((color, cluster))
        return clusters
        

    # Remove clusters e atualiza o goal
    def pop_clusters(self):
        # while True:
        #     clusters = self.find_clusters()
        #     print("clusts:" ,clusters)
        #     if not clusters:
        #         break
        #     for color, cluster in clusters:
        #         total_popped = 0
        #         for row, col in cluster:
        #             total_popped += self.pop_color_at(row, col, color)
        #         self.goal.pop_color(color, total_popped)
        while True:
            clusters = self.find_clusters()
            # print("clusters:" ,clusters)
            if not clusters:
                break
            for color, cluster in clusters:
                # print("cluster:", color, cluster)
                total_popped = 0
                for (row, col, q) in cluster:
                    total_popped += self.pop_color_at2(row, col, q, color)
                self.goal.pop_color(color, total_popped)


    def pop_clusters_ui(self, ui):
        while True:
          
            clusters = self.find_clusters()
            if not clusters:
                break
            for color, cluster in clusters:
                total_popped = 0
                for (row, col, q) in cluster:
                    total_popped += self.pop_color_at2(row, col, q, color)
                self.goal.pop_color(color, total_popped) 
            ui.make_board()
            pygame.display.flip()
            pygame.time.delay(1000)
            

                           


    # Retorna uma representação do tabuleiro como uma string
    def __str__(self):
        board_str = ""
        separator_line = "-" * (self.cols * 6 + 1)
        
        for row_index, row in enumerate(self.grid):
            top_row = ""
            bottom_row = ""
            for col_index, piece in enumerate(row):
                if piece:
                    piece_str = str(piece).split(' | ')
                    top_row += piece_str[0] + " | "
                    bottom_row += piece_str[1] + " | "
                else:
                    top_row += "    | "
                    bottom_row += "    | "
            board_str += f"{separator_line}\n| {top_row[:-3]} |\n| {bottom_row[:-3]} |\n"
        
        board_str += separator_line
        return board_str.strip()



# Objetivo do jogo; contém a quantidade de cada cor que deve ser removida
class Goal:
    def __init__(self, blue, green, red, yellow):
        self.goal = {
            Color.BLUE: blue,
            Color.GREEN: green,
            Color.RED: red,
            Color.YELLOW: yellow
        }

    # Remove uma quantidade de uma cor do objetivo
    def pop_color(self, color, count):
        if color in self.goal and self.goal[color] > 0:
            self.goal[color] = max(0, self.goal[color] - count)

    # Retorna True se o objetivo for atingido
    def is_goal_met(self):
        return all(count == 0 for count in self.goal.values())

    # Retorna numero total do objetivo
    def goal_sum(self):
        return sum(self.goal.values())

    # Retorna uma representação do objetivo como uma string
    def __str__(self):
        return f"Goal: {self.goal}"



# Contém as peças que o jogador pode usar
class Hand:
    # max_pieces: número máximo de peças que a mão pode conter
    def __init__(self, max_pieces):
        self.pieces = []
        self.max_pieces = max_pieces

    # Adiciona uma peça à mão
    def add_piece(self, piece):
        if len(self.pieces) < self.max_pieces:
            self.pieces.append(piece)
        else:
            raise ValueError("Hand is full")

    # Remove uma peça da mão
    def get_piece(self, index):
        if 0 <= index < len(self.pieces):
            return self.pieces.pop(index)
        else:
            return None

    # Retorna uma representação da mão como uma string
    def __str__(self):
        hand_str = "Hand:\n"
        top_row = ""
        bottom_row = ""
        for piece in self.pieces:
            piece_str = str(piece).split(' | ')
            top_row += piece_str[0] + "    "
            bottom_row += piece_str[1] + "    "
        hand_str += f"{top_row.strip()}\n{bottom_row.strip()}"
        return hand_str.strip()



# Peças seguintes que irão entrar na mão do jogador
class Queue:
    def __init__(self, pieces):
        self.pieces = deque(pieces)

    # Remove e retorna a próxima peça da fila
    def draw_piece(self):
        if self.pieces:
            self.piece = self.pieces.popleft()
            # self.pieces.append(self.piece)
            piece_copy = copy.deepcopy(self.piece)
            self.pieces.append(piece_copy)
            return self.piece
        else:
            return None

    # Retorna uma representação da fila como uma string
    def __str__(self):
        queue_str = "Queue:\n"
        top_row = ""
        bottom_row = ""
        for piece in self.pieces:
            piece_str = str(piece).split(' | ')
            top_row += piece_str[0] + "    "
            bottom_row += piece_str[1] + "    "
        if not top_row:
            top_row = "Empty"
        queue_str += f"{top_row.strip()}\n{bottom_row.strip()}"
        return queue_str.strip()



# Contém o estado do jogo: tabuleiro, objetivo, mão e fila de peças
class Game:
    def __init__(self, rows, cols, goal, hand, queue):
        self.board = Board(rows, cols, goal)
        self.goal = goal
        self.hand = hand
        self.queue = queue
        self.refill_hand()

    def make_board(self):
        self.board.place_piece(0,0,self.queue.draw_piece())
        self.board.place_piece(0,2,self.queue.draw_piece())
        self.board.place_piece(1,1,self.queue.draw_piece())
        self.board.place_piece(2,0,self.queue.draw_piece())
        self.board.place_piece(2,2,self.queue.draw_piece())
        self.board.place_piece(2,1,self.queue.draw_piece())

        

    # Preenche a mão com peças da fila
    def refill_hand(self):
        while len(self.hand.pieces) < self.hand.max_pieces:
            piece = self.queue.draw_piece()
            if piece:
                self.hand.add_piece(piece)
            else:
                break

    # Coloca uma peça na posição (row, col) do tabuleiro
    def place_piece(self, row, col, hand_index):
        piece = self.hand.get_piece(hand_index)
        if piece:
            self.board.place_piece(row, col, piece)
            self.board.pop_clusters()
            self.refill_hand()
        else:
            print("Invalid piece index or no more pieces in hand.")

    # verifica se o objetivo foi atingido
    def is_goal_met(self):
        return self.goal.is_goal_met()
    
    # Retorna uma representação do jogo como uma string
    def __str__(self):
        return f"{self.board}\n\n{self.goal}\n\n{self.hand}\n\n{self.queue}"
    






# LEVELS
def level_1():
    # Create pieces with a single color
    pieces = [
        Piece(Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE),
        Piece(Color.RED, Color.RED, Color.RED, Color.RED),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW),
        Piece(Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE),
        Piece(Color.RED, Color.RED, Color.RED, Color.RED),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW),
        Piece(Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE),
        Piece(Color.RED, Color.RED, Color.RED, Color.RED),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW),
        Piece(Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE),
        Piece(Color.RED, Color.RED, Color.RED, Color.RED),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW),
    ]

    goal = Goal(blue=16, green=16, red=16, yellow=16)

    hand = Hand(max_pieces=2)

    queue = Queue(pieces)

    game = Game(4, 4, goal, hand, queue)

    game.place_piece(0, 2, 0)  # Place first piece at (0, 0)
    game.place_piece(2, 1, 1)  # Place second piece at (0, 1)
   

    return game


def level_2():
    pieces = [
        Piece(Color.YELLOW, Color.YELLOW, Color.GREEN, Color.GREEN),
        Piece(Color.GREEN, Color.GREEN, Color.RED, Color.RED),
        Piece(Color.YELLOW, Color.YELLOW, Color.BLUE, Color.BLUE),
        Piece(Color.RED, Color.RED, Color.BLUE, Color.BLUE),
        Piece(Color.GREEN, Color.YELLOW, Color.GREEN, Color.YELLOW),
        Piece(Color.RED, Color.RED, Color.GREEN, Color.GREEN),
        Piece(Color.BLUE, Color.RED, Color.BLUE, Color.RED),
        Piece(Color.BLUE, Color.BLUE, Color.RED, Color.RED),
        Piece(Color.GREEN, Color.GREEN, Color.YELLOW, Color.YELLOW),
        Piece(Color.RED, Color.BLUE, Color.RED, Color.BLUE),
        Piece(Color.BLUE, Color.RED, Color.BLUE, Color.RED),
        Piece(Color.YELLOW, Color.GREEN, Color.YELLOW, Color.GREEN),
        Piece(Color.GREEN, Color.YELLOW, Color.GREEN, Color.YELLOW),
        Piece(Color.RED, Color.RED, Color.BLUE, Color.BLUE),
        Piece(Color.YELLOW, Color.YELLOW, Color.GREEN, Color.GREEN),
        Piece(Color.BLUE, Color.BLUE, Color.YELLOW, Color.YELLOW),
        Piece(Color.RED, Color.RED, Color.GREEN, Color.GREEN),
        Piece(Color.GREEN, Color.GREEN, Color.YELLOW, Color.YELLOW),
        Piece(Color.YELLOW, Color.YELLOW, Color.BLUE, Color.BLUE),
        Piece(Color.BLUE, Color.RED, Color.BLUE, Color.RED)  
    ]

    goal = Goal(blue=20, green=20, red=20, yellow=20)

    hand = Hand(max_pieces=3)

    queue = Queue(pieces)

    game = Game(5, 5, goal, hand, queue)

    game.place_piece(0, 0, 0)  # Place first piece at (0, 0)
    game.place_piece(1, 2, 1)  # Place second piece at (0, 1)
    game.place_piece(0, 4, 2)  # Place third piece at (0, 2)
    game.place_piece(3, 3, 0)  # Place fourth piece at (0, 3)
    game.place_piece(4, 1, 1)  # Place fifth piece at (0, 4)
    


    return game


def level_3():
    pieces = [
        Piece(Color.YELLOW, Color.GREEN, Color.YELLOW, Color.GREEN),
        Piece(Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE),
        Piece(Color.GREEN, Color.BLUE, Color.YELLOW, Color.RED),
        Piece(Color.BLUE, Color.RED, Color.BLUE, Color.RED),
        Piece(Color.RED, Color.YELLOW, Color.BLUE, Color.GREEN),
        Piece(Color.RED, Color.RED, Color.RED, Color.RED),
        Piece(Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW),
        Piece(Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW),
        Piece(Color.GREEN, Color.YELLOW, Color.RED, Color.BLUE),
        Piece(Color.RED, Color.RED, Color.GREEN, Color.GREEN),
        Piece(Color.GREEN, Color.GREEN, Color.YELLOW, Color.YELLOW),
        Piece(Color.YELLOW, Color.GREEN, Color.RED, Color.BLUE),
        Piece(Color.BLUE, Color.YELLOW, Color.GREEN, Color.RED),
        Piece(Color.RED, Color.GREEN, Color.YELLOW, Color.BLUE),
        Piece(Color.RED, Color.BLUE, Color.RED, Color.BLUE),
        Piece(Color.YELLOW, Color.YELLOW, Color.GREEN, Color.GREEN),
        Piece(Color.YELLOW, Color.YELLOW, Color.BLUE, Color.BLUE),
        Piece(Color.BLUE, Color.GREEN, Color.RED, Color.YELLOW),
        Piece(Color.BLUE, Color.RED, Color.BLUE, Color.RED),
        Piece(Color.GREEN, Color.RED, Color.BLUE, Color.YELLOW),
        Piece(Color.BLUE, Color.BLUE, Color.RED, Color.RED),
        Piece(Color.RED, Color.RED, Color.BLUE, Color.BLUE),
        Piece(Color.YELLOW, Color.RED, Color.GREEN, Color.BLUE),
        Piece(Color.GREEN, Color.GREEN, Color.RED, Color.RED),
        Piece(Color.RED, Color.BLUE, Color.YELLOW, Color.GREEN),
        Piece(Color.GREEN, Color.YELLOW, Color.GREEN, Color.YELLOW),
        Piece(Color.GREEN, Color.YELLOW, Color.BLUE, Color.RED),
        Piece(Color.RED, Color.RED, Color.BLUE, Color.BLUE)
    ]


    goal = Goal(blue=30, green=30, red=30, yellow=30)

    hand = Hand(max_pieces=2)

    queue = Queue(pieces)

    game = Game(6, 6, goal, hand, queue)

    piece_index = 0
    for row in range(6):
        for col in range(6):
            if (row + col) % 2 == 0:  # Chess-like pattern: alternate placement
                game.place_piece(row, col, piece_index % len(hand.pieces))
                piece_index += 1



    return game

def level_teste():
    # Create pieces with a single color
    piecesboard = [
        Piece(Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE),
        Piece(Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW),
        Piece(Color.RED, Color.RED, Color.RED, Color.RED),
    ]

    pieces = [
        Piece(Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.RED, Color.RED, Color.RED, Color.RED),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE),
        Piece(Color.RED, Color.RED, Color.RED, Color.RED),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW),
        Piece(Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE),
        Piece(Color.RED, Color.RED, Color.RED, Color.RED),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW),
        Piece(Color.BLUE, Color.BLUE, Color.BLUE, Color.BLUE),
        Piece(Color.RED, Color.RED, Color.RED, Color.RED),
        Piece(Color.GREEN, Color.GREEN, Color.GREEN, Color.GREEN),
        Piece(Color.YELLOW, Color.YELLOW, Color.YELLOW, Color.YELLOW),
    ]

    goal = Goal(blue=12, green=12, red=0, yellow=0)

    hand = Hand(max_pieces=2)

    queue = Queue(pieces)

    game = Game(3, 3, goal, hand, queue)

    game.board.place_piece(2, 0, piecesboard[0])
    game.board.place_piece(2, 2, piecesboard[1])
    game.board.place_piece(2, 1, piecesboard[2])
    game.board.place_piece(0, 2, piecesboard[3])
    game.board.place_piece(0, 0, piecesboard[4])
    game.board.place_piece(0, 1, piecesboard[5])
    
    

    return game



print("Choose your level")
print("level 1 4x4")
print("level 2 5x5")
print("level 3 6x6")
print("level teste 4")
n_game = input()
# n_game = '1'

if n_game == '1':
    game = level_1()
elif n_game == '2':
    game = level_2()
elif n_game == '3':
    game = level_3()    
elif n_game == '4':
    game = level_teste()

print(game)

# UI---

pygame.init()
pygame.display.set_caption("Jelly Field")
WIDTH, HEIGHT = 1000, 900  # Screen size
screen = pygame.display.set_mode((WIDTH, HEIGHT))

BLOCKSIZE = (screen.get_width() - 500) / game.board.cols
LILBLOCK = BLOCKSIZE / 2
gamesize = BLOCKSIZE * game.board.cols
selected = 0

ui = gui(game, screen,BLOCKSIZE, gamesize)
screen.fill("purple")
ui.make_board()
ui.draw_hand(selected)
ui.draw_goal()
print(game)
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse = pygame.mouse.get_pos()
            print(mouse)
            if((250<=mouse[0]<=250+gamesize) and (100<= mouse[1]<= (100+ (BLOCKSIZE*game.board.rows)))):
                print("(" , (mouse[0]-250) // BLOCKSIZE , "," , (mouse[1]-100) // BLOCKSIZE , ")")
                x = (mouse[0]-250) // BLOCKSIZE
                y = (mouse[1]-100) // BLOCKSIZE
                if(selected != 0):
                    try: 
                        print("teste")
                        piece = copy.deepcopy(game.hand.pieces[selected-1])
                        game_teste = copy.deepcopy(game)
                        game_teste.board.place_piece(int(y), int(x), piece)
                    except (IndexError, ValueError):
                        print("ola")
                    else:
                        game.board.place_piece(int(y), int(x), game.hand.get_piece(selected-1))
                        ui.make_board()
                        game.refill_hand()
                        ui.draw_hand(0)
                        pygame.display.flip()
                        pygame.time.delay(1000)
                        game.board.pop_clusters_ui(ui)
                        screen.fill("purple")
                        ui.make_board()
                    finally:
                        selected = 0
                        ui.draw_hand(selected)
                        ui.draw_goal()
            if(game.hand.max_pieces == 1):
                if((WIDTH/2)-LILBLOCK-20<=mouse[0]<=(WIDTH/2)+LILBLOCK+20 and 640 <= mouse[1]<= 825):
                    selected = 1
                    ui.draw_hand(selected)
                else:
                    selected = 0
                    ui.draw_hand(selected)
            if(game.hand.max_pieces == 2):
                if((WIDTH/2)-25-BLOCKSIZE-20 <=mouse[0] <= (WIDTH/2)-25+20 and 640 <= mouse[1]<= 825):
                    print("selected= 1")
                    selected = 1
                    ui.draw_hand(selected)
                elif((WIDTH/2)+25-20 <= mouse[0]<= (WIDTH/2)+25+BLOCKSIZE+20 and 640 <= mouse[1]<= 825):
                    print("selected = 2")
                    selected= 2
                    ui.draw_hand(selected)
                else:
                    selected = 0
                    ui.draw_hand(selected)


    
    pygame.display.flip()

pygame.quit()

print(game)

# Usage example:
solver = Solver(game)
best_moves = []

while True:
    print("What algorithm do you want to test?")
    print("1.DFS")
    print("2.BFS")
    print("3.A*")
    print("4.greedy")
    n_algorithm = input()
    print("\n")

    if n_algorithm == '1':
        start = time.time()
        best_moves = solver.find_best_moves()
        end = time.time()
    elif n_algorithm == '2':
        start = time.time()
        best_moves = solver.find_best_moves_bfs()
        end = time.time()
    elif n_algorithm == '3':
        start = time.time()
        best_moves = solver.a_star()
        end = time.time()
    elif n_algorithm == '4':
        start = time.time()
        best_moves = solver.greedy()
        end = time.time()

    print("Best Move Sequence:", best_moves)

    if n_algorithm == '1':
        print("Time DFS:",end - start)
        # print("Memory used:", mend.statistics())
    elif n_algorithm == '2':
        print("Time BFS:",end - start)
        # print("Memory used:", mend.statistics())
    elif n_algorithm == '3':
        print("Time A*:", end - start)
        # print("Memory used:", mend.statistics('lineno'))
    elif n_algorithm == '4':
        print("Time Greedy:", end - start)
        # print("Memory used:", mend.statistics())

    print("\n")
