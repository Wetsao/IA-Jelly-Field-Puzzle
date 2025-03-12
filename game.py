from collections import deque




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
            if index in [0, 2]:  # Horizontal neighbors for index 0 and 2 are 1 and 3 respectively
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

    # Encontra clusters com a mesma cor e remove-as
    # Cluster: conjunto de partes de peças adjacentes com a mesma cor
    def find_clusters(self):
        clusters = []
        visited = set()

        # DFS para encontrar clusters
        def dfs(r, c, color, direction):
            stack = [(r, c, direction)]
            cluster = []
            while stack:
                row, col, direction = stack.pop()
                if (row, col, direction) not in visited and 0 <= row < self.rows and 0 <= col < self.cols:
                    piece = self.get_piece(row, col)
                    if piece and color in piece.get_colors():
                        visited.add((row, col, direction))
                        cluster.append((row, col))
                        # Check adjacent cells based on direction
                        if direction == "top-left":
                            if row > 0 and piece.colors[0] == color:  # Top
                                top_piece = self.get_piece(row - 1, col)
                                if top_piece and top_piece.colors[2] == color:
                                    stack.append((row - 1, col, "bottom-left"))
                            if col > 0 and piece.colors[0] == color:  # Left
                                left_piece = self.get_piece(row, col - 1)
                                if left_piece and left_piece.colors[1] == color:
                                    stack.append((row, col - 1, "top-right"))
                        if direction == "top-right":
                            if row > 0 and piece.colors[1] == color:  # Top
                                top_piece = self.get_piece(row - 1, col)
                                if top_piece and top_piece.colors[3] == color:
                                    stack.append((row - 1, col, "bottom-right"))
                            if col < self.cols - 1 and piece.colors[1] == color:  # Right
                                right_piece = self.get_piece(row, col + 1)
                                if right_piece and right_piece.colors[0] == color:
                                    stack.append((row, col + 1, "top-left"))
                        if direction == "bottom-left":
                            if row < self.rows - 1 and piece.colors[2] == color:  # Bottom
                                bottom_piece = self.get_piece(row + 1, col)
                                if bottom_piece and bottom_piece.colors[0] == color:
                                    stack.append((row + 1, col, "top-left"))
                            if col > 0 and piece.colors[2] == color:  # Left
                                left_piece = self.get_piece(row, col - 1)
                                if left_piece and left_piece.colors[3] == color:
                                    stack.append((row, col - 1, "bottom-right"))
                        if direction == "bottom-right":
                            if row < self.rows - 1 and piece.colors[3] == color:  # Bottom
                                bottom_piece = self.get_piece(row + 1, col)
                                if bottom_piece and bottom_piece.colors[1] == color:
                                    stack.append((row + 1, col, "top-right"))
                            if col < self.cols - 1 and piece.colors[3] == color:  # Right
                                right_piece = self.get_piece(row, col + 1)
                                if right_piece and right_piece.colors[2] == color:
                                    stack.append((row, col + 1, "bottom-left"))
            return cluster

        for row in range(self.rows):
            for col in range(self.cols):
                piece = self.get_piece(row, col)
                if piece:
                    for i, color in enumerate(piece.get_colors()):
                        if color != Color.NULL:
                            direction = ["top-left", "top-right", "bottom-left", "bottom-right"][i]
                            cluster = dfs(row, col, color, direction)
                            if len(cluster) > 1:
                                clusters.append((color, cluster))
        return clusters

    # Remove clusters e atualiza o goal
    def pop_clusters(self):
        while True:
            clusters = self.find_clusters()
            if not clusters:
                break
            for color, cluster in clusters:
                total_popped = 0
                for row, col in cluster:
                    total_popped += self.pop_color_at(row, col, color)
                self.goal.pop_color(color, total_popped)

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
            return self.pieces.popleft()
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
    












# Example usage
pieces = [
    #Piece(Color.BLUE, Color.RED, Color.BLUE, Color.BLUE),
    Piece(Color.BLUE, Color.RED, Color.BLUE, Color.YELLOW),
    Piece(Color.RED, Color.BLUE, Color.YELLOW, Color.BLUE),
    Piece(Color.YELLOW, Color.RED, Color.GREEN, Color.BLUE),
    Piece(Color.RED, Color.YELLOW, Color.BLUE, Color.GREEN),
    Piece(Color.GREEN, Color.BLUE, Color.YELLOW, Color.RED),
    Piece(Color.RED, Color.YELLOW, Color.BLUE, Color.YELLOW),
    Piece(Color.YELLOW, Color.BLUE, Color.RED, Color.GREEN),
    Piece(Color.GREEN, Color.GREEN, Color.YELLOW, Color.BLUE),
]

goal = Goal(blue=20, green=20, red=20, yellow=20)
hand = Hand(max_pieces=2)
queue = Queue(pieces)
game = Game(3, 3, goal, hand, queue)

# Place pieces on the board
game.place_piece(0, 0, 0)  
game.place_piece(0, 1, 1)  
game.place_piece(1, 0, 1)  




# PARA VER O TABULEIRO ANTES E DEPOIS DO POP, COMENTAR LINHA 336 E DESCOMENTAR AS LINHAS SEGUINTES:
#print("Before popping:")
#print(game)
#game.board.pop_clusters()

print("\nAfter popping clusters:")
print(game)