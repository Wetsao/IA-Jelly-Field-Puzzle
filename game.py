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

    def __str__(self):
        return f"{self.colors[0]} {self.colors[1]} | {self.colors[2]} {self.colors[3]}"

class Board:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[None for _ in range(cols)] for _ in range(rows)]

    def place_piece(self, row, col, piece):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.grid[row][col] = piece
        else:
            raise IndexError("Position out of bounds")

    def get_piece(self, row, col):
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.grid[row][col]
        else:
            raise IndexError("Position out of bounds")

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
    

class Goal:
    def __init__(self, blue, green, red, yellow):
        self.goal = {
            Color.BLUE: blue,
            Color.GREEN: green,
            Color.RED: red,
            Color.YELLOW: yellow
        }

    def pop_color(self, color):
        if color in self.goal and self.goal[color] > 0:
            self.goal[color] -= 1

    def is_goal_met(self):
        return all(count == 0 for count in self.goal.values())

    def __str__(self):
        return f"Goal: {self.goal}"



class Hand:
    def __init__(self, pieces):
        self.pieces = pieces

    def get_pieces(self):
        return self.pieces

    def __str__(self):
        return "Hand: " + ", ".join(str(piece) for piece in self.pieces)


    


# Example usage
piece1 = Piece(Color.BLUE, Color.GREEN, Color.RED, Color.YELLOW)
piece2 = Piece(Color.YELLOW, Color.RED, Color.GREEN, Color.BLUE)
piece3 = Piece(Color.RED, Color.YELLOW, Color.BLUE, Color.GREEN)

board = Board(3, 3)
board.place_piece(0, 0, piece1)
board.place_piece(1, 1, piece2)
board.place_piece(2, 2, piece3)

print(board)

goal = Goal(blue=3, green=2, red=4, yellow=1)
print(goal)
