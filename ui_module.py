import pygame


class gui:
    def __init__(self, game, screen, BLOCKSIZE, gamesize):
        self.game = game
        self.screen = screen
        self.BLOCKSIZE = BLOCKSIZE
        self.LILBLOCK = self.BLOCKSIZE / 2
        self.gamesize = gamesize

    def drawlil(self,screen, a, y, x):
        for i in range(4):
            if(i == 0 or i == 3):
                dist = 0
            if(i == 1 or i ==2):
                dist= self.LILBLOCK
            if(i == 0 or i == 1):
                dist2 = 0
            if(i == 2 or i == 3):
                dist2 = self.LILBLOCK
            value = self.getcolor(a[i])
            posx = ((screen.get_width() - self.gamesize) / 2 ) + (x * self.BLOCKSIZE) + dist
            posy =100 + (y * self.BLOCKSIZE) + dist2
            pygame.draw.rect(screen,
            value,
            (posx, posy, self.LILBLOCK,self.LILBLOCK)
            )
            

    def getcolor(self, a):
        if(a == 'B'):
            return (0,0,255)
        elif (a == 'R'):
            return (255, 0 , 0)
        elif (a == 'G'):
            return (0, 255, 0)
        elif (a == 'Y'):
            return (255, 255, 0)
        else:
            return (255, 255, 255)
        
    def draw_grid(self):
        start_x = (self.screen.get_width() - self.gamesize) / 2
        start_y = 100
        
        # Draw vertical lines
        for col in range(self.game.board.cols + 1):
            x = start_x + col * self.BLOCKSIZE
            pygame.draw.line(self.screen, 
                            (0,0,0), 
                            (x, start_y), 
                            (x, start_y + self.game.board.rows * self.BLOCKSIZE), 
                            5)
        
        # Draw horizontal lines
        for row in range(self.game.board.rows + 1):
            y = start_y + row * self.BLOCKSIZE
            pygame.draw.line(self.screen, 
                            (0,0,0), 
                            (start_x, y), 
                            (start_x + self.game.board.cols * self.BLOCKSIZE, y), 
                            5)
        
    def draw_goal(self):
        """Draws the goal information at the top of the screen"""
        panel_height = 80
        panel_color = (240, 240, 240)  # Light gray background
        text_color = (0, 0, 0)  # Black text
        margin = 20
        font_size = 24
        
        # Create goal panel background
        pygame.draw.rect(self.screen, panel_color, (0, 0, self.screen.get_width(), panel_height))
        
        # Load font
        try:
            font = pygame.font.Font(None, font_size)
        except:
            font = pygame.font.SysFont('arial', font_size)
        
        # Draw title
        title = font.render("Goal:", True, text_color)
        self.screen.blit(title, (margin, margin))
        
        # Draw each color's remaining count
        color_info = [
            ("Blue", self.game.goal.goal['B'], (0, 0, 255)),
            ("Green", self.game.goal.goal['G'], (0, 255, 0)),
            ("Red", self.game.goal.goal['R'], (255, 0, 0)),
            ("Yellow", self.game.goal.goal['Y'], (255, 255, 0))
        ]
        
        x_position = 120  # Starting position after "Goal:"
        for name, count, color in color_info:
            if count > 0:  # Only show colors with remaining goals
                # Draw color circle
                pygame.draw.circle(self.screen, color, (x_position, margin + font_size//2), 10)
                
                # Draw text
                text = font.render(f"{name}: {count}", True, text_color)
                self.screen.blit(text, (x_position + 20, margin))
                
                # Update x position for next item
                text_width = text.get_width() + 40  # 40 = circle + spacing
                x_position += text_width + 20  # Additional spacing

    # def draw_hand(self, selected, hand, max_piece):
    #     print("Yet to be implemented")
    def draw_hand(self, selected):
        """Draws all pieces in the hand with white background, centered on screen"""
        if not self.game.hand.pieces:
            return  # No pieces to draw
        
        # Calculate total width needed for all pieces
        piece_width = self.BLOCKSIZE
        total_width = len(self.game.hand.pieces) * piece_width
        spacing = 50  # Space between pieces
        
        # Calculate starting x position to center the pieces
        start_x = (self.screen.get_width() - total_width - 
                ((len(self.game.hand.pieces) - 1) * spacing)) // 2
        hand_y = self.screen.get_height() - 250  # Position above bottom of screen
        
        # Draw white background for each piece
        for i, piece in enumerate(self.game.hand.pieces):
            bg_x = start_x + i * (piece_width + spacing)
            if(selected == i+1):
                pygame.draw.rect(self.screen, 
                                (100, 100, 100),  # White
                                (bg_x-10, hand_y-10, piece_width+20, piece_width+20))
            else:
                pygame.draw.rect(self.screen, 
                                (255, 255, 255),  # White
                                (bg_x-10, hand_y-10, piece_width+20, piece_width+20))
            
            # Draw the piece's sub-blocks (lilblocks)
            colors = piece.get_colors()
            for j in range(4):
                if colors[j] == 'N':
                    continue  # Skip NULL colors
                
                # Calculate sub-block position
                sub_x = bg_x + (j % 2) * self.LILBLOCK
                sub_y = hand_y + (j // 2) * self.LILBLOCK
                
                pygame.draw.rect(self.screen,
                                self.getcolor(colors[j]),
                                (sub_x, sub_y, self.LILBLOCK, self.LILBLOCK))
        
        # Optional: Draw border around each piece
        for i in range(len(self.game.hand.pieces)):
            border_x = start_x + i * (piece_width + spacing)
            pygame.draw.rect(self.screen,
                            (100, 100, 100),  # Dark gray border
                            (border_x, hand_y, piece_width, piece_width),
                            2)  # Border thickness

    def make_board(self):
        for i in range (self.game.board.rows):
            for j in range (self.game.board.cols):
                c = self.game.board.get_piece(i, j)
                if c:
                    b = c.get_colors()
                    self.drawlil(self.screen, b ,i, j)
                else:
                    self.drawlil(self.screen, ["N", "N", "N", "N"], i , j)
        self.draw_grid()
        
        
                    
        

                    