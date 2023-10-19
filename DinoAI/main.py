import pygame
from random import uniform, choices
import pygame.font
import numpy as np

pygame.font.init()
pygame.init()

# Load sprites
cactus_big = pygame.image.load("data/cactus.png")
cactus_big = pygame.transform.scale_by(cactus_big, 0.6)
berd = pygame.image.load("data/berd.png")
berd = pygame.transform.scale_by(berd, 0.6)

# Initialize essential pygame variables
screen = pygame.display.set_mode((1280, 1280), pygame.RESIZABLE)
clock = pygame.time.Clock()
font = pygame.font.Font(None, 30)
pygame.display.set_caption("Dino Game AI")

# Core variables
running = True
obstacles = []
population = []
speed = 2
gens = 1
scores_list = []


# Function initializing NN weights
def weights_init(input_size, hidden_size1, hidden_size2, output_size):
    variance_in = 2.0 / (input_size + hidden_size1)
    std_dev_in = np.sqrt(variance_in)
    weights_in = np.random.normal(0, std_dev_in, size=(input_size, hidden_size1))

    variance_hid1 = 2.0 / (hidden_size1 + hidden_size2)
    std_dev_hid1 = np.sqrt(variance_hid1)
    weights_hid1 = np.random.normal(0, std_dev_hid1, size=(hidden_size1, hidden_size2))

    variance_hid2 = 2.0 / (hidden_size2 + output_size)
    std_dev_hid2 = np.sqrt(variance_hid2)
    weights_hid2 = np.random.normal(0, std_dev_hid2, size=(hidden_size2, output_size))

    variance_out = 2.0 / (hidden_size2 + output_size)
    std_dev_out = np.sqrt(variance_out)
    weights_out = np.random.normal(0, std_dev_out, size=(hidden_size2, output_size))

    return weights_in, weights_hid1, weights_hid2, weights_out

#Create the neural network
class NN:
    def __init__(self, inputs: int, hiddens1: int, hiddens2: int, outputs: int):
        self.inputs = inputs
        self.hiddens1 = hiddens1
        self.hiddens2 = hiddens2
        self.outputs = outputs
        (
            self.weights_in,
            self.weights_hid1,
            self.weights_hid2,
            self.weights_out,
        ) = weights_init(inputs, hiddens1, hiddens2, outputs)

    @staticmethod
    def relu(x):
        return np.maximum(0.0, x)

    def calculate(self, input_values):
        hidden1_values = self.relu(np.dot(input_values, self.weights_in))
        hidden2_values = self.relu(np.dot(hidden1_values, self.weights_hid1))
        output_values = self.relu(np.dot(hidden2_values, self.weights_out))
        return output_values

    def mutate(self, w_in, w_hid1, w_hid2, w_out):
        self.weights_in = w_in.copy()
        self.weights_hid1 = w_hid1.copy()
        self.weights_hid2 = w_hid2.copy()
        self.weights_out = w_out.copy()

        for weights in [
            self.weights_in,
            self.weights_hid1,
            self.weights_hid2,
            self.weights_out,
        ]:
            for i in range(len(weights)):
                for j in range(len(weights[i])):
                    weights[i][j] *= uniform(0.8, 1.2)

    def copy(self, w_in, w_hid1, w_hid2, w_out):
        self.weights_in = w_in.copy()
        self.weights_hid1 = w_hid1.copy()
        self.weights_hid2 = w_hid2.copy()
        self.weights_out = w_out.copy()


# Create player
class Player:
    def __init__(self):
        self.rect = pygame.Rect(200, screen.get_height() / 2, 50, 100)
        self.movey = 0
        self.alive = True
        # self.isduck = False
        self.score = 0
        self.nn = NN(7, 5, 5, 3)
        self.show = True

    def jump(self):
        if (
            self.rect.y == screen.get_height() / 2
            and self.rect.width == 50
            and self.rect.height == 100
        ):
            self.movey -= 150

    def duck(self):
        self.rect.width = 80
        self.rect.height = 40
        self.rect.y = screen.get_height() / 2 + 60

    def update(self):
        self.rect.y += self.movey
        self.movey = 0
        if pygame.Rect.colliderect(
            self.rect,
            front_obstacles[0] if front_obstacles else pygame.Rect(0, 0, 10, 10),
        ):
            self.alive = False
        if self.rect.y < screen.get_height() / 2:
            self.rect.y += 1
        if self.show:
            pygame.draw.rect(screen, "grey", self.rect, 5)


# Create obstacle
class Obstacle:
    def __init__(self):
        choice = choices(
            [0, 1, 2, 3],
            weights=[0.7, 0.1, 0.1, 0.1],
        )[0]
        if choice == 0:
            self.rect = cactus_big.get_rect()
            self.rect.topleft = (screen.get_width() + 100, screen.get_height() / 2 + 28)
        elif choice == 1:
            self.rect = berd.get_rect()
            self.rect.topleft = (screen.get_width() + 100, screen.get_height() / 2 -20)
        elif choice == 2:
            self.rect = berd.get_rect()
            self.rect.topleft = (screen.get_width() + 100, screen.get_height() / 2 - 5)
        else:
            self.rect = berd.get_rect()
            self.rect.topleft = (screen.get_width() + 100, screen.get_height() / 2 + 40)
        self.onscreen = True

    def update(self):
        global speed
        self.rect.x -= speed
        if self.rect.y == screen.get_height() / 2 + 28:
            screen.blit(
                cactus_big, self.rect
            )
            if self.rect.x < -cactus_big.get_width():
                self.onscreen = False
        else :
            screen.blit(
                berd, self.rect
            )
            if self.rect.x < -berd.get_width():
                self.onscreen = False
        if self.rect.x < -60:
            self.onscreen = False

# Function to end a gen, create a new one, and mutate the players at the same time
def new_gen() :
    global gens
    global obstacles
    global front_obstacles
    global scores_list

    # Reset the obstacles lists
    obstacles = []
    front_obstacles = []
    
    # Find last best player and its NN weights
    best = max(population, key=lambda x: x.score)
    score = best.score
    inp = best.nn.weights_in
    h1 = best.nn.weights_hid1
    h2 = best.nn.weights_hid2
    out = best.nn.weights_out

    # Mutate and reset all but one player
    for i in population[:-1]:
        i.score = 0
        i.nn.mutate(inp,h1,h2,out)
        i.alive = True
        i.rect = pygame.Rect(200, screen.get_height() / 2, 50, 100)

    # Keep the same last best weights and reset one player
    last = population[-1]
    last.nn.copy(inp,h1,h2,out)
    last.alive = True
    last.rect = pygame.Rect(200, screen.get_height() / 2, 50, 100)
    last.score = 0

    # Save score gen and weights data to file
    f = open("data/weights_hist.txt", "a")
    f.write(f"GEN : {gens} ; SCORE : {int(score)} ; WEIGHTS : {inp.tolist()}, {h1.tolist()}, {h2.tolist()}, {out.tolist()}\n")
    f.close()
    
    # Keep track of gens and scores
    gens += 1
    scores_list.append(score)

# Check if everyone is dead
def alldead():
    for i in population:
        if i.alive == True:
            return False
    return True


# Initialize population
population = [Player() for _ in range(500)]

while running:
    keys = pygame.key.get_pressed()
    screen.fill("white")

    # Check obstacles in front of the player
    front_obstacles = list(filter(lambda e: e.rect.x >= 205, obstacles))

    # show last best (might be dead and not show up)
    if keys[pygame.K_SPACE]:
        for i in population[:-1]:
            i.show = not i.show
    # Draw floor
    pygame.draw.rect(
        screen,
        "black",
        (0, screen.get_height() / 2 + 100, screen.get_width(), screen.get_height() / 2),
    )
    # One generation
    if not alldead():
        # Set speed
        speed = max(population, key=lambda x: x.score).score / 500 + 2
        # Spawn randomly obstacles
        if (
            choices([0, 1], weights=(0.99, 0.01))[0] == 1
            and obstacles[-1].rect.x < screen.get_width() / 2 + 250
            if obstacles
            else True
        ):
            obstacles.append(Obstacle())
        # Check if obstacle is on the screen, if not stop rendering it
        for obstacle in obstacles:
            if obstacle.onscreen:
                obstacle.update()
            else:
                obstacles.remove(obstacle)
        # Make move each player
        for player in population:
            if player.alive:
                closest_obstacle = front_obstacles[0] if front_obstacles else False
                if closest_obstacle:
                    # Set input values of neural network
                    input_values = [
                        closest_obstacle.rect.x - player.rect.x,
                        closest_obstacle.rect.height,
                        closest_obstacle.rect.width,
                        closest_obstacle.rect.y,
                        speed,
                        player.rect.y,
                        front_obstacles[1].rect.x - closest_obstacle.rect.x
                        if len(front_obstacles) >= 2
                        else screen.get_width() - player.rect.x,
                    ]
                    output_values = player.nn.calculate(input_values)
                    movement_index = np.argmax(output_values)
                    if movement_index == 0:
                        player.duck()
                    if movement_index == 1:
                        player.jump()
                    elif movement_index == 2 and player.rect.width == 80:
                        player.rect = pygame.Rect(200, screen.get_height() / 2, 50, 100)
                player.score += 0.1
                player.update()
    # Generation is done
    else:
        new_gen()
    
    # Find an alive player to check its score and render all the text
    for i in population:
        if i.alive:
            score_text = font.render(f"Score : {int(i.score)} ", True, (0, 0, 0))
            best_text = font.render(
                f"Best Score : {int(max(scores_list) if scores_list else 0)}",
                True,
                (0, 0, 0),
            )
            gens_text = font.render(f"Generation : {gens}", True, (0, 0, 0))
            screen.blit(score_text, (10, 10))
            screen.blit(best_text, (10, 40))
            screen.blit(gens_text, (10, 70))
            break

    # Check if closed
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            
    pygame.display.flip()
    clock.tick(200)

pygame.quit()
