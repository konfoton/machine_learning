import pygame
import neat
import math
import time
def game(genomes, config):
    a = time.time()
    pygame.init()
    clock = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("AI LEARNS TO WALK")
    #special rect
    SURFACE = pygame.Surface((50, 50))
    SURFACE.fill((0, 255, 0))
    RECT = SURFACE.get_rect(center=(550, 60))
    class Rect:
        def __init__(self, genome):
            self.surface = pygame.Surface((30, 30))
            self.surface.fill((255, 0, 0))
            self.genome = genome
            self.genome.fitness = 0
            self.rect = self.surface.get_rect(center=(30, 500))
            self.net = neat.nn.FeedForwardNetwork.create(genome, config)
            self.old = math.sqrt((self.rect.x - RECT.x) ** 2 + (self.rect.y- RECT.y) ** 2)
            self.oldx = self.rect.x
            self.oldy = self.rect.y
            self.dx = 0
            self.dy = 0

        def draw(self):
            SCREEN.blit(self.surface, self.rect)

        def movement(self):
            output = self.net.activate([math.sqrt((self.rect.x - RECT.x) ** 2 + (self.rect.y- RECT.y) ** 2), abs(self.rect.x - RECT.x)])
            #move up
            if output[0] > 0.5: 
                self.rect.y -= 1
            #move down
            if output[1] > 0.5: 
                self.rect.y += 1
            #move right
            if output[2] > 0.5: 
                self.rect.x += 5
            #move left
            if output[3] > 0.5: 
                self.rect.x -= 1
            self.dx = abs(self.rect.x - self.oldx)
            self.dx = abs(self.rect.y - self.oldy)
            self.new = math.sqrt((self.rect.x - RECT.x) ** 2 + (self.rect.y- RECT.y) ** 2)

        def fitness(self):
            self.genome.fitness += self.old - self.new
            if self.dx == 0 or self.dy == 0:
                genome.fitness -= 0.1
            self.oldx = self.rect.x
            self.oldy = self.rect.y
            self.old = self.new
            if RECT.colliderect(self.rect):
                genome.fitness += 50

    rep = []
    for genome_id, genome in genomes:
        rep.append(Rect(genome))
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        SCREEN.fill((0, 0, 0))
        SCREEN.blit(SURFACE, RECT)
        for instance in rep:
            instance.draw()
            instance.movement()
            instance.fitness()
        clock.tick(60)
        pygame.display.update()
        if time.time() - a > 15:
            pygame.quit()
            running = False
            break
def main(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5))
    winner = p.run(game, 30)
if __name__ == "__main__":
    main("configg.txt")
    