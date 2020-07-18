#!/usr/bin/python

import pygame
from pygame.locals import *
#import random
import sys
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import os


BOARD_DIM = np.array((20,20))
SCREEN_DIM = (BOARD_DIM * 32).astype(int)

BIRTH_CHANCE =0.25


player1="tribe1"
player2="tribe2"


colorKey = {"air":(0,0,0),
            "dirt":(90,60,30),
            "water":(50,50,200),
            "farm":(50,200,50),
            "food":(200,200,50),
            "colony":(150,150,150)}
            
name = {0:"air",
            1:"dirt",
            2:"water",
            3:"farm",
            4:"food",
            5:"colony"}
            
ID = dict([(value, key) for key, value in name.items()])

class Tile:

    def __init__(self,bpos,material,tribe=None):
        self.bpos = np.array(bpos)
        self.material = material
        self.set_textures()
        self.tribe = tribe
        
    def set_textures(self):
        texture = pygame.image.load(os.path.join('textures', self.material+'.png')).convert_alpha()
        self.texture = pygame.transform.scale(texture,np.array(SCREEN_DIM/BOARD_DIM).astype(int)) 
            
    def color(self):
        colorKey = {"air":(0,0,0),
                "dirt":(90,60,30),
                "water":(50,50,200),
                "farm":(50,200,50),
                "food":(200,200,50),
                "colony":(150,150,150)}
        color = colorKey[self.material]
        return color
    
    def place(self,material):
        self.material = material
        self.set_textures()
        
    def draw(self,Surface):
            i,j = self.bpos
            h,k = SCREEN_DIM/BOARD_DIM
            m = SCREEN_DIM[1]
            x = (int(h*i),m-int(k*(j+1)))
            Surface.blit(self.texture, x)
        



class Environment:

    def generate_grid(self,dim):
        m,n = dim
        grid = []
        for i in range(m):
            for j in range(n):
                grid.append(Tile((i,j),'dirt'))
        return np.array(grid).reshape(dim)
    
    def get_tile(self,bpos):
        i,j = bpos
        return self.grid[i,j]
    
    def __init__(self,dim):
        self.dim = dim
        self.grid = self.generate_grid(dim)
        self.occupied = np.zeros(dim)
        self.smell_matrices = dict()
        self.food_growth_rate = 0.05
        
    def get_material(self,bpos):
        i,j = bpos
        tile = self.grid[i,j]
        return tile.material
        
    def place_tile(self,bpos,material):
        i,j = bpos
        self.grid[i,j].material = material
        
    def draw_environment(self,Surface):
        m,n = self.dim
        for i in range(m):
            for j in range(n):
                tile = self.grid[i,j]
                tile.draw(Surface)
    
    
    def update_smell_matrix(self,material,tribe,var):
        m,n = self.dim
        S = np.zeros(self.dim)
        for i in range(m):
            for j in range(n):
                tile = self.grid[i,j]
                if tile.material == material and (tribe == tile.tribe or tile.tribe == None):
                    one_hot = np.zeros(self.dim)
                    one_hot[i,j]=1
                    # Gaussian kernel allows ant to smell food at any distance albeit with lower density
                    G = gaussian_filter(one_hot, sigma=var,mode='constant')
                    S = np.amax((S,G),axis=0)
        if tribe == None:
            self.smell_matrices[material] = S
        else:
            self.smell_matrices[(material,tribe)] = S
    
    def smell(self,material,tribe,bpos):
        i,j = bpos
        if material=="food":
            smell_matrix = self.smell_matrices[material]
        else:
            smell_matrix = self.smell_matrices[(material,tribe)]
        
        return smell_matrix[i,j]
        
    def grow_environment(self,Surface):
        m,n = self.dim
        for i in range(m):
            for j in range(n):
                tile = self.grid[i,j]
                if tile.material == "farm":
                    if np.random.binomial(2,self.food_growth_rate):
                        tile.place("food")
                        tile.draw(Surface)
                        self.grid[i,j] = tile
            

class Ant:
    def __init__(self,environment,tribe,pos=(0,0),texture=None):
        self.environment = environment
        self.tribe = tribe
        self.job = "worker"
        self.current_pos = np.array(pos)
        self.next_pos = np.array(pos)
        
        self.dead = False
        self.hasFood = False
        self.goal = "food"
        
        
        self.job = "worker"
        self.set_texture(texture,color)
        
    def set_texture(self,text=None,color=(0,0,0)):
        if text!=None:
            texture = pygame.image.load(os.path.join('textures', text)).convert_alpha()
            self.texture = pygame.transform.scale(texture,np.array(SCREEN_DIM/BOARD_DIM).astype(int)) 
        else:
            self.texture = pygame.Surface(np.array(SCREEN_DIM/BOARD_DIM).astype(int))
            
    def neighborhood(self):
        e = self.environment
        dim = e.dim
        (x,y) = self.current_pos
        x_coords = np.arange( max(x-1,0),min(x+2,dim[0]))
        y_coords = np.arange( max(y-1,0),min(y+2,dim[1]))
        nbhd = [e.grid[x,y] for x in x_coords for y in y_coords]
        nbhd = [tile for tile in nbhd if tile.material != 'water']
        return {i:nbhd[i] for i in range(len(nbhd))}
        
    def get_pos(self):
        return self.current_pos
        
    def select_position(self):
        e = self.environment
        self.current_pos = self.next_pos
        
        nbhd = self.neighborhood()
        
        if len(nbhd)>0:
            # Smell for goal (i.e. food) in nearby radius
            w = [e.smell(self.goal, self.tribe, tile.bpos) for tile in nbhd.values()]
            
            # Add some noise (important when there is no food around)
            w = w + 0.00001*np.random.random(len(w))
            
            weights = w / np.sum(w)
            k=np.argmax(weights)
            
            self.next_pos = nbhd[k].bpos
        else:
            self.next_pos = self.current_pos
        
        
    
        
    def draw_ant(self,Surface,r):
        q=min(max(0,2*r-.5),1)
        
        bpos = (1-q)*self.current_pos + q*self.next_pos
        color = (0,0,0)

        i,j = bpos
        h,k = SCREEN_DIM/BOARD_DIM
        m = SCREEN_DIM[1]
        x = (int(h*i),m-int(k*(j+1)))
        Surface.blit(self.texture, x)
        



def main():

    # Initialise screen
    pygame.init()
    pygame.display.set_caption('pyAnts')
    screen = pygame.display.set_mode(SCREEN_DIM)
    clock=pygame.time.Clock()

    # Fill background
    bg = pygame.Surface(screen.get_size())
    bg = bg.convert()

    # main environment called E
    e = Environment(BOARD_DIM)
    
    # Blit everything to the screen
    screen.blit(bg, (0,0))
    pygame.display.flip()
    

    
    FPS = 60
    SPS = 15
    n=int(FPS/SPS)
    t = 0
    
    # some map making... need to be able to save to file
    tile1 = e.grid[5,5]
    tile1.place('colony')
    tile1.tribe = player1
    
    tile2 = e.grid[5,15]
    tile2.place('colony')
    tile2.tribe = player2
    
    tile3 = e.grid[15,10]
    tile3.place('farm')
    
    e.draw_environment(bg)
    
    ANTS = []
    
    for i in range(1):
        ANTS += [Ant(e,player1,(5,5),texture="worker1.png")]
        ANTS += [Ant(e,player2,(5,15),texture="worker2.png")]

    paused = False
    brush='dirt'
    
    # Event loop
    t = 0
    while 1:

        screen.blit(bg, (0,0))

        r = t%1
        if (not paused) or r != 0:
            if r==0:
                # grow envionment every 10 steps
                if t % 10 == 0:
                    e.grow_environment(bg)
                    
                e.update_smell_matrix("food",tribe=None,var=5)
                e.update_smell_matrix("colony",tribe=player1,var=5)
                e.update_smell_matrix("colony",tribe=player2,var=5)
                
                # Each ant does brownian walk
                for ant in ANTS:
                    if ant.dead:
                        ANTS.remove(ant)
                    else:
                        ant.select_position()
                        
                for ant in ANTS:
                    bpos = ant.get_pos()
                    tile = e.get_tile(bpos)
                    
                    # kill ants on water
                    if tile.material=='water':
                        ant.dead=True
                    
                    # else do something when ant reaches their goal tile
                    elif tile.material == ant.goal:
                        if ant.goal == "food":
                            ant.hasFood=True
                            ant.goal = "colony"
                            tile.place('farm')
                            tile.draw(bg)
                            
                        elif ant.goal == "colony" and ant.tribe == tile.tribe:
                            ant.hasFood=True
                            ant.goal = "food"
                            ant.hasFood = False
                            
                            if np.random.binomial(2,BIRTH_CHANCE):
                                txt = {player1:"worker1.png",player2:"worker2.png"}[ant.tribe]
                                ant = Ant(e,ant.tribe,bpos,texture=txt)
                                ANTS += [ant]
                                
            # iterate time
            t += 1/n
            
        # draw ants
        for ant in ANTS:
            ant.draw_ant(screen, r)
               

        pygame.display.flip()
        clock.tick(FPS)
        
        
        ##############################
        ## Player control
        
        if pygame.mouse.get_pressed()[0]:
            x,y = np.array(pygame.mouse.get_pos())
            m=SCREEN_DIM[1]
            bpos = (np.array((x,m-y)) / SCREEN_DIM * BOARD_DIM).astype(int)
            
            if brush!="ant":
                i,j = bpos
                tile = e.grid[i,j]
                tile.place(brush)
                tile.draw(bg)
                if brush=="colony":
                    tile.tribe = player1
            if brush=="ant":
                ant = Ant(e,player1,bpos,texture="worker1.png")
                ANTS += [ant]
             

        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                    if paused:
                        print('Paused')
                    else:
                        print('Unpaused')

                if event.key == pygame.K_x:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_1:
                    print("Dirt selected")
                    brush="dirt"
                if event.key == pygame.K_2:
                    print("Water selected")
                    brush="water"
                if event.key == pygame.K_4:
                    print("Farm selected")
                    brush="farm"
                if event.key == pygame.K_3:
                    print("Ant selected")
                    brush="ant"
                if event.key == pygame.K_5:
                    print("Colony selected")
                    brush="colony"
                if event.key == pygame.K_6:
                    print("Food selected")
                    brush="food"
                
                if event.key == pygame.K_g:
                    e = Environment(BOARD_DIM)
                    e.draw_environment(bg)
                if event.key == pygame.K_k:
                    ANTS = []
                if event.key == pygame.K_UP:
                    SPS = min(FPS,2*SPS)
                    n=int(FPS/SPS)
                if event.key == pygame.K_DOWN:
                    SPS /= 2
                    n=int(FPS/SPS)

if __name__ == '__main__': main()