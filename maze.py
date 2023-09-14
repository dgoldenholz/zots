import random
import time
import matplotlib.pyplot as plt
import numpy as np
import zot_constants as CC

# maze code was adopted and changed based on original code from
#OrWestSide  Orestis Zekai
#https://github.com/OrWestSide/python-scripts/blob/master/maze.py

# Maze generator -- Randomized Prim Algorithm
class maze():
    def __init__(self,height=10,width=20):
        self.height = height
        self.width = width
        # Init variables
        wall = 'w'
        cell = 'c'
        unvisited = 'u'
        self.maze = []
        self.grid = []
        self.scores = []

        # Denote all cells as unvisited
        for i in range(0, self.height):
            line = []
            for j in range(0, self.width):
                line.append(unvisited)
            self.maze.append(line)

        # Randomize starting point and set it a cell
        starting_height = int(random.random()*self.height)
        starting_width = int(random.random()*self.width)
        if (starting_height == 0):
            starting_height += 1
        if (starting_height == self.height-1):
            starting_height -= 1
        if (starting_width == 0):
            starting_width += 1
        if (starting_width == self.width-1):
            starting_width -= 1

        # Mark it as cell and add surrounding walls to the list
        self.maze[starting_height][starting_width] = cell
        walls = []
        walls.append([starting_height - 1, starting_width])
        walls.append([starting_height, starting_width - 1])
        walls.append([starting_height, starting_width + 1])
        walls.append([starting_height + 1, starting_width])

        # Denote walls in maze
        self.maze[starting_height-1][starting_width] = 'w'
        self.maze[starting_height][starting_width - 1] = 'w'
        self.maze[starting_height][starting_width + 1] = 'w'
        self.maze[starting_height + 1][starting_width] = 'w'

        while (walls):
            # Pick a random wall
            rand_wall = walls[int(random.random()*len(walls))-1]

            # Check if it is a left wall
            if (rand_wall[1] != 0):
                if (self.maze[rand_wall[0]][rand_wall[1]-1] == 'u' and self.maze[rand_wall[0]][rand_wall[1]+1] == 'c'):
                    # Find the number of surrounding cells
                    s_cells = self.surroundingCells(rand_wall)

                    if (s_cells < 2):
                        # Denote the new path
                        self.maze[rand_wall[0]][rand_wall[1]] = 'c'

                        # Mark the new walls
                        # Upper cell
                        if (rand_wall[0] != 0):
                            if (self.maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]-1, rand_wall[1]])


                        # Bottom cell
                        if (rand_wall[0] != self.height-1):
                            if (self.maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]+1, rand_wall[1]])

                        # Leftmost cell
                        if (rand_wall[1] != 0):	
                            if (self.maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                            if ([rand_wall[0], rand_wall[1]-1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]-1])
                    

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)

                    continue

            # Check if it is an upper wall
            if (rand_wall[0] != 0):
                if (self.maze[rand_wall[0]-1][rand_wall[1]] == 'u' and self.maze[rand_wall[0]+1][rand_wall[1]] == 'c'):

                    s_cells = self.surroundingCells(rand_wall)
                    if (s_cells < 2):
                        # Denote the new path
                        self.maze[rand_wall[0]][rand_wall[1]] = 'c'

                        # Mark the new walls
                        # Upper cell
                        if (rand_wall[0] != 0):
                            if (self.maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]-1, rand_wall[1]])

                        # Leftmost cell
                        if (rand_wall[1] != 0):
                            if (self.maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                            if ([rand_wall[0], rand_wall[1]-1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]-1])

                        # Rightmost cell
                        if (rand_wall[1] != self.width-1):
                            if (self.maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                            if ([rand_wall[0], rand_wall[1]+1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]+1])

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)

                    continue

            # Check the bottom wall
            if (rand_wall[0] != self.height-1):
                if (self.maze[rand_wall[0]+1][rand_wall[1]] == 'u' and self.maze[rand_wall[0]-1][rand_wall[1]] == 'c'):

                    s_cells = self.surroundingCells(rand_wall)
                    if (s_cells < 2):
                        # Denote the new path
                        self.maze[rand_wall[0]][rand_wall[1]] = 'c'

                        # Mark the new walls
                        if (rand_wall[0] != self.height-1):
                            if (self.maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]+1, rand_wall[1]])
                        if (rand_wall[1] != 0):
                            if (self.maze[rand_wall[0]][rand_wall[1]-1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]-1] = 'w'
                            if ([rand_wall[0], rand_wall[1]-1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]-1])
                        if (rand_wall[1] != self.width-1):
                            if (self.maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                            if ([rand_wall[0], rand_wall[1]+1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]+1])

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)


                    continue

            # Check the right wall
            if (rand_wall[1] != self.width-1):
                if (self.maze[rand_wall[0]][rand_wall[1]+1] == 'u' and self.maze[rand_wall[0]][rand_wall[1]-1] == 'c'):

                    s_cells = self.surroundingCells(rand_wall)
                    if (s_cells < 2):
                        # Denote the new path
                        self.maze[rand_wall[0]][rand_wall[1]] = 'c'

                        # Mark the new walls
                        if (rand_wall[1] != self.width-1):
                            if (self.maze[rand_wall[0]][rand_wall[1]+1] != 'c'):
                                self.maze[rand_wall[0]][rand_wall[1]+1] = 'w'
                            if ([rand_wall[0], rand_wall[1]+1] not in walls):
                                walls.append([rand_wall[0], rand_wall[1]+1])
                        if (rand_wall[0] != self.height-1):
                            if (self.maze[rand_wall[0]+1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]+1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]+1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]+1, rand_wall[1]])
                        if (rand_wall[0] != 0):	
                            if (self.maze[rand_wall[0]-1][rand_wall[1]] != 'c'):
                                self.maze[rand_wall[0]-1][rand_wall[1]] = 'w'
                            if ([rand_wall[0]-1, rand_wall[1]] not in walls):
                                walls.append([rand_wall[0]-1, rand_wall[1]])

                    # Delete wall
                    for wall in walls:
                        if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                            walls.remove(wall)

                    continue

            # Delete the wall from the list anyway
            for wall in walls:
                if (wall[0] == rand_wall[0] and wall[1] == rand_wall[1]):
                    walls.remove(wall)
            


        # Mark the remaining unvisited cells as walls
        for i in range(0, self.height):
            for j in range(0, self.width):
                if (self.maze[i][j] == 'u'):
                    self.maze[i][j] = 'w'

        # Set entrance and exit
        for i in range(0, self.width):
            if (self.maze[1][i] == 'c'):
                self.maze[0][i] = 'c'
                break

        for i in range(self.width-1, 0, -1):
            if (self.maze[self.height-2][i] == 'c'):
                self.maze[self.height-1][i] = 'c'
                break
            
        return

    def get_starting_finishing_points(self):
        _start = [i for i in range(len(self.maze[0])) if self.maze[0][i] == 'c']
        _end = [i for i in range(len(self.maze[0])) if self.maze[len(self.maze)-1][i] == 'c']
        #return [0, _start[0]], [self.height - 1, _end[0]]
        return [_start[0],0], [ _end[0], self.height - 1]
        

    def get_startXY(self):
        startxy,endxy = self.get_starting_finishing_points()
        return startxy[0], startxy[1]
    
    def convertToGrid(self):
        img = np.ones((self.width,self.height))
        #img = np.zeros((self.height,self.width))

        for thisRow in range(0, self.height):
            for thisCol in range(0, self.width):
                if (self.maze[thisRow][thisCol] == 'u'):
                    img[thisCol,thisRow] = 0.5
                elif (self.maze[thisRow][thisCol] == 'c'):
                    img[thisCol,thisRow] = 0
                else:
                    img[thisCol,thisRow] = 1
        self.grid = img
        print(img.shape)
        return img
    
    def scoreGrid(self):
        startXY,endXY = self.get_starting_finishing_points()
        print(endXY)
        self.scores = np.ones((self.width,self.height)) * np.inf
        self.scores[endXY[0],endXY[1]] = 0
        iter = 0

        while np.isinf(self.scores[startXY[0],startXY[1]])==True:
            
            iter +=1
            for x in range(self.width):
                for y in range(self.height):
                    # is this location a hall? if not, do nothing
                    if self.grid[x,y]==0:
                        UPx , UPy = x , y-1
                        DOWNx, DOWNy = x , y+1
                        LEFTx, LEFTy = x-1, y
                        RIGHTx, RIGHTy = x+1, y
                        thisScore = self.scores[x,y]
                        self.scores[x,y] = np.min([thisScore,
                            self.checkValid(UPx,UPy),
                            self.checkValid(DOWNx,DOWNy),
                            self.checkValid(LEFTx,LEFTy),
                            self.checkValid(RIGHTx,RIGHTy)] )
        print(f'Solved in {iter} cycles.')
    
    def getLocalScore(self,x,y):
        return self.scores[x,y]
    
    def isValid(self,x,y):
        # if out of bounds, or this is a wall, return infinity
        if x<0 or y<0 or x>=self.width or y>=self.height or (self.grid[x,y]==1):
            return False
        else:
            return True
        
    def checkValid(self,x,y):
        # if out of bounds, or this is a wall, return infinity
        if self.isValid(x,y)==False:
            return np.inf
        else:
            return self.scores[x,y] + 1
                    
    def drawMaze(self,doSHOW=True):
        plt.imshow(np.transpose(self.grid))
        if doSHOW==True:
            plt.show()

    def drawMaze_andLocation(self,x,y,ax=None):
        self.drawMaze(doSHOW=False)
        # figure out how to put a full zot in there
        if ax==None:
            plt.plot(x,y,'rx')
        else:
            ax.plot(x,y,'rx')

    def drawMaze_andZot(self,one_zot,ax=None):
        one_zot.ran_by_DNA(self)
        xlist,ylist = one_zot.report_DNA_path()
        x,y = one_zot.showPos()
        self.drawMaze_andLocation(x,y,ax)
        #if ax == None:
        #    plt.plot(xlist,ylist,'ro')
        #    plt.show()
        #else:
        #    ax.plot(xlist,ylist,'ro')

        for iter in range(len(xlist)):
            if ax == None:
                plt.plot(xlist[iter],ylist[iter],'ro')
                
            else:
                ax.plot(xlist[iter],ylist[iter],'ro')        
        if ax==None:
            plt.show()
    
    def isWall(self,x,y):
        # returns 1 if wall, 0 if hall
        return self.grid[x,y]==1
    
    def drawScores(self):
        plt.imshow(np.transpose(self.scores),norm='linear')
        plt.colorbar()
        plt.show()

    # Find number of surrounding cells
    def surroundingCells(self,rand_wall):
        s_cells = 0
        if (self.maze[rand_wall[0]-1][rand_wall[1]] == 'c'):
            s_cells += 1
        if (self.maze[rand_wall[0]+1][rand_wall[1]] == 'c'):
            s_cells += 1
        if (self.maze[rand_wall[0]][rand_wall[1]-1] == 'c'):
            s_cells +=1
        if (self.maze[rand_wall[0]][rand_wall[1]+1] == 'c'):
            s_cells += 1

        return s_cells


