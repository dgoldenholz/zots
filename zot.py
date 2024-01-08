import numpy as np
import zot_constants as CC
import matplotlib.pyplot as plt

class zot():
    def __init__(self,start_x,start_y,randomDNA=True,dnaData=None,
                 number_of_mutations_per_parent=CC.number_of_mutations_per_parent):
        if randomDNA==False:
            self.dna = dnaData
        else:
            self.dna = np.random.randint(low=0,high=4,size=(2,CC.dna_length))
        self.start_x, self.start_y = start_x, start_y
        self.number_of_mutations_per_parent = number_of_mutations_per_parent

    def offer_dna(self):
        # apply genetic crossover
        num_crossovers = np.random.randint(low=0,high=CC.max_crossovers)
        newdna = self.dna.copy()
        lowestInd = 0
        for _ in range(num_crossovers):
            if lowestInd<CC.dna_length-1:
                startInd = np.random.randint(low=lowestInd,high=CC.dna_length-1)
                endInd = np.random.randint(low=startInd,high=CC.dna_length)
                lowestInd = endInd
                newdna[0,startInd:endInd] = self.dna[1,startInd:endInd]
                newdna[1,startInd:endInd] = self.dna[0,startInd:endInd]
            
        # now pick which dna to hand over
        coinFlip = np.random.randint(low=0,high=2)
        thisDNA = newdna[coinFlip,:]

        # apply random mutations
        if np.random.random()<CC.prob_One_Parent_mutates:
            mask = np.random.permutation(np.arange(CC.dna_length))[0:self.number_of_mutations_per_parent]
            thisDNA[mask] = np.random.randint(low=0,high=4,size=self.number_of_mutations_per_parent)
        return thisDNA.copy()
    
    def reset_pos(self):
        self.x, self.y = self.start_x, self.start_y
        self.ind = 0
        self.aliveTF = True
        self.xlist = [self.x]
        self.ylist = [self.y]

    def stepOnce(self,theMaze):
        # get the base from chromosome 0 and 1
        base0 = self.dna[0,self.ind]
        base1 = self.dna[1,self.ind]
        # convert to a protein
        prot0 = CC.protiens[base0,:]
        prot1 = CC.protiens[base1,:]
        
        if 0:
            # movements are autosomal dominant
            full_prot = prot0 + prot1
            # if both prot0 = prot1, don't double it
            if base0==base1:
                full_prot = prot0

            # if there is a vertical and horizontal movement, use these rules
            if abs(full_prot[0])==1 and full_prot[1]==1:
                full_prot[0]=0
            if full_prot[0]==1 and abs(full_prot[1])==1:
                full_prot[1]=0
            if abs(full_prot[0])==1 and full_prot[1]==-1:
                full_prot[1]=0
            if full_prot[0]==-1 and abs(full_prot[1])==1:
                full_prot[0]=0
            
        full_prot = prot0
        # for now, just inactivate dad altogether

        newx= self.x + full_prot[0]
        newy= self.y + full_prot[1]
        
        if theMaze.isValid(newx,newy)==True:
            self.x = newx
            self.y = newy
            self.xlist.append(newx)
            self.ylist.append(newy)
        else:
            self.aliveTF = False

        self.ind += 1
    
    def report_DNA_path(self):
        return self.xlist,self.ylist
    
    def ran_by_DNA(self,theMaze):
        # run through the maze a number of times and find the ave score
        totScore=0
        for __ in range(CC.how_many_chances_for_score):
            self.reset_pos()
            for _ in range(CC.dna_length):
                if self.aliveTF==True:
                    self.stepOnce(theMaze)
            totScore += theMaze.getLocalScore(self.x,self.y)

        return totScore / CC.how_many_chances_for_score
    
    def showPos(self):
        return self.x, self.y
    
class all_zots():
    def __init__(self,theMaze):
        self.theMaze = theMaze
        self.startx, self.starty = theMaze.get_startXY()
        self.number_of_mutations_per_parent = theMaze.number_of_mutations_per_parent
        self.zotList = [zot(self.startx,self.starty,number_of_mutations_per_parent=self.number_of_mutations_per_parent) for _ in range(CC.how_many_zots)]

    def score_population(self,gen,doSHOW=True):
        self.scoreList = np.array(
            [self.zotList[i].ran_by_DNA(self.theMaze) for i in range(CC.how_many_zots)])

        #with Parallel(n_jobs=CC.numCPUs, verbose=False) as par:
        #    temp = par(delayed(self.do_wrapper)(i) for i in range(CC.how_many_zots))
        #self.scoreList= np.array(temp)

        self.indList = np.argsort(self.scoreList)
        bestScore = self.scoreList[self.indList[0]]
        if (gen%100) == 0 and doSHOW==True:
            self.show_best() 
        return self.scoreList, bestScore
    
    def get_best(self):
        # return best zot and best score of that zot
        return self.zotList[self.indList[0]],self.scoreList[self.indList[0]]
    
    def show_best(self,ax=None):
        bestZot,bestScore = self.get_best()
        self.theMaze.drawMaze_andZot(bestZot,ax=ax)
        print(f'Best score ={bestScore}, Mean score={np.nanmean(self.scoreList)}, Max={self.scoreList[self.indList[-1]]}')

    def do_wrapper(self,i):
        return self.zotList[i].ran_by_DNA(self.theMaze)
    
    def choose_mates(self):
        x = np.arange(0,CC.how_many_zots)
        y = np.cumsum(1/((x+1)**CC.population_rule))
        y /= y[-1]
        p_vals = np.random.random(size=2*CC.how_many_zots)
        matelist = np.zeros(2*CC.how_many_zots)
        for i,thisp in enumerate(p_vals):
            thisind = np.argmin(abs(thisp-y))
            matelist[i] = self.indList[x[thisind]]

        # organize so that moms and dads are next to each other
        matelist = np.reshape(matelist,(2,CC.how_many_zots)).astype(int)
        # now build next gem
        new_zot_generation = []
        for new_zot_ind in range(CC.how_many_zots):
            # note, each zot has 2 chromosomes. They donate one for kid.
            momDNA = self.zotList[matelist[0,new_zot_ind]].offer_dna()
            dadDNA = self.zotList[matelist[1,new_zot_ind]].offer_dna()
            myDNA = np.stack([momDNA,dadDNA],axis=0)
            new_zot = zot(self.startx, self.starty, randomDNA=False, dnaData=myDNA)
            new_zot_generation.append(new_zot)
        
        self.zotList = new_zot_generation





def figure_out_value_of_population_rule(this_rule):
    x = np.arange(0,CC.how_many_zots)
    y = np.cumsum(1/(x+1)**this_rule)
    y /= y[-1]
    plt.plot(x,y)
    frac = x[np.argmin(abs(y-0.5))] / CC.how_many_zots
    # this is what fraction of the population 50% of mates will come from
    print(frac)