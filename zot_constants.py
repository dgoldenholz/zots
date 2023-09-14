import numpy as np

how_many_zots = 300
dna_length = 300
letter_up = 0
letter_down = 1
letter_left = 2
letter_right = 3
protiens = np.array(    [[ 0,-1],
                        [ 0, 1],
                        [-1, 0],
                        [ 1, 0]])
how_many_chances_for_score = 1
population_rule = 1.2       # if 0.6, about 50 of mates come from upper 20%, 0.7 -->13.7%, 0.8 --> 8%, 1-->6% 1.2 -->3%
max_crossovers = 5
prob_One_Parent_mutates = 1
#number_of_mutations_per_parent = int(dna_length*0.01)
number_of_mutations_per_parent = 2
max_generations = 20000
numCPUs=9