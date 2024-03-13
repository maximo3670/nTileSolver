"""
Created on Thu Feb 29 10:57:23 2024

Author: Max Neil

Version 1.4

Last Updated: Mar 05 2024 

Changes: Added comments

This script is to determine a solution to the N-Tile Problem by using an IDDFS
algorithm


Output:
Seed,Case number,Case start state,Solution found,Number of moves,Number of nodes opened,Computing time
574,1,"[1, 0, [[8, 2, 3], [0, 5, 1], [6, 7, 4]]]",1,15,22433,0.08135747909545898
574,2,"[1, 1, [[2, 3, 7], [1, 0, 4], [6, 5, 8]]]",1,18,127168,0.4992687702178955
574,3,"[0, 1, [[5, 0, 3], [7, 8, 6], [1, 2, 4]]]",1,21,717749,2.87959885597229
574,4,"[2, 1, [[7, 4, 5], [6, 2, 3], [1, 0, 8]]]",1,21,963714,3.86483097076416
574,5,"[1, 2, [[7, 1, 5], [8, 3, 0], [4, 2, 6]]]",1,23,2391029,9.531949520111084
574,6,"[1, 0, [[1, 7, 6], [0, 5, 3], [8, 2, 4]]]",1,19,187106,0.7477054595947266
574,7,"[1, 2, [[1, 6, 4], [8, 2, 0], [3, 5, 7]]]",0,0,77433285,363.7311851978302
574,8,"[2, 0, [[8, 2, 7], [3, 5, 6], [0, 4, 1]]]",1,26,10972154,44.466230154037476
574,9,"[0, 2, [[2, 5, 0], [8, 3, 4], [7, 1, 6]]]",1,16,45662,0.17453432083129883
574,10,"[0, 2, [[2, 7, 0], [8, 1, 3], [4, 6, 5]]]",0,0,66495184,311.56998777389526

"""

import random
import copy
import time
import csv


#Seed for random functions
seed = 574
random.seed(seed)

#Template state and expected final state
template_state = [2, 2, [[8, 7, 6], [5, 4, 3], [2, 1, 0]]]
goal_state = [1, 1, [[1, 2, 3], [8, 0, 4], [7, 6, 5]]]

#Setting headers for the CSV file output
headers = ["Seed", "Case number", "Case start state", "Solution found", "Number of moves", "Number of nodes opened", "Computing time"]
results = []

"""
write_to_csv()

parameters:
    data: data to put into the csv
    
returns:
    void

Description:
    The function is to write the results into the CSV file.
"""
def write_to_csv(data):
    with open('../Output/IDDFS_output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        writer.writerows(data)
        
"""
flatten()

parameters:
    grid: A two dimensional array
    
retuns:
    A one dimensional array
    
Description:    
    The purpose of this function is transform a 2d array into a 1d array.
    This is called before shuffling the grid because the random.shuffle
    function will change the structure of the array when shuffling.
   
"""
def flatten(grid):
    return [tile for row in grid for tile in row]



"""
reshape()
  
parameters
    flat_list: A one dimensional array
    shape: the desired shape to transform the array into
    
returns
    A multi dimensional array
    
Description:    
    This function is to convert the 1d array back into the 3x3 grid which
    the N-Tile problem requires  
"""
def reshape(flat_list, shape):
    return [flat_list[i:i+shape[1]] for i in range(0, len(flat_list), shape[1])]



"""
generating_start_states()
   
Parameters:
   num_states: the number of random shuffles to do
   
Returns:
    start_states: An array of start states
    
Description:
    The purpose of this function is to generate random start states by
    shuffling the template state.
"""
def generating_start_states(state, num_states):
    start_states = [] 
    for _ in range(num_states):
        # Creates a copy of the template
        template_copy = copy.deepcopy(state)
        
        # Flattens into a 1d array
        # Also removes the "0" position as this is going to change
        flattened_state = flatten(template_copy[2])
        
        # Shuffles the array
        random.shuffle(flattened_state)
        
        # Converts back into 3x3 grid
        shuffled_state = reshape(flattened_state, (len(template_copy[2]), len(template_copy[2][0])))
        
        # Find the location of the blank tile (0) in the reshaped grid
        i_blank, j_blank = find_blank_position(shuffled_state)
        
        #Append the position of the blank peice to the state
        shuffled_state = [i_blank, j_blank, shuffled_state]
        
        # Appends to start_states
        start_states.append(shuffled_state)
    return start_states

"""
find_blank_position()

parameters:
    grid: a 2 dimensional array

returns:
    i, j: position of the "0" in the array
    -1 ,-1: there is no "0" in array
    
Description:
    This function is to find the position of the blank tile after shuffling.
    It loops through a two dimensional array and stops when a "0" has been 
    found.
"""
def find_blank_position(grid):
    for i in range(len(grid)):
        for j in range(len(grid)):
            if grid[i][j] == 0:
                return i, j
    # Blank tile not found so returns -1, -1
    return -1, -1

"""
move()

parameters:
    state: current state of the tiles
    
Returns:
    A generator yielding the next possible state after moving the blank tile.

Description:
    This function generates all possible states that can be obtained by moving 
    the blank tile "0" in the grid. It yields the next possible state 
    after moving the blank tile in each direction.
    
    
THIS PIECE CODE IS NOT MY OWN AND HAS BEEN TAKEN FROM THE LECTURE SLIDES AS PER 
THE BRIEF STATES
"""
def move(state): 
  [i,j,grid]=state
  n = len(grid)
  for pos in move_blank(i,j,n):
    i1,j1 = pos
    grid[i][j], grid[i1][j1] = grid[i1][j1], grid[i][j]
    yield [i1,j1,grid]
    grid[i][j], grid[i1][j1] = grid[i1][j1], grid[i][j]
    
    
"""
move_blank()

Parameters:
    i: row index of the current position of the blank tile.
    j: column index of the current position of the blank tile.
    n: length of the grid.

Returns:
    A generator yielding the positions of adjacent tiles where the blank 
    tile can be moved.

Description:
    This function generates the positions of adjacent tiles where the blank 
    tile can be moved in the grid. It yields the positions of 
    adjacent tiles in all four directions: up, down, left, and right. 

THIS PIECE CODE IS NOT MY OWN AND HAS BEEN TAKEN FROM THE LECTURE SLIDES AS PER 
THE BRIEF STATES
"""
def move_blank(i,j,n): 
  if i+1 < n:
    yield (i+1,j)
  if i-1 >= 0:
    yield (i-1,j)
  if j+1 < n:
    yield (i,j+1)
  if j-1 >= 0:
    yield (i,j-1)


"""
dfs()

Parameters:
    state: current state of the tiles.
    visited_states: set containing the visited states.
    depth_limit: maximum depth during the depth-first search.

Returns:
    solution: Either True or False if a solution has been found 
    moves_counter: Number of moves to get a solution

Description:
    This function performs a recursive depth-first search to explore possible 
    states of the puzzle until it reaches the final state or the depth limit.
    A set of visited states is used to avoid checking the same states twice.
    It counts the number of moves made to reach the final state and the 
    number of nodes explored during the search.
"""
def dfs(state, depth_limit, visited_states):
    global node_counter
    
    # Converting the state into a tuple so it can be used in a 
    # dictionary. normal lists cannot be hashed so the conversion is necessary
    state_tuple = tuple(map(tuple, state[2]))  
    
    # This is to avoid checking the same state twice as it is not 
    # computationaly efficient. It also tracks the depth limit.
    if state_tuple in visited_states or depth_limit <= 0:
        return False, 0
    
    #Adding the currrent state to visited
    visited_states.add(state_tuple)  
    
    #Increasing node count
    node_counter += 1  

    # Check if the current state is the goal state
    if state == goal_state:
        return True, 0  

    # Searching all possible states from the current state
    for next_state in move(state):
        #Searching a level deeper
        solution_found, moves = dfs(next_state, depth_limit - 1, visited_states.copy())
        
        #If a solution is found then backtrack. Incriment moves by +1 as it
        # backtracks
        if solution_found:
            return True, moves + 1  

    return False, 0

"""
iddfs()

Parameters:
    start_state: The initial state of the tiles.
    max_depth: The maximum depth to explore.

Returns:
    Result: either 1 or 0  if a solution has been found
    moves_counter: Number of moves to solution
    node_counter: Number of nodes checked during search

Description:
    This function implements the Iterative Deepening Depth-First Search 
    algorithm to find a solution to the puzzle. 
    
    It does a series of DFS() at different levels of depth which increase by 1 
    until the max_depth is reached. It maintains a count of the number of nodes
    explored througout the search which is cummalative to the dfs() node count.
"""
def iddfs(start_state, max_depth):
    global node_counter
    node_counter = 0

    #Looping through each depth
    for depth_limit in range(1, max_depth + 1):
        visited_states = set()
        
        #Calling the search algorithm
        solution_found, moves = dfs(start_state, depth_limit, visited_states)
        
        #returning if a solution is found or not
        if solution_found:
            return 1, moves, node_counter
    return 0, 0, node_counter

"""

solve_puzzle()

parameters:
    start_state: State to solve
    goal_state: The goal for the start state to get to
    
returns:
    state_copy: copy of the start state
    solution: 1 or 0 whether a solution was found
    moves: number of moves for the solution
    nodes: number of nodes searched
    end_time: time taken to run the algorithm
    
Description:
    This function is to take a state and a goal state and compute attributes
    from it such as whether it is solvable, number of moves until solved etc...

"""
def solve_puzzle(start_state, goal_state, case_num):
    # Creating a copy as passed by reference.
    state_copy = copy.deepcopy(start_state)
    
    #Time the solution started
    start_time = time.time()
    
    #Running the IDDFS algorithm
    solution, moves, nodes = iddfs(state, 30)
    
    #Time it took to compute
    end_time = time.time() - start_time
    
    return (seed, case_num, state_copy, solution,moves,nodes,end_time)
    

    
#Generating 10 random start states
start_states = generating_start_states(template_state, 10)

#Looping through the start states. Also keeping track of the case number
for case_num, state in enumerate(start_states, 1):
    
    #Sending the state to be solved
    result= solve_puzzle(state, goal_state, case_num)
    
    #Adding them to a list to be ouputted to CSV
    results.append(result)

#Writing into the CSV file
write_to_csv(results)





    