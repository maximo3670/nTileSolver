"""
Created on Tue Mar  5 14:56:03 2024

Author: Max Neil

Version 1.5

Last Updated: 12/03/2024

Changes: Commented code

Description:
    The start of this code is the same as "IDDFS.py", it reuses the code to
    generate the random start positions.
    
    This is an implementation of an IDA* algorithm to solve the 8-tile puzzle.
    It uses the manhattan heuristic function to calculate the best next node.
    
    
Output:
Seed,Case number,Case start state,Solution found,Number of moves,Number of nodes opened,Computing time
574,1,"[1, 0, [[8, 2, 3], [0, 5, 1], [6, 7, 4]]]",1,15,299,0.01237630844116211
574,2,"[1, 1, [[2, 3, 7], [1, 0, 4], [6, 5, 8]]]",1,18,395,0.011422872543334961
574,3,"[0, 1, [[5, 0, 3], [7, 8, 6], [1, 2, 4]]]",1,21,1234,0.036261796951293945
574,4,"[2, 1, [[7, 4, 5], [6, 2, 3], [1, 0, 8]]]",1,21,1125,0.043390750885009766
574,5,"[1, 2, [[7, 1, 5], [8, 3, 0], [4, 2, 6]]]",1,23,7016,0.25040411949157715
574,6,"[1, 0, [[1, 7, 6], [0, 5, 3], [8, 2, 4]]]",1,19,424,0.013737678527832031
574,7,"[1, 2, [[1, 6, 4], [8, 2, 0], [3, 5, 7]]]",0,0,275623,5.1623146533966064
574,8,"[2, 0, [[8, 2, 7], [3, 5, 6], [0, 4, 1]]]",1,26,5652,0.09594178199768066
574,9,"[0, 2, [[2, 5, 0], [8, 3, 4], [7, 1, 6]]]",1,16,513,0.009432315826416016
574,10,"[0, 2, [[2, 7, 0], [8, 1, 3], [4, 6, 5]]]",0,0,489703,8.202016592025757

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
    with open('../Output/IDAstar_output.csv', mode='w', newline='') as file:
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
        i_blank, j_blank = find_position(shuffled_state, 0)
        
        #Append the position of the blank peice to the state
        shuffled_state = [i_blank, j_blank, shuffled_state]
        
        # Appends to start_states
        start_states.append(shuffled_state)
    return start_states

"""
find_position()

parameters:
    grid: a 2 dimensional array
    value: the value being searched for

returns:
    i, j: position in the array
    -1 ,-1: there is no value in array
    
Description:
    This function is to find the position of any tile in te grid.
    It loops through a two dimensional array and stops when a the value has been 
    found.
"""
def find_position(grid, value):  
    for i in range(len(grid)):
        for j in range(len(grid)):
            if grid[i][j] == value:
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

state_to_tuple()

parameters:
    state: the state to convert into a tuple
    
returns:
    the state transformed into a tuple
    
Description:
    The purpose of this function is to convert states into a tuple format.
    The reason for doing this is because within the IDA* search algorithm
    I am storing states in a set. List type (Which is what a state is) cannot
    be stored in sets because they are unhashable data types. This is why the
    conversion is made.

"""
def state_to_tuple(state):
    return tuple(map(tuple, state[2]))


"""

manhattan_distance()

parameters:
    state: The state to be checked
    goal_state: The goal state for the current state to reach
    
returns:
    distance: The distance from the current state to the goal state
    
Description:
    This is a heuristic funciton to determine how far from being solved the
    current state is. It gives estimates on the best route for the search 
    algorithm to take.

"""
def manhattan_distance(state, goal_state):
    distance = 0
    n = len(state[2])
    for value in range(1, (n * n)):  
        x1, y1 = find_position(state[2], value)
        x2, y2 = find_position(goal_state[2], value)
        distance += abs(x1 - x2) + abs(y1 - y2)
    return distance

"""
IDA_star()

parameters:
    start_state: The initial state / or current state
    goal_state: The desired state to reach
    depth_limit: Maximum depth the algorithm is allowed to search
    
returns:
    solution: Either 1 or 0 depending if a solution has been found
    move_count: The number of moves to complete the puzzle
    node_count: The number of nodes checked to find the solution

Description:
    This is an Iterative Deeping A* algorithm to find the best solution to the
    n-tile problem. It creates the initial threshold for the algorithm and the
    path it will take.
    
    The process on how the algorithm was made can be found here:
    "https://www.geeksforgeeks.org/iterative-deepening-a-algorithm-ida-artificial-intelligence/"
"""
def IDA_star(start_state, goal_state, depth_limit):

    #Initial threshold of the algorithm
    threshold = manhattan_distance(start_state, goal_state)
    
    visited_states = set()
    
    #setting first node in path to the start_state
    path = [start_state]  
    node_count = 0
    
    #Loop to update the threshold when a solution is not found
    #There is also a maximum threshold limit. The threshold is different to the
    #   maximum depth however this is added because unsolvable solutions will 
    #   increase the depth to no limit which is a waste of computational power.
    #When changed to "while True" the algorithm searched ~979134326 nodes when
    #   there is no solution. This just helps push it to compute faster.
    while threshold <= depth_limit:

        #Calling the search() function to look for a solution
        min_threshold, solution_found, node_count = search(path, 0, threshold, visited_states, goal_state, node_count, depth_limit)
        
        if solution_found:
            #Solution found, return
            return 1, len(path) - 1, node_count  
        
        if min_threshold == float('inf'):
            #No solution found. return
            return 0, 0, node_count  
        
        #Update the threshold to the lowest F-score from the search
        threshold = min_threshold
        
    return 0, 0, node_count

"""
search()

parameters:
    path: The path taken to the solution
    g: The cost from initial node to current node. In this case it is the depth
    threshold: The maxium F-Score for the search algorithm to check a branch
    visited_states: Contains states already visited to avoid checking multiple times
    goal_state:The desired finish state the algorithm is looking for
    depth_limit: Maximum depth the algorithm is allowed to search
        
returns:
    threshold: The maxium F-Score for the search algorithm to check a branch
    solution: True or False whether a solution has been found =
    node_count: The number of nodes visited
        
Description:
    This is the search function for the IDA* algorithm. It is a recursive
    alorithm which looks for an optimal soluton based on the F-Score. The F-score
    determines which is the best path to look down with the lowest score being
    the best.
    
    g = The cost from the initial node to the current node.
    h = Heuristic estimated cost from the current node to the goal state.
    
    f = g + h = Total cost evaluation function.

    
"""
def search(path, g, threshold, visited_states, goal_state, node_count, depth_limit):
    
    # Checking if the depth limit has been exceeded.
    # If so backtrack and return
    if g > depth_limit:
        return float('inf'), False, node_count
    
    # Setting the current state to the last element in the path
    current_state = path[-1]
    
    # f = g + h calcualtion
    h = manhattan_distance(current_state, goal_state)
    f = g + h
    
    # If threshold is exceeded return
    if f > threshold:
        return f, False, node_count  

    # Check to see if goal state has been found, return.
    if current_state[2] == goal_state[2]:  
        return f, True, node_count  

    # Resetting the minimum threshold
    min_threshold = float('inf')
    
    # All possible moves from the current state
    for next_state in move(current_state):
        
        # Check it hasn't been visited yet. Prevents checking itself.
        if state_to_tuple(next_state) not in visited_states:
            
            # Adds to visited and path
            visited_states.add(state_to_tuple(next_state))
            path.append(next_state)

            # Searches a level deeper in the path
            new_threshold, solution_found, new_node_count = search(path, g + 1, threshold, visited_states, goal_state, node_count + 1, depth_limit)
            
            # If a solution was found back track.
            if solution_found:
                return new_threshold, True, new_node_count  
            
            # Else remove from the path as it was not a successful search
            path.pop()
            visited_states.discard(state_to_tuple(next_state))
            
            # Update the threshold
            if new_threshold < min_threshold:
                min_threshold = new_threshold
            
            # Increase the node count
            node_count = new_node_count  

    # return the minimum threshold value to update
    return min_threshold, False, node_count


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
    
    #Creating a copy so the state isn't changed
    start_state_copy= copy.deepcopy(start_state)
    
    #getting start time
    start_time = time.time()
    
    #Running the IDA* algorithm
    solution, moves, nodes = IDA_star(start_state, goal_state, 30)
    
    # Calcualting the time taken
    end_time = time.time() - start_time

    return seed, case_num, start_state_copy, solution, moves, nodes, end_time



#Generating 10 random start states

start_states = generating_start_states(template_state, 10)

for case_num, state in enumerate(start_states, 1):

    result = solve_puzzle(state, goal_state, case_num)
    
    results.append(result)
    

#Writing to the output file
write_to_csv(results)

        
