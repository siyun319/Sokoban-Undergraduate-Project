import board as bd
import collections
import heapq
from hungarian_algorithm import algorithm
from munkres import Munkres
import deadlock_detection as dd
import copy
import time
import heapdict
m = Munkres()

heuristic_hash = dict()
hash = dict()

global matrix
global free_square


action = {'r': (1, 0),
          'l': (-1, 0),
          'd': (0, 1),
          'u': (0, -1),
          'R': (1, 0),
          'L': (-1, 0),
          'D': (0, 1),
          'U': (0, -1)}

global number_nodes_explored


# https://towardsdatascience.com/priority-queues-in-python-3baf0bac2097
class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.count = 0

    def is_empty(self):
        return not self.queue

    def push(self, value, priority):
        heapq.heappush(self.queue, (priority, id(value), value))
        self.count += 1

    def pop(self):
        return heapq.heappop(self.queue)[-1]

    def decrease_key(self, value, priority):
        self.count -= 1
        heapq.heapreplace(self.queue, (priority, id(value), value))






def depthFirstSearch(board):
    initial_state = bd.State(board)
    frontier = []
    frontier.append(initial_state)

    # avoid to search the same state multiple times
    exploredSet = set()
    parent = dict()
    parent[initial_state.toString()] = [None, None]
    # actions = []
    number_nodes_explored = 0

    while frontier:
        state = frontier.pop()
        valid_moves = state.get_valid_moves()
        if state.toString() not in exploredSet:
            exploredSet.add(state.toString())
            for the_move in valid_moves:
                new_state = state.update_state(action[the_move])
                if (new_state.toString() in exploredSet):
                    continue
                number_nodes_explored = number_nodes_explored + 1
                frontier.append(new_state)
                # actions.append(the_move)
                parent[new_state.toString()] = [state.toString(), the_move]
                # print(new_state.toString())
                if (new_state.is_end()):
                    # print("Game has been solved.")
                    # print(number_nodes_explored)
                    return get_Solution(parent, initial_state.toString(), new_state.toString())
    return False


def breathFirstSearch(board):
    initial_state = bd.State(board)
    frontier = []
    frontier.append(initial_state)

    # avoid to search the same state multiple times
    exploredSet = set()
    parent = dict()
    parent[initial_state.toString()] = [None, None]
    # actions = []
    number_nodes_explored = 0

    while frontier:
        state = frontier.pop(0)
        valid_moves = state.get_valid_moves()
        if state.toString() not in exploredSet:
            exploredSet.add(state.toString())
            for the_move in valid_moves:
                new_state = state.update_state(action[the_move])
                if (new_state.toString() in exploredSet):
                    continue
                number_nodes_explored = number_nodes_explored + 1
                frontier.append(new_state)
                # actions.append(the_move)
                parent[new_state.toString()] = [state.toString(), the_move]
                # print(new_state.toString())
                if (new_state.is_end()):
                    # print("Game has been solved.")
                    # print(number_nodes_explored)
                    return get_Solution(parent, initial_state.toString(), new_state.toString())
    return False



############# Those two do not consider the cost of the parent!!!!!!!#######
# Give state when we can push a box lower cost
# Encourage to push box rather than moving around
def get_single_cost_encourages_pushing(state, if_pushing):
    cost = 2
    if if_pushing:
        return 0
    return cost

# Give state when we can push a box lower cost
# Encourage to push box rather than moving around
def get_single_cost_encourages_states_with_pushing(state):
    valid_moves = state.get_valid_moves()
    cost = 2
    for move in valid_moves:
        if move.isupper():
            return 0
    return cost


########################################

def get_cost_encourages_pushing(state, parent, cost_dict, is_pushing):
    cost = 0
    if parent not in cost_dict.keys():
        cost = cost_dict[state.toString()] = get_single_cost_encourages_pushing(state, is_pushing)
    else:
        cost = cost_dict[state.toString()] = get_single_cost_encourages_pushing(state, is_pushing) + cost_dict[parent]
    return cost

def get_cost_record_depth_enrougas_potential_pushing(state, parent, cost_dict):
    cost = 0
    if parent not in cost_dict.keys():
        cost = cost_dict[state.toString()] = get_single_cost_encourages_states_with_pushing(state)
    else:
        cost = cost_dict[state.toString()] = get_single_cost_encourages_states_with_pushing(state) + cost_dict[parent]
    return cost


#############
def get_cost(new_state, parent, is_pushing, cost_dict, cost_function):

    if cost_function == 'potential_pushing':
        current_cost = get_cost_record_depth_enrougas_potential_pushing(new_state, parent, cost_dict)
    elif cost_function == 'encourages_pushing':
        current_cost = get_cost_encourages_pushing(new_state, parent, cost_dict, is_pushing)
    else:
        current_cost = get_cost_encourages_pushing(new_state, parent, cost_dict, is_pushing)
    return current_cost


def not_in_goals_cost(state):
    map = state.get_map2()
    cost = len(state.goals)
    for goal in state.goals:
        if map[goal[1]][goal[0]] == 'X':
            cost -= 1
    return cost


# Change if the pushing direction is towards some goals
def get_cost_pushing_towards_goal(state):
    valid_moves = state.get_valid_moves()
    cost = 2
    for move in valid_moves:
        if move.isupper():
            player = state.player[0]
            box = (player[0] + action[move][0], player[1] + action[move][1])
            new_box = (player[0] + 2 * action[move][0], player[1] + 2*action[move][1])
            for goal in state.goals:
                if goal[0] == new_box[0] or goal[1] == new_box[1]:
                    if get_Manhattan_distance(new_box, goal) < get_Manhattan_distance(box, goal):
                        return 0
            return 1
    return cost

def get_Manhattan_distance(pointx, pointy):
    return abs(pointx[0] - pointy[0]) + abs(pointx[1] - pointy[1])

def generate_sub_map(the_map, box, goal, player):
    map = [x[:] for x in the_map]
    if box != goal:
        map[box[1]][box[0]] = 'B'
        map[goal[1]][goal[0]] = '.'
    else:
        map[box[1]][box[0]] = 'X'

    if player == goal:
        map[goal[1]][goal[0]] = '+'
    if player == box:
        if map[box[1] - 1][box[0]] == ' ':
            map[box[1] - 1][box[0]] = '&'

        elif map[box[1] + 1][box[0]] == ' ':
            map[box[1] + 1][box[0]] = '&'

        elif map[box[1]][box[0] + 1] == ' ':
            map[box[1]][box[0] + 1] = '&'

        elif map[box[1]][box[0] - 1] == ' ':
            map[box[1]][box[0] - 1] = '&'

    return map

def pre_compute_heuristic(board, simple_deadlocks):
    sorted_boxes = sorted(board.get_inital_boxes())
    sorted_target = sorted(board.get_inital_goals())
    global free_square
    free_square = []
    free_square = sorted(board.get_free_square())


    initial_player = board.get_inital_player()[0]


    i = 0
    global matrix
    matrix = []
    empty_map_with_only_player = board.get_map_only_walls()
    for each_free_square in free_square:
        matrix.append([])
        if each_free_square in simple_deadlocks:
            for each_goal in sorted_target:
                 matrix[i].append(-1)
            i += 1
            continue
        for each_goal in sorted_target:
            sub_level = generate_sub_map(empty_map_with_only_player, each_free_square, each_goal, initial_player)
            the_board = bd.Board(map = sub_level)
            sub_sol = breathFirstSearch(the_board)
            if sub_sol == False:
                matrix[i].append(-1)
            else:
                first_push = 0
                cost = len(sub_sol)
                for each_action in sub_sol:
                    if each_action.isupper():
                        cost = cost - first_push
                        break
                    else:
                        first_push += 1

                matrix[i].append(cost)
        i += 1
    return matrix

# Better than just using Manhattan distance
# Use this sololy not good!!
# call this with linear condflicts const!!
def get_heuristic_min_match(state):
    cost = 0
    cost_matrix = state.get_target_box_matrix()
    indexes = m.compute(cost_matrix)
    for row, column in indexes:
        value = cost_matrix[row][column]
        cost += value
    return cost


# Doesn't work!! Hard to find this cost since it's kind of solving a subproblem already!
# Complexity becomes really big compared to before
def backout_conflicts_heuristic(state, map_only_walls):
    total_cost = 0
    for box in state.boxes:
        for goal in state.goals:
            current_map = copy.deepcopy(map_only_walls)
            current_map[box[1]][box[0]] = 'B'
            current_map[goal[1]][goal[0]] = '.'
            # print("cu", current_map)
            if current_map[box[1] + 1][box[0]] == ' ':
                current_map[box[1] + 1][box[0]] = '&'
            elif current_map[box[1] - 1][box[0]] == ' ':
                current_map[box[1] - 1][box[0]] = '&'
            elif current_map[box[1]][box[0] + 1] == ' ':
                current_map[box[1]][box[0] + 1] = '&'
            elif current_map[box[1]][box[0] - 1] == ' ':
                current_map[box[1]][box[0] - 1] = '&'
            mini_board = bd.Board(map = current_map)
            sol = depthFirstSearch(mini_board)
            if sol == False:
                current_cost = 100
            else:
                # print("lalalalal")
                current_cost = len(sol)
            total_cost = total_cost + current_cost
    return total_cost

# Gives better solution, but excution time longer.
def get_heuristic2(state):
    cost = 0
    for i in range(len(state.boxes)):
        cost += get_Manhattan_distance(state.boxes[i], state.goals[i])
    return cost

# Maybe better?
def linear_conficts_with_manhattan(state, box):
    linear_conficts_costs = 0
    if box == None:
        return get_heuristic_min_match(state)
    for goal in state.goals:
        if goal[0] == box[0]:
            sign = 0
            if goal[1] - box[1] >= 0:
                sign = 1
            else:
                sign = -1
            if (box[0], box[1] + sign) in state.boxes:
                linear_conficts_costs += 1
        if goal[1] == box[1]:
            sign = 0
            if goal[0] - box[0] >= 0:
                sign = 1
            else:
                sign = -1
            if (box[0] + sign, box[1]) in state.boxes:
                linear_conficts_costs += 1
    return get_heuristic_min_match(state) + linear_conficts_costs

def simple_linear_conficts(state, box):
    linear_conficts_costs = 0
    if box == None:
        return just_manhattan(state)
    for goal in state.goals:
        if goal[0] == box[0]:
            sign = 0
            if goal[1] - box[1] >= 0:
                sign = 1
            else:
                sign = -1
            if (box[0], box[1] + sign) in state.boxes:
                linear_conficts_costs += 1
        if goal[1] == box[1]:
            sign = 0
            if goal[0] - box[0] >= 0:
                sign = 1
            else:
                sign = -1
            if (box[0] + sign, box[1]) in state.boxes:
                linear_conficts_costs += 1
    total_cost = linear_conficts_costs +  just_manhattan(state)
    return total_cost

def linear_conficts_with_pre_computed_heuristic(state, box):
    linear_conficts_costs = 0
    if box == None:
        return get_pre_conmuted_heuristic(state)
    for goal in state.goals:
        if goal[0] == box[0]:
            sign = 0
            if goal[1] - box[1] >= 0:
                sign = 1
            else:
                sign = -1
            if (box[0], box[1] + sign) in state.boxes:
                linear_conficts_costs += 1
        if goal[1] == box[1]:
            sign = 0
            if goal[0] - box[0] >= 0:
                sign = 1
            else:
                sign = -1
            if (box[0] + sign, box[1]) in state.boxes:
                linear_conficts_costs += 1
    return get_pre_conmuted_heuristic(state) + linear_conficts_costs * 5

pre_conmuted_heuristic = dict()

def get_pre_conmuted_heuristic(state):
    if str(sorted(state.boxes)) in pre_conmuted_heuristic.keys():
        return pre_conmuted_heuristic[str(sorted(state.boxes))]
    boxes = state.boxes
    current_matrix = []
    for box in boxes:
        index = free_square.index(box)
        current_matrix.append(matrix[index])
    indexes = m.compute(current_matrix)
    # print('index------')
    # print(indexes)
    cost = 0
    for row, column in indexes:
        value = current_matrix[row][column]
        cost += value
    pre_conmuted_heuristic[str(sorted(state.boxes))] = cost
    return cost




def just_manhattan(state):
    boxes = sorted(state.boxes)
    goals = sorted(state.goals)

    total_distance = 0

    i = 0
    for box in boxes:
        total_distance = total_distance +  get_Manhattan_distance(box, goals[i])
        i = i + 1
    return total_distance

def uniformCostSearch(board, parameter_dict):
    initial_state = bd.State(board)
    frontier = PriorityQueue()
    frontier.push(initial_state, 0)
    exploredSet = set()
    parent = dict()
    parent[initial_state.toString()] = [None, None]
    # actions = []
    number_nodes_explored = 0
    cost_dict = dict()
    cost_function = parameter_dict['cost']

    # potential_states = PriorityQueue()
    while frontier:
        if frontier.is_empty():
            return False
        state = frontier.pop()
        valid_moves = state.get_valid_moves()
        if state.toString() not in exploredSet:
            exploredSet.add(state.toString())
            for the_move in valid_moves:
                new_state = state.update_state(action[the_move])
                if (new_state.toString() in exploredSet):
                    continue
                number_nodes_explored = number_nodes_explored + 1
                is_pushing = the_move.isupper()
                current_cost = get_cost(new_state, state, is_pushing,cost_dict, cost_function) + not_in_goals_cost(new_state)
                frontier.push(new_state, current_cost)
                # potential_states.push(new_state, current_cost)
                parent[new_state.toString()] = [state.toString(), the_move]
                if (new_state.is_end()):
                    print("Game has been solved.")
                    print(number_nodes_explored)
                    return get_Solution(parent, initial_state.toString(), new_state.toString())



# My thought: use dfs, to search the reachable positions.
# If the player is at the boundary of the reachable area, then we need to deal with this state earlier!!
def dectect_non_accessiable_area(the_state):
    initial_state = the_state
    frontier = []
    frontier.append(initial_state)

    # avoid to search the same state multiple times
    exploredSet = set()

    while frontier:
        print(len(frontier))
        state = frontier.pop()
        valid_moves = state.get_valid_moves()
        exploredSet.add(state.player[0])
        for the_move in valid_moves:
            if the_move.islower():
                new_state = state.update_state(action[the_move])
                if new_state.player[0] not in exploredSet:
                    frontier.append(new_state)
            else:
                # We don't push box here
                # Just need to check for accessable area, by walking around
                box_posit = (state.player[0][0] + action[the_move][0], state.player[0][1] + action[the_move][1])
                exploredSet.add(box_posit)


    return bd.all_sqaure - exploredSet



def is_player_next_to_corral(non_accessiable_area,state, map):
    player = state.player[0]

    if (player[0] + 2, player[0]) in non_accessiable_area and (player[0] + 2, player[0]) not in bd.walls and (player[0] + 1, player[0]) in state.boxes:
        return True
    if (player[0] - 2, player[0]) in non_accessiable_area and (player[0] - 2, player[0]) not in bd.walls and (player[0] - 1, player[0]) in state.boxes:
        return True
    if (player[0], player[0] + 2) in non_accessiable_area and (player[0], player[0] + 2) not in bd.walls and (player[0], player[0] + 1) in state.boxes:
        return True
    if (player[0], player[0] - 2) in non_accessiable_area and (player[0], player[0] - 2) not in bd.walls and (player[0], player[0] - 1) in state.boxes:
        return True
    return False


def initiate_class_variable():
    global heuristic_hash
    global matrix
    global free_square
    global pre_conmuted_heuristic
    global hash
    global prority_hash
    heuristic_hash = dict()
    matrix = []
    free_square = []
    pre_conmuted_heuristic = dict()

    prority_hash = dict()

    hash = dict()



def search(board, parameter_dict):
    algorithm_used = parameter_dict['algorithm']
    cost_function  = parameter_dict['cost']
    heuristic_function = parameter_dict['heuristic']
    dd_detection = parameter_dict['dd_detection']
    moving_ordering = parameter_dict['move_ordering']

    if algorithm_used == 'DFS':
        return depthFirstSearch(board)
    elif algorithm_used == 'BFS':
        return breathFirstSearch(board)
    elif algorithm_used == 'UCS':
        return uniformCostSearch(board, parameter_dict)
    elif algorithm_used == 'A_star':
        return aStarSearch_with_changing_parameters(board, parameter_dict)
    elif algorithm_used == 'Greedy':
        return greedy_simple_herustic(board, parameter_dict)
    elif algorithm_used == 'Floor_fill':
        return aStarSearch_simple_herustic_floor_fill(board, parameter_dict)
    else:
        print("Search algorithm is not well defined. Please check your input.")
        return

def aStarSearch_with_changing_parameters(board, parameter_dict):
    initiate_class_variable()
    initial_state = bd.State(board)


    cost_function  = parameter_dict['cost']
    heuristic_function = parameter_dict['heuristic']
    dd_detection = parameter_dict['dd_detection']
    move_ordering = parameter_dict['move_ordering']

    # print("map!!", board.getMap())
    if initial_state.is_end():
        return []
    frontier = heapdict.heapdict()
    the_map = board.getMap()

    needs_recheck_accessiable_dict = dict()

    simple_deadlocks = None

    dd.initialize()
    if dd_detection:
        simple_deadlocks = dd.simple_deadlocks(board.getMap())
        print('simple dd found!')
    # print(simple_deadlocks)
    start_time = time.time()

    if heuristic_function == "pre computed":
        if simple_deadlocks == None:
            simple_deadlocks = dd.simple_deadlocks(board.getMap())
            print('simple dd found!')
        pre_compute_heuristic(board, simple_deadlocks)
        heuristic = linear_conficts_with_pre_computed_heuristic(initial_state, None)
        print('heuristic function finished!')
    else:
        min_match = parameter_dict['min_match']
        linear_conflicts = parameter_dict['linear_conflict']

        if not min_match and not linear_conflicts:
            heuristic = just_manhattan(initial_state)

        if not min_match and linear_conflicts:
            heuristic = simple_linear_conficts(initial_state, None)

        if min_match and not linear_conflicts:
            heuristic = get_heuristic_min_match(initial_state)

        if min_match and linear_conflicts:
            heuristic = linear_conficts_with_manhattan(initial_state, None)


    is_corral_on = False

    frontier[initial_state] = heuristic

    heuristic_hash[str(sorted(initial_state.boxes))] = heuristic

    prority_hash[str(sorted(initial_state.toString()))] = 0 + heuristic
    exploredSet = set()
    parent = dict()
    parent[initial_state.toString()] = [None, None]
    # actions = []
    number_nodes_explored = 0


    cost_dict = dict()

    while frontier:
        if not bool(frontier):
            print("---Can't find a solutiion---")
            print(the_map)
            return False

        state = frontier.popitem()[0]
        # print("stateee", state.toString())
        valid_moves = state.get_valid_moves()
        pushed_box = None
        if state.toString() not in exploredSet:
            exploredSet.add(state.toString())

            # print(is_corral)
            for the_move in valid_moves:
                add_state = True
                new_state = state.update_state(action[the_move])
                new_map = new_state.get_map()
                total_cost = None
                # print(the_map)
                if (new_state.toString() in exploredSet):
                    add_state = False

                if add_state:
                    # Push a box
                    if the_move.isupper():
                        pushed_box = (state.player[0][0] + action[the_move][0] * 2, state.player[0][1] + action[the_move][1] * 2)
                    # for box in new_state.boxes:
                        # Pay attention to those goals!!
                        if dd_detection:
                            if pushed_box in simple_deadlocks and pushed_box not in new_state.goals:
                                add_state = False
                            # #  free_deadlocks already has a hash. No need to use here.
                            elif dd.free_deadlocks([x[:] for x in new_map], pushed_box, simple_deadlocks, new_state.boxes.copy()) and pushed_box not in new_state.goals:
                                add_state = False

                if add_state:
                    # print("adding")
                    number_nodes_explored = number_nodes_explored + 1
                    # current_cost = (get_cost3(new_state, state, cost_dict) + not_in_goals_cost(new_state)) * 2
                    current_cost = get_cost(new_state, state, the_move.isupper(), cost_dict, cost_function)
                    # No influence move (One kind of moving ordering)
                    if the_move.isupper() and the_move in new_state.get_valid_moves():
                        # Otherwise no point to continue push the box.
                        # Move ordering + tunnel maros (Some cases)
                        if move_ordering and  (pushed_box not in state.goals and len(new_state.get_valid_moves()) == 1):
                            print("need further push")
                            old_state = new_state
                            new_state = new_state.update_state(action[the_move])
                            parent[old_state.toString()] = [state.toString(), the_move]
                            total_cost = -1
                            # frontier[move_ordering_state] = 0
                            current_heuristic = 0

                        else:
                            if str(sorted(new_state.boxes)) in heuristic_hash.keys():
                                current_heuristic = heuristic_hash[str(sorted(new_state.boxes))]
                            else:
                                # current_heuristic = linear_conficts_with_pre_computed_heuristic(new_state, pushed_box)
                                if heuristic_function == "pre computed":
                                    current_heuristic = linear_conficts_with_pre_computed_heuristic(new_state, pushed_box)
                                else:
                                    if not min_match and not linear_conflicts:
                                        current_heuristic = just_manhattan(new_state)

                                    if min_match and not linear_conflicts:
                                        current_heuristic = get_heuristic_min_match(new_state)

                                    if min_match and linear_conflicts:
                                        current_heuristic = linear_conficts_with_manhattan(new_state, pushed_box)

                                    if not min_match and linear_conflicts:
                                        current_heuristic = simple_linear_conficts(state, pushed_box)


                    else:
                        # No move ordering
                        if str(sorted(new_state.boxes)) in heuristic_hash.keys():
                            current_heuristic = heuristic_hash[str(sorted(new_state.boxes))]
                        else:
                            if heuristic_function == "pre computed":
                                current_heuristic = linear_conficts_with_pre_computed_heuristic(new_state, pushed_box)
                            else:
                                if not min_match and not linear_conflicts:
                                    current_heuristic = just_manhattan(new_state)

                                if min_match and not linear_conflicts:
                                    current_heuristic = get_heuristic_min_match(new_state)

                                if min_match and linear_conflicts:
                                    current_heuristic = linear_conficts_with_manhattan(new_state, pushed_box)

                                if not min_match and linear_conflicts:
                                    current_heuristic = simple_linear_conficts(state, pushed_box)

                    if total_cost == None:
                        total_cost = current_cost + current_heuristic
                    else:
                        total_cost = 0
                    if new_state.toString() in prority_hash.keys():
                        # print("checking")
                        previvous_cost = prority_hash[new_state.toString()]
                        # print("?")
                        # print("pc", current_heuristic)
                        if total_cost < previvous_cost:
                            # frontier.decrease_key(new_state, total_cost)
                            frontier[new_state] = total_cost
                            # print("dreasing")
                            if total_cost != -1:
                                parent[new_state.toString()] = [state.toString(), the_move]
                            else:
                                parent[new_state.toString()] = [old_state.toString(), the_move]
                            prority_hash[new_state.toString()] = total_cost
                    else:
                        # frontier.push(new_state, total_cost)
                        frontier[new_state] = total_cost
                        prority_hash[new_state.toString()] = total_cost
                        parent[new_state.toString()] = [state.toString(), the_move]
                        # potential_states.push(new_state, current_cost)
                        # parent[new_state.toString()] = [state.toString(), the_move]
                    if (new_state.is_end()):
                        print("Game has been solved.")
                        print(time.time() - start_time)
                        sol = get_Solution(parent, initial_state.toString(), new_state.toString())
                        # print(sol)
                        # print(len(sol))
                        # print(the_map)
                        # print(sum(1 for c in sol if c.isupper()))
                        print("A star with ", parameter_dict, number_nodes_explored)
                        return sol
    return False


# Pre-computed heuristic
def aStarSearch(board):
    initiate_class_variable()
    initial_state = bd.State(board)
    # print("map!!", board.getMap())
    if initial_state.is_end():
        return []
    frontier = heapdict.heapdict()
    the_map = board.getMap()

    needs_recheck_accessiable_dict = dict()

    dd.initialize()
    simple_deadlocks = dd.simple_deadlocks(board.getMap())
    print('simple dd found!')
    # print(simple_deadlocks)
    start_time = time.time()
    pre_compute_heuristic(board, simple_deadlocks)

    print('heuristic function finished!')


    is_corral_on = False

    map_without_boxes = board.get_map_without_boxes()
    heuristic = linear_conficts_with_pre_computed_heuristic(initial_state, None)
    # frontier.push(initial_state, heuristic)
    frontier[initial_state] = heuristic

    heuristic_hash[str(sorted(initial_state.boxes))] = heuristic

    prority_hash[str(sorted(initial_state.toString()))] = 0 + heuristic
    exploredSet = set()
    parent = dict()
    parent[initial_state.toString()] = [None, None]
    # actions = []
    number_nodes_explored = 0

    needs_recheck_accessible_area = True
    accessiable_area = None

    cost_dict = dict()


    # assignment = min_matching_assign(state)
    while frontier:
        if not bool(frontier):
            print("---Can't find a solutiion---")
            print(the_map)
            return False

        state = frontier.popitem()[0]
        # print("stateee", state.toString())
        valid_moves = state.get_valid_moves()
        pushed_box = None
        if state.toString() not in exploredSet:
            exploredSet.add(state.toString())

            if is_corral_on:
                # Try to improve the key by normalizing
                if state.toString() not in needs_recheck_accessiable_dict.keys():
                    accessiable_area = dectect_non_accessiable_area(state)
                    needs_recheck_accessiable_dict[state.toString] = accessiable_area
                else:
                    accessiable_area =  needs_recheck_accessiable_dict[state.toString]
                is_corral = is_player_next_to_corral(accessiable_area, state, the_map)

            # print(is_corral)
            for the_move in valid_moves:
                add_state = True
                new_state = state.update_state(action[the_move])
                new_map = new_state.get_map()
                # print(the_map)
                if (new_state.toString() in exploredSet):
                    add_state = False

                if add_state:
                    # Push a box
                    if the_move.isupper():
                        pushed_box = (state.player[0][0] + action[the_move][0] * 2, state.player[0][1] + action[the_move][1] * 2)
                    # for box in new_state.boxes:
                        # Pay attention to those goals!!
                        if pushed_box in simple_deadlocks and pushed_box not in new_state.goals:
                            add_state = False
                        # #  free_deadlocks already has a hash. No need to use here.
                        elif dd.free_deadlocks([x[:] for x in new_map], pushed_box, simple_deadlocks, new_state.boxes.copy()) and pushed_box not in new_state.goals:
                            add_state = False
                    else:
                        # If corral, we should encourage to push box, rather than move around, so not add the state
                        if is_corral_on:
                            if is_corral:
                                add_state = False

                    if is_corral_on:
                        # Only need to recalculate the accessiable area if then state is produced by pushing some box!
                        if add_state and the_move.islower():
                            needs_recheck_accessiable_dict[new_state.toString] = accessiable_area


                if add_state:
                    # print("adding")
                    number_nodes_explored = number_nodes_explored + 1
                    # current_cost = (get_cost3(new_state, state, cost_dict) + not_in_goals_cost(new_state)) * 2
                    current_cost = get_single_cost_encourages_states_with_pushing(new_state)
                    # No influence move (One kind of moving ordering)
                    if the_move.isupper() and the_move in new_state.get_valid_moves():
                        # Otherwise no point to continue push the box.
                        # Move ordering + tunnel maros (Some cases)
                        if (pushed_box in state.goals and (pushed_box[0] + action[the_move][0], pushed_box[1] + action[the_move][1]) in state.goals) or (pushed_box not in state.goals):
                            current_heuristic = 0
                        else:
                            if str(sorted(new_state.boxes)) in heuristic_hash.keys():
                                current_heuristic = heuristic_hash[str(sorted(new_state.boxes))]
                            else:
                                current_heuristic = linear_conficts_with_pre_computed_heuristic(new_state, pushed_box)


                    else:
                        # No move ordering
                        if str(sorted(new_state.boxes)) in heuristic_hash.keys():
                            current_heuristic = heuristic_hash[str(sorted(new_state.boxes))]
                        else:
                            current_heuristic = linear_conficts_with_pre_computed_heuristic(new_state, pushed_box)
                        # total_cost = linear_conficts_with_manhattan(new_state, pushed_box)

                    total_cost = current_cost + current_heuristic
                    if new_state.toString() in prority_hash.keys():
                        # print("checking")
                        previvous_cost = prority_hash[new_state.toString()]
                        # print("?")
                        # print("pc", current_heuristic)
                        if total_cost < previvous_cost:
                            # frontier.decrease_key(new_state, total_cost)
                            frontier[new_state] = total_cost
                            # print("dreasing")
                            parent[new_state.toString()] = [state.toString(), the_move]
                            prority_hash[new_state.toString()] = total_cost
                    else:
                        # frontier.push(new_state, total_cost)
                        frontier[new_state] = total_cost
                        prority_hash[new_state.toString()] = total_cost
                        parent[new_state.toString()] = [state.toString(), the_move]
                        # potential_states.push(new_state, current_cost)
                        # parent[new_state.toString()] = [state.toString(), the_move]
                    if (new_state.is_end()):
                        print("Game has been solved.")
                        print(time.time() - start_time)
                        sol = get_Solution(parent, initial_state.toString(), new_state.toString())
                        # print(sol)
                        # print(len(sol))
                        # print(the_map)
                        # print(sum(1 for c in sol if c.isupper()))
                        print("A start with pre", number_nodes_explored)
                        return sol


def aStarSearch_simple_herustic(board):
    initiate_class_variable()
    initial_state = bd.State(board)
    if initial_state.is_end():
        return []
    frontier = heapdict.heapdict()
    the_map = board.getMap()

    cost_dict = dict()

    needs_recheck_accessiable_dict = dict()

    dd.initialize()
    # To be modified
    # Simple_deadlocks take too much time for large maps.
    simple_deadlocks = dd.simple_deadlocks(board.getMap())
    print(sorted(simple_deadlocks))
    print('simple dd found!')
    # print(simple_deadlocks)
    start_time = time.time()
    # pre_compute_heuristic(board, simple_deadlocks)

    # print('heuristic function finished!')


    is_corral_on = False

    print(the_map)

    # assignment = min_matching_assign(initial_state)
    # assignment = False
    map_without_boxes = board.get_map_without_boxes()
    # frontier.push(initial_state, (initial_state, map_with_only_walls))
    # frontier.push(initial_state, get_heuristic_min_match(initial_state))
    heuristic = linear_conficts_with_manhattan(initial_state, None)
    frontier[initial_state] = heuristic

    prority_hash = dict()
    heuristic_hash[str(sorted(initial_state.boxes))] = heuristic
    prority_hash[str(sorted(initial_state.toString()))] = 0 + heuristic
    exploredSet = set()
    parent = dict()
    parent[initial_state.toString()] = [None, None]
    # actions = []
    number_nodes_explored = 0

    # heuristic_dict = dict()
    # heuristic_dict[initial_state.getHashValue()] = linear_conficts_with_manhattan(initial_state, None)



    cost_dict = dict()

    needs_recheck_accessible_area = True
    accessiable_area = None

    # assignment = min_matching_assign(state)
    while frontier:
        if not bool(frontier):
            print("---Can't find a solutiion---")
            print(the_map)
            return False

        state = frontier.popitem()[0]
        valid_moves = state.get_valid_moves()
        pushed_box = None
        if state.toString() not in exploredSet:
            exploredSet.add(state.toString())

            if is_corral_on:
                # Try to improve the key by normalizing
                if state.toString not in needs_recheck_accessiable_dict.keys():
                    accessiable_area = dectect_non_accessiable_area(state)
                    needs_recheck_accessiable_dict[state.toString] = accessiable_area
                else:
                    accessiable_area =  needs_recheck_accessiable_dict[state.toString]
                is_corral = is_player_next_to_corral(accessiable_area, state, the_map)

            # print(is_corral)
            for the_move in valid_moves:
                add_state = True
                new_state = state.update_state(action[the_move])
                new_map = new_state.get_map()
                # print(the_map)
                if (new_state.toString() in exploredSet):
                    add_state = False
                # Push a box
                if add_state:
                    if the_move.isupper():
                        pushed_box = (state.player[0][0] + action[the_move][0] * 2, state.player[0][1] + action[the_move][1] * 2)
                    # for box in new_state.boxes:
                        # Pay attention to those goals!!
                        if pushed_box in simple_deadlocks and pushed_box not in new_state.goals:
                            add_state = False
                        #  free_deadlocks already has a hash. No need to use here.
                        elif dd.free_deadlocks([x[:] for x in new_map], pushed_box, simple_deadlocks, new_state.boxes.copy()) and pushed_box not in new_state.goals:
                            add_state = False
                    else:
                        # If corral, we should encourage to push box, rather than move around, so not add the state
                        if is_corral_on:
                            if is_corral:
                                add_state = False

                    if is_corral_on:
                        # Only need to recalculate the accessiable area if then state is produced by pushing some box!
                        if add_state and the_move.islower():
                            needs_recheck_accessiable_dict[new_state.toString] = accessiable_area


                if add_state:
                    number_nodes_explored = number_nodes_explored + 1
                    # current_cost = get_cost(new_state)
                    current_cost = get_cost_record_depth_enrougas_potential_pushing(new_state, state, cost_dict)

                    # No influence move (One kind of moving ordering)
                    if the_move.isupper() and the_move in new_state.get_valid_moves():
                        # Otherwise no point to continue push the box.
                        # Move ordering + tunnel maros (Some cases)
                        if (pushed_box in state.goals and (pushed_box[0] + action[the_move][0], pushed_box[1] + action[the_move][1]) in state.goals) or (pushed_box not in state.goals):
                            current_heuristic = 0
                        else:
                            if str(sorted(new_state.boxes)) in heuristic_hash:
                                current_heuristic = heuristic_hash[str(sorted(new_state.boxes))]
                            else:
                                current_heuristic = linear_conficts_with_manhattan(new_state, pushed_box)
                    else:
                        if str(sorted(new_state.boxes)) in heuristic_hash.keys():
                            current_heuristic = heuristic_hash[str(sorted(new_state.boxes))]
                        else:
                            current_heuristic = linear_conficts_with_manhattan(new_state, pushed_box)
                        # current_heuristic = linear_conficts_with_manhattan(new_state, pushed_box)
                    total_cost = current_cost + current_heuristic

                    if new_state.toString() in prority_hash.keys():
                        # print("checking")
                        previvous_cost = prority_hash[new_state.toString()]
                        # print("?")
                        # print("pc", current_heuristic)
                        if total_cost < previvous_cost:
                            # frontier.decrease_key(new_state, total_cost)
                            frontier[new_state] = total_cost
                            # print("dreasing")
                            parent[new_state.toString()] = [state.toString(), the_move]
                            prority_hash[new_state.toString()] = total_cost
                    else:
                        # frontier.push(new_state, total_cost)
                        frontier[new_state] = total_cost
                        prority_hash[new_state.toString()] = total_cost
                        parent[new_state.toString()] = [state.toString(), the_move]
                        # potential_states.push(new_state, current_cost)

                    if (new_state.is_end()):
                        print("Game has been solved.")
                        print(time.time() - start_time)
                        sol = get_Solution(parent, initial_state.toString(), new_state.toString())
                        # print(sol)
                        # print(len(sol))
                        # print(the_map)
                        # print(sum(1 for c in sol if c.isupper()))
                        print("A star", number_nodes_explored)
                        return sol



def get_Solution(path, source, goal):

    parent = path[goal]
    actions = []
    while True:
        if len(parent) >=3:
            print(parent[3])
        actions.insert(0, parent[1])
        parent = path[parent[0]]

        if parent[0] == None:
            break

    return actions





# ---------- Complete best first search ----------#
def greedy_simple_herustic(board, parameter_dict):
    initiate_class_variable()
    initial_state = bd.State(board)
    if initial_state.is_end():
        return []
    frontier = PriorityQueue()
    the_map = board.getMap()

    needs_recheck_accessiable_dict = dict()

    cost_function  = parameter_dict['cost']
    heuristic_function = parameter_dict['heuristic']
    dd_detection = parameter_dict['dd_detection']
    move_ordering = parameter_dict['move_ordering']

    # To be modified
    # Simple_deadlocks take too much time for large maps.
    dd.initialize()
    simple_deadlocks = None
    if dd_detection:
        simple_deadlocks = dd.simple_deadlocks(board.getMap())
        print('simple dd found!')
    # print(simple_deadlocks)
    start_time = time.time()
    # pre_compute_heuristic(board, simple_deadlocks)

    # print('heuristic function finished!')


    is_corral_on = False

    print(the_map)

    # assignment = min_matching_assign(initial_state)
    # assignment = False
    map_without_boxes = board.get_map_without_boxes()
    # frontier.push(initial_state, (initial_state, map_with_only_walls))
    # frontier.push(initial_state, get_heuristic_min_match(initial_state))
    # heuristic = linear_conficts_with_manhattan(initial_state, None)
    if heuristic_function == "pre computed":
        if simple_deadlocks == None:
            simple_deadlocks = dd.simple_deadlocks(board.getMap())
            print('simple dd found!')
        pre_compute_heuristic(board, simple_deadlocks)
        heuristic = linear_conficts_with_pre_computed_heuristic(initial_state, None)
        print('heuristic function finished!')
    else:
        min_match = parameter_dict['min_match']
        linear_conflicts = parameter_dict['linear_conflict']

        if not min_match and not linear_conflicts:
            heuristic = just_manhattan(initial_state)

        if not min_match and linear_conflicts:
            heuristic = simple_linear_conficts(initial_state, None)

        if min_match and not linear_conflicts:
            heuristic = get_heuristic_min_match(initial_state)

        if min_match and linear_conflicts:
            heuristic = linear_conficts_with_manhattan(initial_state, None)

    frontier.push(initial_state, heuristic)

    heuristic_hash[str(sorted(initial_state.boxes))] = heuristic
    exploredSet = set()
    parent = dict()
    parent[initial_state.toString()] = [None, None]
    # actions = []
    number_nodes_explored = 0

    cost_dict = dict()
    # heuristic_dict = dict()
    # heuristic_dict[initial_state.getHashValue()] = linear_conficts_with_manhattan(initial_state, None)



    needs_recheck_accessible_area = True
    accessiable_area = None

    # assignment = min_matching_assign(state)
    while frontier:
        if frontier.is_empty():
            print("---Can't find a solutiion---")
            print(the_map)
            return False

        state = frontier.pop()
        valid_moves = state.get_valid_moves()
        pushed_box = None
        if state.toString() not in exploredSet:
            exploredSet.add(state.toString())

            if is_corral_on:
                # Try to improve the key by normalizing
                if state.toString not in needs_recheck_accessiable_dict.keys():
                    accessiable_area = dectect_non_accessiable_area(state)
                    needs_recheck_accessiable_dict[state.toString] = accessiable_area
                else:
                    accessiable_area =  needs_recheck_accessiable_dict[state.toString]
                is_corral = is_player_next_to_corral(accessiable_area, state, the_map)

            # print(is_corral)
            for the_move in valid_moves:
                add_state = True
                new_state = state.update_state(action[the_move])
                new_map = new_state.get_map()
                # print(the_map)
                if (new_state.toString() in exploredSet):
                    add_state = False
                # Push a box
                if add_state:
                    if the_move.isupper():
                        pushed_box = (state.player[0][0] + action[the_move][0] * 2, state.player[0][1] + action[the_move][1] * 2)
                    # for box in new_state.boxes:
                        # Pay attention to those goals!!
                        if pushed_box in simple_deadlocks and pushed_box not in new_state.goals:
                            add_state = False
                        #  free_deadlocks already has a hash. No need to use here.
                        elif dd.free_deadlocks([x[:] for x in new_map], pushed_box, simple_deadlocks, new_state.boxes.copy()) and pushed_box not in new_state.goals:
                            add_state = False
                    else:
                        # If corral, we should encourage to push box, rather than move around, so not add the state
                        if is_corral_on:
                            if is_corral:
                                add_state = False

                    if is_corral_on:
                        # Only need to recalculate the accessiable area if then state is produced by pushing some box!
                        if add_state and the_move.islower():
                            needs_recheck_accessiable_dict[new_state.toString()] = accessiable_area


                if add_state:
                    number_nodes_explored = number_nodes_explored + 1
                    current_cost = get_cost(new_state, state, the_move.isupper(),cost_dict, cost_function)

                    # No influence move (One kind of moving ordering)
                    if the_move.isupper() and the_move in new_state.get_valid_moves():
                        # Otherwise no point to continue push the box.
                        # Move ordering + tunnel maros (Some cases)
                        if (pushed_box in state.goals and (pushed_box[0] + action[the_move][0], pushed_box[1] + action[the_move][1]) in state.goals) or (pushed_box not in state.goals):
                            current_heuristic = 0
                        else:
                            if str(sorted(new_state.boxes)) in heuristic_hash.keys():
                                current_heuristic = heuristic_hash[str(sorted(new_state.boxes))]
                            else:
                                if heuristic_function == "pre computed":
                                    current_heuristic = linear_conficts_with_pre_computed_heuristic(new_state,
                                                                                                    pushed_box)
                                else:
                                    if not min_match and not linear_conflicts:
                                        current_heuristic = just_manhattan(new_state)

                                    if min_match and not linear_conflicts:
                                        current_heuristic = get_heuristic_min_match(new_state)

                                    if min_match and linear_conflicts:
                                        current_heuristic = linear_conficts_with_manhattan(new_state, pushed_box)

                                    if not min_match and linear_conflicts:
                                        current_heuristic = simple_linear_conficts(state, pushed_box)
                    else:
                        if str(sorted(new_state.boxes)) in heuristic_hash.keys():
                            current_heuristic = heuristic_hash[str(sorted(new_state.boxes))]
                        else:
                            if heuristic_function == "pre computed":
                                current_heuristic = linear_conficts_with_pre_computed_heuristic(new_state, pushed_box)
                            else:
                                if not min_match and not linear_conflicts:
                                    current_heuristic = just_manhattan(new_state)

                                if min_match and not linear_conflicts:
                                    current_heuristic = get_heuristic_min_match(new_state)

                                if min_match and linear_conflicts:
                                    current_heuristic = linear_conficts_with_manhattan(new_state, pushed_box)

                                if not min_match and linear_conflicts:
                                    current_heuristic = simple_linear_conficts(state, pushed_box)
                        # current_heuristic = linear_conficts_with_manhattan(new_state, pushed_box)
                    frontier.push(new_state, current_cost + current_heuristic)
                    # potential_states.push(new_state, current_cost)
                    parent[new_state.toString()] = [state.toString(), the_move]
                    if (new_state.is_end()):
                        print("Game has been solved.")
                        print(time.time() - start_time)
                        sol = get_Solution(parent, initial_state.toString(), new_state.toString())
                        # print(sol)
                        # print(len(sol))
                        # print(the_map)
                        # print(sum(1 for c in sol if c.isupper()))
                        print("Complete best first search with ", parameter_dict, number_nodes_explored)
                        return sol




# ---------------------Floor fill-----------------



accessiable_hash = dict()

# Tested
def determine_accessiable_area(x, y, map, state, temp_dict, walls):
        accessiable_list = []
        temp_dict[state.toString()] = accessiable_list
        floor_fill(x, y, map, state, temp_dict, walls)
        return sorted(temp_dict[state.toString()])

def floor_fill(x, y, map, state, temp_dict, walls):
    accessiable_list = temp_dict[state.toString()]
    accessiable_list.append((x, y))
    if y + 1 <= len(map) and x <= len(map[y + 1]) and (x, y + 1) not in state.boxes and  (x, y + 1) not in walls and (x, y + 1) not in accessiable_list:
        floor_fill(x, y + 1, map, state, temp_dict, walls)
    if y - 1 >=0 and x <= len(map[y - 1]) and (x, y - 1) not in state.boxes and (x, y - 1) not in walls and (x, y - 1) not in accessiable_list:
        floor_fill(x, y - 1, map, state, temp_dict, walls)

    if x + 1 <= len(map[y]) and (x + 1, y) not in state.boxes and (x+1, y) not in walls and (x + 1, y) not in accessiable_list:
        floor_fill(x + 1, y, map, state, temp_dict, walls)
    if x - 1 >=0 and (x - 1, y) not in state.boxes and (x - 1, y) not in walls and (x - 1, y) not in accessiable_list:
        floor_fill(x - 1, y, map, state, temp_dict, walls)



# return the possible_player_position where next to some boxes
def get_valid_pushes(box, accessiable_squares):
    push_directions = []

    if (box[0] + 1,box[1]) in accessiable_squares:
        push_directions.append(('l', (box[0] + 1,box[1])))
    if (box[0] - 1, box[1]) in accessiable_squares:
        push_directions.append(('r', (box[0] - 1, box[1])))
    if (box[0], box[1] + 1) in accessiable_squares:
        push_directions.append(('u', (box[0], box[1] + 1)))
    if (box[0] , box[1] - 1) in accessiable_squares:
        push_directions.append(('d', (box[0], box[1] - 1)))
    return push_directions


moving_action = {'r': (1, 0),
          'l': (-1, 0),
          'd': (0, 1),
          'u': (0, -1)}

def construct_path(old_position, goal_position,accessiable_squares, direction):

    # if abs(goal_position[0] - old_position[0]) + abs(goal_position[1] - old_position[1]) == 1:
    #     return None

    if goal_position[0] == old_position[0] and goal_position[1] - old_position[1] == 1 and direction == 'u':
        return None

    if goal_position[0] == old_position[0] and goal_position[1] - old_position[1] == -1 and direction == 'd':
        return None

    if goal_position[1] == old_position[1] and goal_position[0] - old_position[0] == 1 and direction == 'l':
        return None

    if goal_position[1] == old_position[1] and goal_position[0] - old_position[0] == -1 and direction == 'r':
        return None



    if old_position == goal_position:
        return None
    actions = []
    explored_positions = set()

    initial_position = old_position
    parent = dict()
    frontier = []
    frontier.append(old_position)
    parent[old_position] = [None, None]
    explored_positions.add(old_position)
    # print("old", old_position)
    # print("goal", goal_position, "start", old_position)


    finished = False
    while len(frontier) != 0 and not finished:
        old_position = frontier.pop(0)

        for the_move in moving_action:
            direction = moving_action[the_move]
            current_position = (old_position[0] + direction[0], old_position[1] + direction[1])
            if current_position in explored_positions:
                continue

            if current_position in accessiable_squares:
                explored_positions.add(current_position)
                frontier.append(current_position)
                parent[current_position] = [old_position, the_move]
            if current_position == goal_position:
                # print("goal achieved", current_position)
                print(accessiable_squares)
                finished = True
                break

    the_parent = parent[goal_position]
    # print("goal", goal_position, "start", old_position)
    # print("the_parent", the_parent)

    actions = []
    count = 0
    while True:
        print(count)
        count = count + 1
        actions.insert(0, the_parent[1])
        # print("parent name", the_parent[0])
        the_parent = parent[the_parent[0]]
        # print("start at", initial_position, "end at", goal_position)

        if the_parent[0] == None:
            # print(actions)
            break

    return actions


def get_Solution2(path, source, goal, initial_Accessable_squares):
    parent = path[goal]
    actions = []


    # parent = path[goal]
    # actions = []
    # while True:
    #     actions.insert(0, parent[1])
    #     parent = path[parent[0]]
    #     if parent[0] == None:
    #         break
    #
    # return actions
    final = None
    while True:
        # parent[1] is the pushing direction, parent[2] is the old_player position, the parent state after pushing
        # parent[3] is new_player_position after pushing, parent[4] is the accessiable_squares
        # parent[5] is after push_player
        # parent[6] is the current state before pushing


        actions.insert(0, parent[1].upper())
        dir = moving_action[parent[1]]
        originaled_position = (parent[2][0] - 1, parent[2][1] - 1)
        # print("start at", parent[2], "end at", parent[6], "direction", parent[1])
        final = parent[3]

        result = construct_path(parent[2], parent[6], parent[4], parent[1])
        if result != None:
            actions = result + actions

        print("actions", actions)
        parent = path[parent[0]]
        if parent[0] == None or parent[4] == None:
            break

    # actions = construct_path(source, final, initial_Accessable_squares) + actions
    # print("actions", actions)
    return actions


def print_map(map):
    for row in map:
        for each in row:
            print(each, end='')
        print()



def aStarSearch_simple_herustic_floor_fill(board, parameter_dict):

    if parameter_dict == None:
        cost_function = "encourages_pushing"
        heuristic_function = "dynamic"
        dd_detection = True
        moving_ordering = True

    else:
        cost_function  = parameter_dict['cost']
        heuristic_function = parameter_dict['heuristic']
        dd_detection = parameter_dict['dd_detection']
        move_ordering = parameter_dict['move_ordering']

    # initiate_class_variable()
    initial_state = bd.State(board)
    if initial_state.is_end():
        return []
    frontier = heapdict.heapdict()
    the_map = board.getMap()
    print_map(initial_state.get_map2())

    walls = board.get_inital_walls()

    cost_dict = dict()

    needs_recheck_accessiable_dict = dict()

    dd.initialize()

    start_time = time.time()

    accessiable_hash = dict()
    # To be modified
    # Simple_deadlocks take too much time for large maps.
    simple_dealocks = None
    if dd_detection:
        simple_deadlocks = dd.simple_deadlocks(copy.deepcopy(board.getMap()))
        print('simple dd found!')
    # print(simple_deadlocks)
    if heuristic_function == "pre computed":
        if simple_deadlocks == None:
            simple_deadlocks = dd.simple_deadlocks(copy.deepcopy(board.getMap()))
            print('simple dd found!')
        pre_compute_heuristic(board, simple_deadlocks)
        heuristic = linear_conficts_with_pre_computed_heuristic(initial_state, None)
    else:
        if parameter_dict == None:
            min_match = True
            linear_conflicts = True
        else:
            min_match = parameter_dict['min_match']
            linear_conflicts = parameter_dict['linear_conflict']

        if not min_match and not linear_conflicts:
            heuristic = just_manhattan(initial_state)

        if not min_match and linear_conflicts:
            heuristic = simple_linear_conficts(initial_state, None)

        if min_match and not linear_conflicts:
            heuristic = get_heuristic_min_match(initial_state)

        if min_match and linear_conflicts:
            heuristic = linear_conficts_with_manhattan(initial_state, None)

    prority_hash = dict()
    # map_without_boxes = board.get_map_without_boxes()
    # heuristic = linear_conficts_with_manhattan(initial_state, None)
    frontier[initial_state] = heuristic
    prority_hash[str(sorted(initial_state.toString()))] = 0 + heuristic
    initial_player = initial_state.player[0]
    temp_dict = dict()


    heuristic_hash[str(sorted(initial_state.boxes))] = heuristic
    exploredSet = set()
    parent = dict()
    parent[initial_state.toString()] = [None, None, None, None, None, None]
    number_nodes_explored = 0

    needs_recheck_accessiable = True

    inital_accessiable_Area = determine_accessiable_area(initial_player[0], initial_player[1], the_map, initial_state, temp_dict, walls)

    initial_state.set_normalized_player(inital_accessiable_Area)
    accessiable_hash[initial_state.toString_normalized()] = inital_accessiable_Area
    i = 0
    while frontier:
        print(time.time() - start_time)

        if not bool(frontier):
            print("---Can't find a solutiion---")
            print(the_map)
            return False

        state = frontier.popitem()[0]

        player = state.player[0]
        temp_dict = dict()
        # accessiable_squares = determine_accessiable_area(player[0], player[1], state.get_map2(), state, temp_dict, walls)
        boxes = state.boxes

        if state.toString_normalized() == None:
            accessiable_squares = determine_accessiable_area(player[0], player[1], state.get_map2(), state, temp_dict,
                                                             walls)

            state.set_normalized_player(accessiable_squares)
            accessiable_hash[state.toString_normalized()] = accessiable_squares
        else:
            accessiable_squares = accessiable_hash[state.toString_normalized()]
        for box in boxes:
            valid_pushes = get_valid_pushes(box, accessiable_squares)
            if len(valid_pushes) != 0:
                for valid_push in valid_pushes:
                    add_state = True
                    result = state.update_state_floor_fill(valid_push)
                    new_state = result[0]
                    after_push_player = result[2]
                    # Can not push!!!!
                    if new_state == False:
                        continue
                    else:
                        # To be changed here, use normalized player position

                        if new_state.toString_normalized() == None:
                            temp_dict = dict()
                            new_player = new_state.player[0]
                            new_accessiable_squares = determine_accessiable_area(new_player[0], new_player[1], new_state.get_map2(),
                                                                             new_state, temp_dict,
                                                                             walls)

                            new_state.set_normalized_player(new_accessiable_squares)
                            accessiable_hash[new_state.toString_normalized()] = new_accessiable_squares
                            # print(new_state.toString_normalized())

                        if new_state.toString_normalized() in exploredSet:
                            add_state = False
                            # print("noo")
                        else:
                            # print(new_state.toString())
                            exploredSet.add(new_state.toString_normalized())
                        new_map = new_state.get_map()
                        pushed_box = result[1]
                        if pushed_box in simple_deadlocks and pushed_box not in new_state.goals:
                            add_state = False
                        elif dd.free_deadlocks([x[:] for x in new_map], pushed_box, simple_deadlocks, new_state.boxes.copy()) and pushed_box not in new_state.goals:
                             add_state = False

                    if add_state:
                        number_nodes_explored = number_nodes_explored + 1
                        current_cost = get_cost(new_state, state, True, cost_dict, cost_function)

                        if str(new_state.boxes) not in heuristic_hash.keys():
                            # current_cost = get_cost(new_state)


                            if heuristic_function == "pre computed":
                                current_heuristic = linear_conficts_with_pre_computed_heuristic(new_state, pushed_box)
                            else:
                                if not min_match and not linear_conflicts:
                                    current_heuristic = just_manhattan(new_state)

                                if min_match and not linear_conflicts:
                                    current_heuristic = get_heuristic_min_match(new_state)

                                if min_match and linear_conflicts:
                                    current_heuristic = linear_conficts_with_manhattan(new_state, pushed_box)

                                if not min_match and linear_conflicts:
                                    current_heuristic = simple_linear_conficts(state, pushed_box)



                            # current_heuristic = linear_conficts_with_pre_computed_heuristic(new_state, pushed_box)
                            heuristic_hash[str(new_state.boxes)] = current_heuristic
                                # current_heuristic = linear_conficts_with_manhattan(new_state, pushed_box)

                        else:
                            current_heuristic = heuristic_hash[str(new_state.boxes)]

                        total_cost = current_cost + current_heuristic
                        if new_state.toString() in prority_hash.keys():
                            # print("checking")
                            previvous_cost = prority_hash[new_state.toString()]
                            # print("?")
                            # print("pc", current_heuristic)
                            if total_cost < previvous_cost:
                                # frontier.decrease_key(new_state, total_cost)
                                frontier[new_state] = total_cost
                                # print("dreasing")
                                parent[new_state.toString()] = [state.toString(), valid_push[0], state.player[0], new_state.player[0], accessiable_squares, after_push_player, valid_push[1]]
                                prority_hash[new_state.toString()] = total_cost
                        else:
                            # frontier.push(new_state, total_cost)
                            frontier[new_state] = total_cost
                            prority_hash[new_state.toString()] = total_cost
                            parent[new_state.toString()] = [state.toString(), valid_push[0], state.player[0], new_state.player[0], accessiable_squares, after_push_player, valid_push[1]]
                            # potential_states.push(new_state, current_cost)

                        # potential_states.push(new_state, current_cost)
                        # new_state.player[0] = after push position
                        # parent[new_state.toString()] = [state.toString(), valid_push[0], state.player[0], new_state.player[0], accessiable_squares, after_push_player, valid_push[1]]
                        if (new_state.is_end()):
                            print("Game has been solved.")
                            print(time.time() - start_time)
                            sol = get_Solution2(parent, initial_player, new_state.toString(),inital_accessiable_Area)
                            # sol = get_Solution(parent, initial_player, new_state.toString())
                            # print(sol)
                            # print(len(sol))
                            # print(the_map)
                            # print(sum(1 for c in sol if c.isupper()))
                            print("Floor fill", number_nodes_explored)
                            print(time.time() - start_time)
                            # print(new_state.get_map())
                            return sol
    return False



