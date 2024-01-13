import deadlock_detection as dd
import board as bd
import copy
import solver


import floor_fill_solver as ff

import level_generation


action = {'r': (1, 0),
          'l': (-1, 0),
          'd': (0, 1),
          'u': (0, -1),
          'R': (1, 0),
          'L': (-1, 0),
          'D': (0, 1),
          'U': (0, -1)}

def test_free_deadlockes():
    the_board = bd.Board(filename = "free_deadlocks_map.txt")

    boxes = the_board.get_inital_boxes()

    print("inital boxes", the_board.get_inital_boxes())
    # print(the_map[2][3])
    # print(new_map)

    the_map = the_board.getMap()
    simple_deadlocks = dd.simple_deadlocks(the_board.getMap())


    if dd.free_deadlocks(copy.deepcopy(the_map), (4, 2), simple_deadlocks, copy.deepcopy(boxes)) == True:
        print("---- test successfully")
    else:
        print("test fails")

    # print(the_map[2][3])
    # print(the_map[3][3])
    # print(dd.is_blocked_vertically(the_map, (2, 3),  simple_deadlocks, copy.deepcopy(boxes)))
    # print(dd.is_blocked_horizontally(the_map, (3, 3),  simple_deadlocks, copy.deepcopy(boxes)))
    # print(dd.is_blocked_vertically(the_map, (3, 2), simple_deadlocks, copy.deepcopy(boxes)))
    # print(dd.is_blocked_horizontally(the_map, (3, 3), simple_deadlocks, copy.deepcopy(boxes)))
    # print(new_map)


def test_solve_the_game(level, sol):
    the_board = bd.Board(filename=level)
    state = bd.State(the_board)

    for step in sol:
        state = state.update_state(action[step])

    if state.is_end():
        print("Solution is correct")

# test_free_deadlockes()

def test_detect_corrals(the_board):
    level = the_board.getMap()
    sol = solver.dectect_accessiable_area(the_board)
    for temp in sol:
        level[temp[1]][temp[0]] = '?'

    print(solver.is_player_next_to_corral(sol, the_board.get_inital_player()[0], level))
    print(level)

def test_free_square(the_board):
    free_square = the_board.get_free_square()
    print("There are ", len(free_square))
    print(free_square)

def test_map_with_only_Walls(the_board):
    map_with_only_walls = the_board.get_map_only_walls()
    original_map = the_board.getMap()

    print(map_with_only_walls)
    print(original_map)

    i = 0
    j = 0
    for row in map_with_only_walls:
        j = 0
        for each in row:
            if (map_with_only_walls[i][j] == '#' and original_map[i][j] == '#') or (map_with_only_walls[i][j] == ' ' and original_map[i][j] != '#')\
                or (map_with_only_walls[i][j] == '&' and original_map[i][j] == '&'):
                j = j+1
            else:
                return False
        i = i + 1
    return True


def test_pre_compute_heuristic(the_board):
    simple_deadlocks = dd.simple_deadlocks(the_board.getMap())
    print(solver.pre_compute_heuristic(the_board, simple_deadlocks))

def test_accessiable_squares(the_state, map):
    print('Original map')
    for row in map:
        for each in row:
            print(each, end='')
        print()
    accessiable_squares = level_generation.accessable_area(the_state)

    for each in accessiable_squares:
        map[each[1]][each[0]] = 'A'

    for row in map:
        for each in row:
            print(each, end='')
        print()


def test_level_generation():
    map = level_generation.generating_map(8, 8, 3)
    the_board = bd.Board(map=map)
    sol = solver.aStarSearch(the_board)
    for step in sol:
        state = state.update_state(action[step])


    if state.is_end():
        print("Solable level")
    else:
        print("generated level is not solvable")


def test_ranking2():
    sol = ['U', 'U', 'R', 'l', 'd', 'd', 'r', 'U', 'r', 'U', 'U', 'U', 'U', 'r', 'u', 'L', 'L', 'L', 'u', 'U', 'D']
    i = level_generation.ranking_2(sol)
    print(i == 5)

def test_is_good_goal_position():
    the_board = bd.Board(filename='generated_test_gui.txt')
    map = the_board.getMap()
    goals = the_board.get_inital_goals()
    print(level_generation.check_good_goal_position(map, goals))

def test_is_valid_action():
    the_board = bd.Board(filename='simple_test.txt')
    state = bd.State(the_board)
    actions = ['r', 'r', 'l', 'L', 'L', 'r', 'd','D', 'l', 'd', 'L']

    ans = ['T', 'F', 'T', 'T', 'F', 'T', 'T', 'F', 'T', 'T', 'T']

    result = []
    for each in actions:
        if state.is_legal_action(action[each])[0]:
            result.append('T')
            state = state.update_state(action[each])
        else:
            result.append('F')
    print(result == ans)

def test_get_valid_actions():
    the_board = bd.Board(filename='simple_test.txt')
    state = bd.State(the_board)
    result = []
    actions = ['l', 'l']

    ans = [['r', 'L', 'u', 'd'], ['r', 'u', 'd']]

    for each in actions:
        result.append(state.get_valid_moves())
        state = state.update_state(action[each])

    print(sorted(result) == ans)



def evaluation(board, state):
    # print(solver.depthFirstSearch(board))

    # solver.uniformCostSearch(board)
    sol = solver.depthFirstSearch(board)
    print(sol)

    for step in sol:
        state = state.update_state(action[step])


    if state.is_end():
        print("Solution is correct")

    # solver.aStarSearch_simple_herustic(board)


def test_floor_fill(board, state):
    player = state.player[0]
    result = ff.determine_accessiable_area(player[0], player[1], board.getMap(), state)
    print(sorted(result))

def test_floor_fill_solver(board, state):
    sol = solver.aStarSearch_simple_herustic_floor_fill(board)

    i = 0

    for step in sol:
        state = state.update_state(action[step])
        if step.isupper():
            i = i + 1



    if state.is_end():
        print("Solution is correct")
    print("returned", sol)
    print(i)

def test_floor_fill_answer(board, state):
    player = state.player[0]
    temp_dict = dict()
    walls = board.get_inital_walls()
    result = solver.determine_accessiable_area(player[0], player[1], board.getMap(), state, temp_dict, walls)

    print(solver.construct_path(player, (2,1), result))



# the_board = bd.Board(filename='levels/level3.txt')
# test_detect_corrals(the_board)
the_board = bd.Board(filename='levels/level set/level3.txt')
map = the_board.getMap()
goals = the_board.get_inital_goals()
# test_free_square(the_board)
# print(test_map_with_only_Walls(the_board))

# test_pre_compute_heuristic(the_board)

state = bd.State(the_board)
# test_accessiable_squares(state, map)
# test_level_generation()

# test_is_good_goal_position()
# test_floor_fill(the_board, state)

# evaluation(the_board, state)
# test_floor_fill_solver(the_board, state)

# test_floor_fill_answer(the_board,state)

# test_ranking2()


# test_is_valid_action()
test_get_valid_actions()





