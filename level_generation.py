import numpy as np
import random
import blocks_template
import sys
import queue
import board as bd
import copy
import time

import solver

import deadlock_detection as dd

# Don't need to record the last move. Because we generate the map by pulling.
# we can not pull back!

action = {'r': (1, 0),
          'l': (-1, 0),
          'd': (0, 1),
          'u': (0, -1),
          'R': (1, 0),
          'L': (-1, 0),
          'D': (0, 1),
          'U': (0, -1)}


# https://bitbucket.org/eightonthebottom/puzzle-generation/src/master/sokoban_gen.py
def insert_template(level, template_to_be_added, x, y):
    for current_x in range(0, 3):
        for current_y in range(0, 3):
            if x + current_x < level.shape[0] and y + current_y < level.shape[1]:
                level[x + current_x][y + current_y] = template_to_be_added[current_x + 1][current_y + 1]
    return level


# main idea is to check if the ghost square match the pattern.
def canInsert(level, template_to_be_added, x, y, height, width):
    can_insert = True
    # check if top row match
    start_time = time.time()
    for i in range(0, 5):
        if time.time() - start_time >= 5:
            return False
        if y + i < width:
            if level[x][y + i] != 0:
                if level[x][y + i] != template_to_be_added[0][i]:
                    can_insert = False

    # left col
    for i in range(0, 5):
        if time.time() - start_time >= 5:
            return False
        if x + i < height:
            if level[x + i][y] != 0:
                if level[x + i][y] != template_to_be_added[i][0]:
                    can_insert = False

    # right col
    for i in range(0, 5):
        if time.time() - start_time >= 5:
            return False
        if x + 4 < height:
            if y + i < width:
                if level[x + 4][y + i] != 0:
                    if level[x + 4][y + i] != template_to_be_added[4][i]:
                        can_insert = False

    # bottom row
    for i in range(0, 5):
        if time.time() - start_time >= 5:
            return False
        if y + 4 < width:
            if x + i < height:
                if level[x + i][y + 4] != 0:
                    if level[x + i][y + 4] != template_to_be_added[i][4]:
                        can_insert = False

    return can_insert


def getEmptyRoom(height, width):
    start_time = time.time()
    x = 0
    y = 0

    level = np.ndarray((height, width))
    level.fill(0)

    count = 0
    while True:
        print("in getting empty room function", time.time())
        # get a random template
        current_time = time.time()
        if current_time - start_time >= 10:
            print("exceed time")
            return []
        index = random.randint(0, 16)
        template_to_be_added = blocks_template.template[index]

        # rotate template random time
        rotateTimes = random.randint(0, 3)
        template_to_be_added = np.rot90(template_to_be_added, rotateTimes)

        flip = random.randint(0, 3)
        if flip == 1:
            # flip horizontal random
            template_to_be_added = np.fliplr(template_to_be_added)
        elif flip == 2:
            # flip vertical random
            template_to_be_added = np.flipud(template_to_be_added)
        # We need use the ghost square to check if we can insert this template
        if canInsert(level, template_to_be_added, x, y, height, width):
            # Only the 3*3 size in each template contributes.
            level = insert_template(level, template_to_be_added, x, y)
            x = x + 3

            # Turn to the next col of blocks
            if x > height - 1:
                y = y + 3
                x = 0

        if y > width - 1:
            return level

        count = count + 1
        if count > 10000:
            return level


def check_floor_connected(level):
    is_connected = True

    start_time = time.time()

    # main idea, starts with a floor, and use dfs to search for all its descendant
    finish = False
    x = 0
    first_floor = (-1, -1)
    while not finish:
        print("hehe")
        if time.time() - start_time >= 5:
            return 0
        for y in range(level.shape[1]):
            if level[x][y] == 1:
                finish = True
                first_floor = (x, y)
                break
        x += 1
        if x >= level.shape[0]:
            finish = True

    if first_floor == (-1, -1):
        return False
    searched_floor = []
    searched_floor.append(first_floor)

    visited_floor = []
    visited_floor.append(first_floor)
    start_time = time.time()

    while len(searched_floor) != 0:
        current_time = time.time()

        print("xixi", current_time)
        print(current_time - start_time)
        if current_time - start_time > 5:
            print("returned")
            return 0
        current_floor = searched_floor.pop()

        if current_floor[0] - 1 >= 0:
            if level[current_floor[0] - 1][current_floor[1]] == 1:
                if (current_floor[0] - 1, current_floor[1]) not in visited_floor:
                    searched_floor.append((current_floor[0] - 1, current_floor[1]))
                    visited_floor.append((current_floor[0] - 1, current_floor[1]))

        if current_floor[0] + 1 < level.shape[0]:
            if level[current_floor[0] + 1][current_floor[1]] == 1:
                if (current_floor[0] + 1, current_floor[1]) not in visited_floor:
                    searched_floor.append((current_floor[0] + 1, current_floor[1]))
                    visited_floor.append((current_floor[0] + 1, current_floor[1]))

        if current_floor[1] - 1 >= 0:
            if level[current_floor[0]][current_floor[1] - 1] == 1:
                if (current_floor[0], current_floor[1] - 1) not in visited_floor:
                    searched_floor.append((current_floor[0], current_floor[1] - 1))
                    visited_floor.append((current_floor[0], current_floor[1] - 1))
        if current_floor[1] + 1 < level.shape[1]:
            if level[current_floor[0]][current_floor[1] + 1] == 1:
                if (current_floor[0], current_floor[1] + 1) not in visited_floor:
                    searched_floor.append((current_floor[0], current_floor[1] + 1))
                    visited_floor.append((current_floor[0], current_floor[1] + 1))

    if len(visited_floor) == (level == 1).sum():
        is_connected = True
    connectivity_rate = len(visited_floor) * 1.0 / (level == 1).sum()
    print("returned")
    return (len(visited_floor), connectivity_rate, visited_floor)


# Pre: needs to ensure the connectivity rate is large than 90,
# then we can remove the unconnected_floor
def remove_connected_floor(level, connected_foor):
    result = np.copy(level)
    # print("before changing:", len(connected_foor))
    for x in range(0, level.shape[0]):
        for y in range(0, level.shape[1]):
            if (x, y) not in connected_foor:
                # change unconnected_floor to wall
                result[x][y] = 2
    print("removed_connected_floor")
    return result


# Bound the room with walls
def boundRoom(level):
    new_level = np.ndarray((level.shape[0] + 2, level.shape[1] + 2))
    for y in range(new_level.shape[1]):
        new_level[0][y] = 2
        new_level[new_level.shape[0] - 1][y] = 2
    for x in range(new_level.shape[0]):
        new_level[x][0] = 2
        new_level[x][new_level.shape[1] - 1] = 2

    for x in range(0, level.shape[0]):
        for y in range(0, level.shape[1]):
            new_level[x + 1][y + 1] = level[x][y]

    return new_level


def getRoom(height, width):
    found = False
    print("getting empty room")
    while not found:
        print("haha")
        level = getEmptyRoom(height, width)
        if level != []:
            found = True

    # Regenerate room until the level has connectivity greater than or equal to 90.
    connectivity_result = check_floor_connected(level)
    while connectivity_result[1] < 0.9:
        print("?????")
        level = getEmptyRoom(height, width)
        print("checking connectivity")
        connectivity_result = check_floor_connected(level)
    level = remove_connected_floor(level, connectivity_result[2])

    return level


# Need to be modified to increase diffiucly!
#  maybe try to force the goals to near each other
def placeGoals(level, no_goals):
    finished = False
    count = 0
    goals = []
    start_time = time.time()
    while not finished:
        if time.time() - start_time >= 6:
            return []
        x = random.randint(0, level.shape[0] - 1)
        y = random.randint(0, level.shape[1] - 1)

        if level[x][y] == 1:
            count = 0
            if level[x-1][y] != 1:
                count += 1
            if level[x+1][y] != 1:
                count += 1
            if level[x][y + 1] != 1:
                count += 1
            if level[x][y - 1] != 1:
                count += 1
            if count < 3:
                level[x][y] = 5
                no_goals -= 1
                goals.append((x, y))
        if no_goals <= 0:
            finished = True
        count = count + 1
        if count > 10000:
            break
    return (level, goals)


# place player near each goal/box
def makeStartSet(level, positions):
    startSet = []
    for position in positions:
        current_level = np.copy(level)
        if position[0] - 1 >= 0 and current_level[position[0] - 1][position[1]] == 1:
            current_level[position[0] - 1][position[1]] = 3
        elif position[0] + 1 < level.shape[0] and current_level[position[0] + 1][position[1]] == 1:
            current_level[position[0] + 1][position[1]] = 3
        elif position[1] - 1 >= 0 and current_level[position[0]][position[1] - 1] == 1:
            current_level[position[0]][position[1] - 1] = 3
        elif position[1] + 1 < level.shape[1] and current_level[position[0]][position[1] + 1] == 1:
            current_level[position[0]][position[1] + 1] = 3
        else:
            print(level)
            print("-----Error when placing player!-----")
            return None
        # Box on the goal
        startSet.append(current_level)
    return startSet


def readMap(level):
    the_map = []
    for x in range(level.shape[0]):
        row = []
        for y in range(level.shape[1]):
            if level[x][y] == 1:
                row.append(' ')
            if level[x][y] == 2:
                row.append('#')
            if level[x][y] == 3:
                row.append('&')
            if level[x][y] == 4:
                row.append('.')
            if level[x][y] == 5:
                row.append('X')
        the_map.append(row)
    return the_map


def readMaps(levels):
    the_maps = []
    for the_level in levels:
        the_maps.append(readMap(the_level))
    return the_maps


def remove_player(map, player):
    map = copy.deepcopy(map)
    if map[player[1]][player[0]] == '+':
        map[player[1]][player[0]] = '.'
    else:
        map[player[1]][player[0]] = ' '
    return map


# takes a set of states and returns the set of states one step futher.
# Here, one step means the player has pulled a box.
def expand(prevResults):
    resultSet = []
    count = 0
    for current_level in prevResults:
        finished = False
        count = 0
        current_board = bd.Board(map=current_level)
        # print("inital_player", current_board.get_inital_player())
        state = bd.State(current_board)
        while not finished:
            valid_moves = state.get_valid_moves_pull()
            # To be changed, should encourage to pull more boxes,
            # rathen than focus on the current one.
            # maybe stop until there is two upper in the valid_moves
            upper = 0
            # print("state is", state.get_map())
            # print(state.player)
            # print(valid_moves)
            random.shuffle(valid_moves)
            for the_move in valid_moves:
                if the_move.isupper():
                    upper += 1
                    state = state.update_state_pull(action[the_move])
                    finished = True
                    # print(state)
                    if state == False:
                        continue
                    if (remove_player(state.get_map2(), state.player[0]), state.boxes) not in resultSet:
                        resultSet.append((remove_player(state.get_map2(), state.player[0]), state.boxes))
                    finished = True
            # At this point, hasn't got any result for this state, continue
            if not finished:
                count += 1
                if (len(valid_moves) == 0):
                    finished = True
                    if (remove_player(state.get_map2(), state.player[0]), state.boxes) not in resultSet:
                        resultSet.append((remove_player(state.get_map2(), state.player[0]), state.boxes))
                else:
                    move = valid_moves[random.randint(0, len(valid_moves) - 1)]
                    state = state.update_state_pull(action[move])
                    if count > 2000:
                        finished = True
                        if (remove_player(state.get_map2(), state.player[0]), state.boxes) not in resultSet:
                            resultSet.append((remove_player(state.get_map2(), state.player[0]), state.boxes))
    return resultSet

def print_map(map):
    for row in map:
        for each in row:
            print(each, end='')
        print()


# takes a set of states and returns the set of states one step futher.
# Here, one step means the player has pulled a box.
def expand2(prevResults):
    resultSet = []
    count = 0
    explored_set = set()
    if prevResults == None:
        return []
    for current_level in prevResults:
        # print_map(current_level)
        print()
        finished = False
        count = 0
        current_board = bd.Board(map=current_level)
        # print("inital_player", current_board.get_inital_player())
        state = bd.State(current_board)
        player = state.player[0]
        # compute the accessiable squares of the current state
        accessable_squares = accessable_area(state)
        map = remove_player(current_level, player)
        # print("after removing")
        # print_map(map)
        map = place_player_to_box2(map, state.boxes, accessable_squares)
        if map == False:
            print("place player error")
            # print(accessable_squares)
            # print(state.boxes)

        # print("after placing")
        # print_map(map)
        # print()
        current_board = bd.Board(map = map)
        state = bd.State(current_board)
        while not finished:
            valid_moves = state.get_valid_moves_pull()
            # To be changed, should encourage to pull more boxes,
            # rathen than focus on the current one.
            # maybe stop until there is two upper in the valid_moves
            upper = 0
            # print("state is", state.get_map())
            # print(state.player)
            # print(valid_moves)
            random.shuffle(valid_moves)
            for the_move in valid_moves:
                if the_move.isupper():
                    upper += 1
                    state = state.update_state_pull(action[the_move], True)
                    finished = True
                    # print(state)
                    if state == False:
                        continue
                    if state.get_map2() not in resultSet:
                        resultSet.append(state.get_map2())
                    finished = True
            # At this point, hasn't got any result for this state, continue
            if not finished:
                count += 1
                if (len(valid_moves) == 0):
                    finished = True
                    if state.get_map2() not in resultSet:
                        resultSet.append(state.get_map2())
                else:
                    move = valid_moves[random.randint(0, len(valid_moves) - 1)]
                    if move.isupper():
                        is_pulling = True
                        finished = True
                    else:
                        is_pulling = False
                    state = state.update_state_pull(action[move], False)
                    if count > 2000:
                        finished = True
                        if state.get_map2() not in resultSet:
                            resultSet.append(state.get_map2())

    print("resulrse", resultSet)
    return resultSet

# input: A map
# Randomply place a player next to a box
def place_player_to_box2(map, boxes, accessable_area):
        # print("boxes", boxes)
    random.shuffle(boxes)
    for i in range(len(boxes)):
        box = boxes[i]
        map = copy.deepcopy(map)
        if map[box[1] + 1][box[0]] == ' ' and (box[0], box[1] + 1) in accessable_area:
            map[box[1] + 1][box[0]] = '&'
            return map
        elif map[box[1] + 1][box[0]] == '.' and (box[0], box[1] + 1) in accessable_area:
            map[box[1] + 1][box[0]] = '+'
            return map

        elif map[box[1] - 1][box[0]] == ' ' and (box[0], box[1] - 1) in accessable_area:
            map[box[1] - 1][box[0]] = '&'
            return map
        elif map[box[1] - 1][box[0]] == '.' and (box[0], box[1] - 1) in accessable_area:
            map[box[1] - 1][box[0]] = '+'
            return map

        elif map[box[1]][box[0] + 1] == ' ' and (box[0] + 1, box[1]) in accessable_area:
            map[box[1]][box[0] + 1] = '&'
            return map
        elif map[box[1]][box[0] + 1] == '.' and (box[0] + 1, box[1]) in accessable_area:
            map[box[1]][box[0] + 1] = '+'
            return map

        elif map[box[1]][box[0] - 1] == ' ' and (box[0] - 1, box[1]) in accessable_area:
            map[box[1]][box[0] - 1] = '&'
            return map
        elif map[box[1]][box[0] - 1] == '.' and (box[0] - 1, box[1]) in accessable_area:
            map[box[1]][box[0] - 1] = '+'
            return map

    return False


def accessable_area(the_state):
    initial_state = the_state
    frontier = []
    frontier.append(initial_state)

    # avoid to search the same state multiple times
    exploredSet = set()

    while frontier:
        # print(len(frontier))
        state = frontier.pop()
        valid_moves = state.get_valid_moves()
        exploredSet.add(state.player[0])
        for the_move in valid_moves:
            if the_move.islower():
                new_state = state.update_state(action[the_move])
                if new_state.player[0] not in exploredSet:
                    frontier.append(new_state)
    return exploredSet



# input: set of maps without player
# return: set of maps with a player located next to some box.
def place_player_to_box(tuples):
    resultSet = []
    for tuple in tuples:
        boxes = tuple[1]
        # print("boxes", boxes)
        for i in range(random.randint(1,1)):
            rnd_index = random.randint(0, len(boxes) - 1)
            box = boxes[rnd_index]
            map = copy.deepcopy(tuple[0])
            board = bd.Board(map=map)
            state = bd.State(board=board)
            accessable_area = accessable_area(state)
            if map[box[1] + 1][box[0]] == ' ' and (box[0], box[1] + 1) in accessable_area:
                map[box[1] + 1][box[0]] = '&'
                resultSet.append(map)
            elif map[box[1] + 1][box[0]] == '.' and (box[0], box[1] + 1) in accessable_area:
                map[box[1] + 1][box[0]] = '+'
                resultSet.append(map)

            elif map[box[1] - 1][box[0]] == ' ' and (box[0], box[1] - 1) in accessable_area:
                map[box[1] - 1][box[0]] = '&'
                resultSet.append(map)
            elif map[box[1] - 1][box[0]] == '.' and (box[0], box[1] - 1) in accessable_area:
                map[box[1] - 1][box[0]] = '+'
                resultSet.append(map)

            elif map[box[1]][box[0] + 1] == ' ' and (box[0] + 1, box[1]) in accessable_area:
                map[box[1]][box[0] + 1] = '&'
                resultSet.append(map)
            elif map[box[1]][box[0] + 1] == '.' and (box[0] + 1, box[1]) in accessable_area:
                map[box[1]][box[0] + 1] = '+'
                resultSet.append(map)

            elif map[box[1]][box[0] - 1] == ' ' and (box[0] - 1, box[1]) in accessable_area:
                map[box[1]][box[0] - 1] = '&'
                resultSet.append(map)
            elif map[box[1]][box[0] - 1] == '.' and (box[0] - 1, box[1]) in accessable_area:
                map[box[1]][box[0] - 1] = '+'
                resultSet.append(map)
            else:
                return False

    return resultSet


def printMaps(maps):
    i = 0
    for map in maps:
        print("map ", i, "is ", map)
        i = i + 1



# To be modified, use normalized player location to reduce map size!
def go(level, limit, no_goals):
    level = boundRoom(level)
    placeGoalsRes = placeGoals(level, no_goals)
    level = placeGoalsRes[0]
    goals = placeGoalsRes[1]

    startSet = makeStartSet(level, goals)
    if startSet == None:
        return False
    # Transfer to the normal map
    startSet = readMaps(startSet)

    resultSet = startSet
    depth = 1
    while True:
        print("depth?", depth)
        prevSet = resultSet
        # Maybe replace the player in each new level?
        # print("before expanding", resultSet)
        resultSet = expand(resultSet)
        # print("resultSet is", resultSet)
        # list(dict.fromkeys(mylist))
        resultSet = place_player_to_box(resultSet)
        if resultSet == False:
            print('place player error')
        # print("resultSet2 is", resultSet)
        if len(resultSet) == 0:
            break
        depth += 1
        if depth >= limit:
            break

    printMaps(resultSet)
    return (resultSet, depth)


def check_good_goal_position(map, goals):
    try:
        for goal in goals:
            if map[goal[1] - 1][goal[0]] == '#' and map[goal[1]][goal[0] + 1] == '#':
                return False
            if map[goal[1] - 1][goal[0]] == '#' and map[goal[1]][goal[0] - 1] == '#':
                return False
            if map[goal[1]][goal[0] - 1] == '#' and map[goal[1] + 1][goal[0]] == '#':
                return False
            if map[goal[1]][goal[0] + 1] == '#' and map[goal[1] + 1][goal[0]] == '#':
                return False
    except:
        return False
    return True

def go2(width, height, limit, no_goals):
    needs_regenerating_map = True
    count = 0
    startSet = None
    while needs_regenerating_map:
        print()
        print("Generating")
        needs_regenerating_map = False
        print("geting room")
        level = getRoom(width - 1, height - 1)
        level = boundRoom(level)

        placeGoalsRes = placeGoals(level, no_goals)
        if placeGoalsRes == []:
            print("continuting")
            continue
        level = placeGoalsRes[0]
        goals = placeGoalsRes[1]

        print("making start set")
        startSet = makeStartSet(level, goals)
        if startSet == None:
            return False
        # Transfer to the normal map
        startSet = readMaps(startSet)

        if check_good_goal_position(startSet[0], goals) == False:
            needs_regenerating_map = True
        #
        # print("finding dd")
        # simple_deadlocks = dd.simple_deadlocks([x[:] for x in startSet[0]])
        # print(startSet[0])
        # for goal in goals:
        #     if goal in simple_deadlocks:
        #         needs_regenerating_map = True
        #     print("finding freeze dd")
        #     if dd.free_deadlocks([x[:] for x in startSet[0]], goal, simple_deadlocks, goals.copy()):
        #         needs_regenerating_map = True

    print("initial_level_done")


    resultSet = startSet
    depth = 1
    while True:
        print("depth??", depth)
        prevSet = resultSet
        # Maybe replace the player in each new level?
        # print("before expanding", resultSet)
        print("expanding")
        resultSet = expand2(resultSet)

        if resultSet == []:
            return False
        if sorted(prevSet) == sorted(resultSet):
            break
        print("expanding finished")
        # print("resultSet is", resultSet)
        # list(dict.fromkeys(mylist))
        if resultSet == False:
            print('place player error')
            return False
        # print("resultSet2 is", resultSet)
        if len(resultSet) == 0:
            break
        depth += 1
        if depth >= limit:
            break

    printMaps(resultSet)
    return (resultSet, depth)


def transfer_to_playable_map(map, myFile):
    for line in map:
        for c in line:
            myFile.write(c)
        myFile.write("\n")


def ranking_1(sol):
    costs = sum(1 for c in sol if c.isupper())
    return costs

def ranking_2(sol):
    last_char = None
    # current_char = None

    i = 0
    for each in sol:
        if each.isupper():
            last_char = each
            break
        else:
            i = i + 1

    changes = 0
    for j in range(i + 1, len(sol)):
        if sol[j].isupper() and sol[j] != last_char:
            last_char = sol[j]
            changes = changes + 1

    return changes

def ranking_3(map):
    board = bd.Board(map=map)
    initial_state = bd.State(board)
    simple_deadlocks = dd.simple_deadlocks(board.getMap())
    solver.pre_compute_heuristic(board, simple_deadlocks)
    heuristic = solver.linear_conficts_with_pre_computed_heuristic(initial_state, None)
    return heuristic







# Needs to save the result!
def generating_map(width, height, no_goals):
    level_generating = True
    solvables = []

    highest_cost = -1
    highest_cost_index = 0


    # To do: value function, choose the most desirable one
    depth = 0
    # When goal is >=5, set depth to < 50 is better.
    while level_generating and depth < 100:
        print("-----------------")
        # level = getRoom(width - 1, height - 1)
        result = go2(width, height, 100, no_goals)
        if result == False:
            continue
        print("!!!!!!!")
        print(len(result[0]))
        for i in range(len(result[0])):
            map = result[0][i]
            # myFile = open("generated_test_gui.txt", 'w')
            # transfer_to_playable_map(map, myFile)
            # myFile.close()
            the_board = bd.Board(map=map)
            print_map(map)
            print("solving!!!")
            solution = solver.aStarSearch_simple_herustic_floor_fill(the_board, None)
            if solution != False and solution != None:
                # costs = ranking_3(solution)
                costs = ranking_3(map)
                print("Has a find a sovable map")
                print(result[1])
                print(costs)
                if costs > highest_cost:
                    highest_cost = costs
                    highest_cost_index = i
            else:
                print("Cannot")

        if highest_cost > no_goals * 2.5:
            level_generating = False
    map = result[0][highest_cost_index]
    # myFile = open("generated_test_gui.txt", 'w')
    # transfer_to_playable_map(map, myFile)
    print("cost", highest_cost)
    return map

