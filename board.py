import sys

# the whole board
global the_map
global level
global map_without_boxes
global map_with_only_walls
global free_square
import copy

import solver

hash = dict()

original_map = []


walls = set()


action = {'r': (1, 0),
          'l': (-1, 0),
          'd': (0, 1),
          'u': (0, -1),
          'R': (1, 0),
          'L': (-1, 0),
          'D': (0, 1),
          'U': (0, -1)}

class Board:
    def __init__(self, filename = None, map = None):
        if map == None:
            file = open(filename, 'r')
        else:
            file = map
        # initialize the map
        # in the form of a 2-D array
        global the_map
        global map_without_boxes
        map_without_boxes = []

        original_map = []

        walls = set()

        global map_with_only_walls
        map_with_only_walls = []

        global boxes_positions_inital
        boxes_positions_inital = []
        global walls_positions_inital
        walls_positions_inital = []
        the_map = []

        board_map = []

        global goals_positions_inital
        goals_positions_inital = []

        global player_position_inital
        player_position_inital = []

        x = 0
        y = 0

        global free_square
        free_square = []

        if_counter_wall = False
        last_first_wall = None
        current_first_wall = None

        for line in file:
            last_first_wall = current_first_wall
            current_first_wall = None
            if_counter_wall = False
            row = []
            row2 = []
            row3 = []
            for c in line:
                if c == "\n":
                    continue
                else:
                    row.append(c)
                    if c == '#':
                        if_counter_wall = True
                        if current_first_wall == None:
                            current_first_wall = (x,y)
                        walls_positions_inital.append((x,y))
                        row2.append(c)
                        row3.append(c)
                        walls.add((y,x))
                    if c == 'B' or c == '$':
                        boxes_positions_inital.append((x,y))
                        row2.append(' ')
                        row3.append(' ')
                    if c == '.':
                        goals_positions_inital.append((x,y))
                        row2.append('.')
                        row3.append(' ')
                    if c == '&' or c == '@':
                        player_position_inital.append((x,y))
                        row2.append(' ')
                        row3.append('&')
                    if c == 'X' or c == '*':
                        boxes_positions_inital.append((x,y))
                        goals_positions_inital.append((x,y))
                        row2.append('.')
                        row3.append(' ')
                    if c == '+':
                        player_position_inital.append((x,y))
                        goals_positions_inital.append((x,y))
                        row2.append(' ')
                        row3.append('+')
                    if c == ' ':
                        row2.append(' ')
                        row3.append(' ')
                    if if_counter_wall :
                        if last_first_wall != None:
                            if x>= last_first_wall[0] and c != '#':
                                free_square.append((x,y))
                        else:
                            if c != '#':
                                free_square.append((x,y))


                    x = x + 1
            x = 0
            y = y + 1

            the_map.append(row)
            map_without_boxes.append(row2)
            map_with_only_walls.append(row3)

            self.original_map = copy.deepcopy(the_map)


    def getSize(self):
        x = 0
        y = len(the_map)
        for row in the_map:
            if len(row) > x:
                x = len(row)

        # 32 is the pixel per image length
        return (x , y)


    # the left-top most corncor is 0
    def getMap(self):
        return the_map

    def get_inital_walls(self):
        return walls_positions_inital

    def get_inital_boxes(self):
        return boxes_positions_inital

    def get_inital_goals(self):
        return goals_positions_inital

    def get_inital_player(self):
        return player_position_inital

    def change_map(self, position, char):
        if the_map[position[1]][position[0]] == '+':
            if the_map[position[1] + 1][position[0]] == ' ':
                the_map[position[1] + 1][position[0]] = '&'
            elif the_map[position[1]][position[0] + 1] == ' ':
                the_map[position[1]][position[0] + 1] = '&'
            elif the_map[position[1] - 1][position[0]] == ' ':
                the_map[position[1] - 1][position[0]] = '&'
            elif the_map[position[1]][position[0] - 1] == ' ':
                the_map[position[1]][position[0] - 1] = '&'
        the_map[position[1]][position[0]] = char
        return the_map

    def get_map_without_boxes(self):
        return map_without_boxes

    def get_free_square(self):
        return free_square

    def get_map_only_walls(self):
        return map_with_only_walls


    # def update_board(self, state):



    # def change_box_position(self):

def get_Manhattan_distance(pointx, pointy):
    return abs(pointx[0] - pointy[0]) + abs(pointx[1] - pointy[1])

# Need to consider the case when box, player on goal!
class State:
    def __init__(self, board = None, new_info = None):
        if board != None:
            self.single_state = (board.get_inital_player(), board.get_inital_boxes(), board.get_inital_goals())
            self.player = board.get_inital_player()
            self.boxes = board.get_inital_boxes()
            self.goals = board.get_inital_goals()
            self.player_normalized = None
            # self.walls = board.get_inital_walls()
        else:
            self.single_state = new_info
            self.player = new_info[0]
            self.boxes = new_info[1]
            self.goals = new_info[2]
            self.player_normalized = None
            # self.walls =

    def toString(self):
        list = (self.player, sorted(self.boxes), sorted(self.goals))
        return str(list)

    def set_normalized_player(self, accessiable_sqaures):
        self.player_normalized = sorted(accessiable_sqaures)[0]

    def toString_normalized(self):
        if self.player_normalized == None:
            # print('forgot to initialize!')
            return None

        list = (self.player_normalized, sorted(self.boxes), sorted(self.goals))
        return str(list)










    def getHashValue(self):
        initial_player = self.player[0]
        left_player_position = (initial_player[0] - 1, initial_player[1])
        current_player_position = (initial_player[0], initial_player[1])

        while left_player_position not in walls_positions_inital and left_player_position not in self.boxes:
            current_player_position = left_player_position
            left_player_position = (left_player_position[0] - 1, left_player_position[1])

        upper_player_position = (current_player_position[0], current_player_position[1] - 1)
        while upper_player_position not in walls_positions_inital and upper_player_position not in self.boxes:
            current_player_position = upper_player_position
            upper_player_position = (upper_player_position[0], upper_player_position[1] - 1)

        return str([current_player_position, self.boxes, self.goals])


    def get_state(self):
        return self.single_state

    def is_end(self):
        return sorted(self.single_state[1]) == sorted(self.single_state[2])

    #  assume the action here is legal (not collide with walls).
    def update_state(self, action):
        if (self.is_legal_action(action)[0] == False):
            return False
        old_player = self.player[0]

        new_player = (old_player[0] + action[0], old_player[1] + action[1])
        pushed_box = None
        # Not push a box
        if (new_player not in self.boxes):
            if (new_player not in self.goals):
                new_info = ([new_player], self.boxes, self.goals)
                if (old_player not in self.goals):
                    the_map[old_player[1]][old_player[0]] = ' '
                else:
                    the_map[old_player[1]][old_player[0]] = '.'
                the_map[new_player[1]][new_player[0]] = '&'
            else:   # new player on the goal
                new_info = ([new_player], self.boxes, self.goals)
                if (old_player not in self.goals):
                    the_map[old_player[1]][old_player[0]] = ' '
                else:
                    the_map[old_player[1]][old_player[0]] = '.'
                # + means player on goal
                the_map[new_player[1]][new_player[0]] = '+'
            return State(None, new_info)
        # push a box
        else:
            new_boxes = []
            for box in self.boxes:
                if box != new_player:
                    new_boxes.append(box)
                else:
                    new_boxes.append((box[0] + action[0], box[1] + action[1]))
                    if (box[0] + action[0], box[1] + action[1]) in self.goals:
                        the_map[box[1] + action[1]][box[0] + action[0]] = 'X'
                    else:
                        the_map[box[1] + action[1]][box[0] + action[0]] = 'B'
            if (old_player not in self.goals):
                the_map[old_player[1]][old_player[0]] = ' '
            else:
                the_map[old_player[1]][old_player[0]] = '.'
            if (new_player not in self.goals):
                the_map[new_player[1]][new_player[0]] = '&'
            else:
                the_map[new_player[1]][new_player[0]] = '+'
            new_info = ([new_player], new_boxes, self.goals)
            return State(None, new_info)


    def update_state_pull(self, action, is_pull):
        if (self.is_legal_action_pull(action)[0] == False):
            return False
        old_player = self.player[0]


        if (old_player in self.goals):
            the_map[old_player[1]][old_player[0]] = '.'

        new_player = (old_player[0] + action[0], old_player[1] + action[1])

        # pull a box
        if self.is_legal_action_pull(action)[1] == True and is_pull:
            old_box = (old_player[0] - action[0], old_player[1] - action[1])
            new_boxes = []
            for box in self.boxes:
                if box != old_box:
                    new_boxes.append(box)
                else:
                    # the old_player position will be the new position of box
                    new_boxes.append((old_box[0] + action[0], old_box[1] + action[1]))
                    if (box[0] + action[0], box[1] + action[1]) in self.goals:
                        the_map[old_box[1] + action[1]][old_box[0] + action[0]] = 'X'
                    else:
                        the_map[old_box[1] + action[1]][old_box[0] + action[0]] = 'B'
                    if (old_box in self.goals):
                        the_map[old_box[1]][old_box[0]] = '.'
                    else:
                        the_map[old_box[1]][old_box[0]] = ' '
            if (new_player not in self.goals):
                the_map[new_player[1]][new_player[0]] = '&'
            else:
                the_map[new_player[1]][new_player[0]] = '+'
            new_info = ([new_player], new_boxes, self.goals)
            return State(None, new_info)
        # Doesn't pull a box
        else:
            if (new_player not in self.boxes):
                if (new_player not in self.goals):
                    new_info = ([new_player], self.boxes, self.goals)
                    if (old_player not in self.goals):
                        the_map[old_player[1]][old_player[0]] = ' '
                    else:
                        the_map[old_player[1]][old_player[0]] = '.'
                    the_map[new_player[1]][new_player[0]] = '&'
                else:   # new player on the goal
                    new_info = ([new_player], self.boxes, self.goals)
                    if (old_player not in self.goals):
                        the_map[old_player[1]][old_player[0]] = ' '
                    else:
                        the_map[old_player[1]][old_player[0]] = '.'
                    # + means player on goal
                    the_map[new_player[1]][new_player[0]] = '+'
                return State(None, new_info)

    def is_legal_action_pull(self, action):
            old_player = self.player[0]
            new_player = (old_player[0] + action[0], old_player[1] + action[1])
            if  (old_player[0] - action[0], old_player[1] - action[1]) not in self.boxes:
                is_pulling = False
            else:
                is_pulling = True

            # we can move to the new position.
            if new_player not in walls_positions_inital and new_player not in self.boxes:
                if is_pulling == True:
                    return (True, True)
                else:
                    return (True, False)
            # New position cannot be in walls or boxes
            else:
                return (False, False)

    # This is subject to if the map is actually bounded by walls.
    def is_legal_action(self, action):
            old_player = self.player[0]
            new_player = (old_player[0] + action[0], old_player[1] + action[1])

            if new_player not in walls_positions_inital:
                if new_player not in self.boxes and new_player not in walls_positions_inital:
                    return (True, False);
                else:
                    # new_player is at the box
                    next_to_the_box = (new_player[0] + action[0], new_player[1] + action[1])
                    if (next_to_the_box not in self.boxes) and (next_to_the_box not in walls_positions_inital):
                        return (True, True)
                    else:
                        return (False, False)
            else:
                return (False, False)

    def get_valid_moves(self):
        valid_moves = []

        if self.is_legal_action(action['r'])[0]:
            if self.is_legal_action(action['r'])[1]:
                valid_moves.append('R')
            else:
                valid_moves.append('r')

        if self.is_legal_action(action['l'])[0]:
            if self.is_legal_action(action['l'])[1]:
                valid_moves.append('L')
            else:
                valid_moves.append('l')

        if self.is_legal_action(action['u'])[0]:
            if self.is_legal_action(action['u'])[1]:
                valid_moves.append('U')
            else:
                valid_moves.append('u')

        if self.is_legal_action(action['d'])[0]:
            if self.is_legal_action(action['d'])[1]:
                valid_moves.append('D')
            else:
                valid_moves.append('d')
        return valid_moves

    def get_valid_moves_pull(self):
        valid_moves = []

        if self.is_legal_action_pull(action['r'])[0]:
            if self.is_legal_action_pull(action['r'])[1]:
                valid_moves.append('R')
                valid_moves.append('r')
            else:
                valid_moves.append('r')

        if self.is_legal_action_pull(action['l'])[0]:
            if self.is_legal_action_pull(action['l'])[1]:
                valid_moves.append('L')
                valid_moves.append('l')
            else:
                valid_moves.append('l')

        if self.is_legal_action_pull(action['u'])[0]:
            if self.is_legal_action_pull(action['u'])[1]:
                valid_moves.append('u')
                valid_moves.append('U')
            else:
                valid_moves.append('u')

        if self.is_legal_action_pull(action['d'])[0]:
            if self.is_legal_action_pull(action['d'])[1]:
                valid_moves.append('d')
                valid_moves.append('D')
            else:
                valid_moves.append('d')
        return valid_moves

    def get_target_box_matrix(self):
        # matrix = {}
        sorted_boxes = sorted(self.boxes)
        sorted_target = sorted(self.goals)
        # for each_target in sorted_target:
        #     matrix[each_target] = {}
        #     for each_box in sorted_boxes:
        #         matrix[each_target][each_box] = get_Manhattan_distance(each_target, each_box)
        # return matrix
        matrix = []
        i = 0
        for each_target in sorted_target:
            matrix.append([])
            for each_box in sorted_boxes:
                matrix[i].append(get_Manhattan_distance(each_target, each_box))
            i += 1
        return matrix


    # # Now the cost the number of pushes required to push each box the each goal
    # Not working well!
    # Again, the similar problem as backout conflict problem!
    # def get_target_box_matrix2(self):
    #     hash = dict()
    #     matrix = []
    #     sorted_boxes = sorted(self.boxes)
    #     sorted_target = sorted(self.goals)
    #     i = 0
    #     for box in sorted_boxes:
    #         matrix.append([])
    #         for goal in sorted_target:
    #             if str((box,goal)) in hash.keys():
    #                 matrix[i].append(hash[str((box,goal))])
    #             else:
    #                 map = [x[:] for x in map_with_only_walls]
    #                 if box == goal:
    #                     matrix[i].append(0)
    #                 else:
    #                     map[box[1]][box[0]] = 'B'
    #                     map[goal[1]][goal[0]] = '.'
    #                     the_board = Board(map = map)
    #                     sol = solver.uniformCostSearch(the_board)
    #                     if sol == False:
    #                         matrix[i].append(1000)
    #                         hash[str((box, goal))] = 1000
    #                     else:
    #                         # print(sum(1 for c in sol if c.isupper()))
    #                         matrix[i].append(sum(1 for c in sol if c.isupper()))
    #                         hash[str((box,goal))] = sum(1 for c in sol if c.isupper())
    #         i += 1
    #     return matrix

    def get_map2(self):
        return the_map


    def get_map(self):
        # map = map_without_boxes
        map = [x[:] for x in map_without_boxes]
        # print(map)
        for box in self.boxes:
            if map[box[1]][box[0]] == '.':
                map[box[1]][box[0]] = 'X'
            else:
                map[box[1]][box[0]] = 'B'

        return map


    def box_surrounded_by_other_boxes(self, pushed_box):
        for box in self.boxes:
            if box == pushed_box:
                continue
            else:
                if box[0] == pushed_box[0]:
                    abs(box[1] - pushed_box[1]) == 1
                    return True
                if box[1] == pushed_box[1]:
                    abs(box[0] - pushed_box[0]) == 1
                    return True
        return False

    def update_state_floor_fill(self, valid_push):
        move = valid_push[0]
        player_position = valid_push[1]
        old_player_position = self.player[0]

        if the_map[old_player_position[1]][old_player_position[0]] == '+':
            the_map[old_player_position[1]][old_player_position[0]] = '.'
        else:
            the_map[old_player_position[1]][old_player_position[0]] = ' '

        if (player_position[0], player_position[1]) in self.goals:
           the_map[player_position[1]][player_position[0]] = '+'
        else:
            the_map[player_position[1]][player_position[0]] = '&'


        direction = action[move]

        if the_map[player_position[1] + 2 * direction[1]][player_position[0] + 2 * direction[0]] == ' ':
            the_map[player_position[1] + 2 * direction[1]][player_position[0] + 2 * direction[0]] = 'B'
        elif the_map[player_position[1] + 2 * direction[1]][player_position[0] + 2 * direction[0]] == '.':
            the_map[player_position[1] + 2 * direction[1]][player_position[0] + 2 * direction[0]] == 'X'
        # elif the_map[player_position[1] + 2 * direction[1]][player_position[0] + 2 * direction[0]] == '#':
        if (player_position[0] + 2 * direction[0], player_position[1] + 2 * direction[1]) in walls_positions_inital or (player_position[0] + 2 * direction[0], player_position[1] + 2 * direction[1]) in self.boxes:
            # return False, False
            return (False, False, False)

        old_box = (player_position[0] + direction[0], player_position[1] + direction[1])
        new_boxes = []
        for the_box in self.boxes:
            if the_box != old_box:
                new_boxes.append(the_box)

        pushed_box = (player_position[0] + 2 * direction[0], player_position[1] + 2 * direction[1])
        new_boxes.append(pushed_box)

        after_pushed_player_position = (player_position[0] + direction[0], player_position[1] + direction[1])

        new_info = ([after_pushed_player_position], new_boxes, self.goals)
        return State(None, new_info), pushed_box, old_box


