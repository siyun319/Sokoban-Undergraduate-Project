import heapq

import board as bd
import copy


action = {'r': (1, 0),
          'l': (-1, 0),
          'd': (0, 1),
          'u': (0, -1),
          'R': (1, 0),
          'L': (-1, 0),
          'D': (0, 1),
          'U': (0, -1)}


deadlocks_hash = dict()
class PriorityQueue:
    def __init__(self):
        self.queue = []
        self.count = 0

    def is_empty(self):
        return not self.queue

    def push(self, value, priority):
        heapq.heappush(self.queue, (priority, self.count, value))
        self.count += 1

    def pop(self):
        return heapq.heappop(self.queue)[-1]


# PULL the box from the goal square to every possible square and mark all reached squares as visited
def find_all_possible_nodes(board):
    possible_nodes = set()

    initial_state = bd.State(board)
    frontier = []
    frontier.append(initial_state)
    # print(initial_state.boxes)
    # avoid to search the same state multiple times
    exploredSet = set()
    # actions = []

    while frontier:
        if len(frontier) == 0:
            return possible_nodes
        state = frontier.pop()
        possible_nodes.add(state.boxes[0])
        valid_moves = state.get_valid_moves_pull()
        if state.toString() not in exploredSet:
            exploredSet.add(state.toString())
            for the_move in valid_moves:
                if the_move.isupper():
                    is_pull = False
                else:
                    is_pull = True
                new_state = state.update_state_pull(action[the_move], is_pull)
                if (new_state.toString() in exploredSet):
                    continue
                frontier.append(new_state)
    return possible_nodes


def simple_deadlocks(map):
# To do this, one can do the following for every goal square in the level:
# Delete all boxes from the board
# Place a box at the goal square
# PULL the box from the goal square to every possible square and mark all reached squares as visited
    all_nodes = set()
    possible_nodes = set()
    # the_map = copy.deepcopy(map)
    the_map = [x[:] for x in map]
    for i in range(len(the_map)):
        for j in range(len(the_map[i])):
            if the_map[i][j] == 'B':
                the_map[i][j] = ' '
                possible_nodes.add((j,i))
                # all_nodes.add((j, i))
            if the_map[i][j] == 'X':
                the_map[i][j] = '.'
                possible_nodes.add((j, i))
            if the_map[i][j] != '#':
                all_nodes.add((j, i))
    # print(map)

    # Now the map is without boxes.
    print("   ?", the_map)
    board = bd.Board(map = the_map)
    # print("inital box position", board.get_inital_boxes())
    number_goals = len(board.get_inital_goals())
    initial_goals = board.get_inital_goals()
    # print(number_goals)
    for i in range(number_goals):
        board = bd.Board(map = the_map)
        # player on the goal
        # if (initial_goals[i] == '+')
        #     new_map = board.change_map(initial_goals[i], 'X')
        new_map = board.change_map(initial_goals[i], 'X')
        board = bd.Board(map = new_map)
        # print("inital box position", board.get_inital_boxes())
        possible_nodes = possible_nodes.union(find_all_possible_nodes(board))
        # print(find_all_possible_nodes(board))

    # print("-----", sorted(all_nodes - possible_nodes))

    return all_nodes - possible_nodes

#     If there is a wall on the left or on the right side of the box then the box is blocked along this axis
#     If there is a simple deadlock square on both sides (left and right) of the box the box is blocked along this axis
#    If there is a box one the left or right side then this box is blocked if the other box is blocked.
def is_blocked_horizontally(map, pushed_box, simple_deadlocks, boxes):
    # The map index is in reverse order for the position in tuple
    # print("There are boxes:", boxes)
    if pushed_box in boxes:
        boxes.remove(pushed_box)
    if ((pushed_box[0] - 1 >= 0 and  map[pushed_box[1]][pushed_box[0] - 1] == '#') or (pushed_box[0] + 1 < len(map[pushed_box[1]]) and  map[pushed_box[1]][pushed_box[0] + 1] == '#')):
        map[pushed_box[1]][pushed_box[0]] = '#'
        return True
    if ((pushed_box[0] - 1, pushed_box[1]) in simple_deadlocks and (pushed_box[0] + 1, pushed_box[1]) in simple_deadlocks):
        map[pushed_box[1]][pushed_box[0]] = '#'
        return True
    if ((pushed_box[0] - 1, pushed_box[1]) in boxes):
        if is_blocked_vertically(map, (pushed_box[0] - 1, pushed_box[1]), simple_deadlocks, boxes) == True:
            map[pushed_box[1]][pushed_box[0]] = '#'
            # boxes.remove((pushed_box[0] - 1, pushed_box[1]))
            return True
        else:
            return False
    if ((pushed_box[0] + 1, pushed_box[1]) in boxes):
        if is_blocked_vertically(map, (pushed_box[0] + 1, pushed_box[1]), simple_deadlocks, boxes) == True:
            map[pushed_box[1]][pushed_box[0]] = '#'
            # boxes.remove((pushed_box[0] + 1, pushed_box[1]))
            return True
        else:
            return False
    return False



def is_blocked_vertically(map, pushed_box, simple_deadlocks, boxes):
    if pushed_box in boxes:
        boxes.remove(pushed_box)

    if ((pushed_box[1] - 1 >= 0 and pushed_box[0] < len(map[pushed_box[1] - 1]) and map[pushed_box[1] - 1][pushed_box[0]] == '#') or (pushed_box[1] + 1 < len(map) and pushed_box[0] < len(map[pushed_box[1] + 1]) and map[pushed_box[1] + 1][pushed_box[0]] == '#')):
        map[pushed_box[1]][pushed_box[0]] = '#'
        return True
    if ((pushed_box[0], pushed_box[1] - 1) in simple_deadlocks and (pushed_box[0], pushed_box[1] + 1) in simple_deadlocks):
        map[pushed_box[1]][pushed_box[0]] = '#'
        return True
    if ((pushed_box[0], pushed_box[1] - 1) in boxes):
        if is_blocked_horizontally(map, (pushed_box[0], pushed_box[1] - 1), simple_deadlocks, boxes) == True:
            map[pushed_box[1]][pushed_box[0]] = '#'
            return True
        else:
            return False
    if ((pushed_box[0], pushed_box[1] + 1) in boxes):
        if is_blocked_horizontally(map, (pushed_box[0], pushed_box[1] + 1), simple_deadlocks, boxes) == True:
            map[pushed_box[1]][pushed_box[0]] = '#'
            # boxes.remove((pushed_box[0], pushed_box[1] + 1))
            return True
        else:
            return False
    return False

# squares that if a box is pushed into, then the box can never be pushed again.
# Note the pushed_box is the new position of this pushed box.
def free_deadlocks(map, pushed_box, simple_deadlocks, boxes):
    if str(sorted(boxes)) in deadlocks_hash.keys():
        return deadlocks_hash[str(sorted(boxes))]
    if (is_blocked_vertically(map, pushed_box, simple_deadlocks, boxes) and is_blocked_horizontally(map, pushed_box, simple_deadlocks, boxes)):
        deadlocks_hash[str(sorted(boxes))] = True
        return True
    else:
        deadlocks_hash[str(sorted(boxes))] = False
        return False
#     If there is a wall on the left or on the right side of the box then the box is blocked along this axis
#     If there is a simple deadlock square on both sides (left and right) of the box the box is blocked along this axis
#    If there is a box one the left or right side then this box is blocked if the other box is blocked.

# def get_corral(map, boxes):

def initialize():
    global deadlocks_hash
    deadlocks_hash = dict()
