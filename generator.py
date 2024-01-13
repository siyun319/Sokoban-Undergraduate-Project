import random
import math

import board as bd
import solver

import time


import multiprocessing
import time

action = {'r': (1, 0),
          'l': (-1, 0),
          'd': (0, 1),
          'u': (0, -1),
          'R': (1, 0),
          'L': (-1, 0),
          'D': (0, 1),
          'U': (0, -1)}

class MapGenerator:
    def __init__(self, x, y, z):
        self.x = x - 1
        self.y = y - 1
        self.z = z
        self.thresh = 15
        self.reGenCount = 0
        self.player_pos = None
        self.first = True


    def genMap(self):
        start_time = time.time()
        print("ReGen:", self.reGenCount)
        self.reGenCount += 1
        print("Generating Map")
        self.map = [["#" for j in range(self.x + 1)] for i in range(self.y + 1)]
        self.dest_point = []
        self.boxes = []
        self.startPos = None

        self.player_pos = None
        # first time try to "pull"
        self.first = True

        self.initBoxes()
        for box in self.dest_point:
            self.randomlyMove(box)

        # box superposition
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                if self.boxes[i] == self.boxes[j]:
                    self.genMap()
                    return

        # box duplicate
        if len(self.boxes) < self.z:
            self.genMap()
            return

        # box collide startPos
        for box in self.boxes:
            if box == self.startPos:
                self.genMap()
                return

        # box initialy sit on the dest pos
        for box in self.boxes:
            for dest_pos in self.dest_point:
                if box == dest_pos:
                    self.genMap()
                    return

        self.putDest()
        self.putBoxes()
        self.putStartPoint()
        self.checkStartDestSuperposition()


    def printMap(self):
        map = self.map.copy()
        for p in self.dest_point:
            map[p[0]][p[1]] = "."
        for p in self.boxes:
            map[p[0]][p[1]] = "B"

        if [self.startPos[0], self.startPos[1]] in  self.dest_point:
            map[self.startPos[0]][self.startPos[1]] = "+"
        else:
            map[self.startPos[0]][self.startPos[1]] = "&"

        counter = 0

        for row in map:
            print(counter, end=" ")
            counter += 1
            for entry in row:
                print(entry, end=" ")
            print()

        print("  ", end="")
        for i in range(self.x + 1):
            print(i, end=" ")
        print()

        return map

    def putStartPoint(self):
        self.setXYOfMap(self.startPos, "&")

    def putDest(self):
        for p in self.dest_point:
            self.map[p[0]][p[1]] = "."

    def putBoxes(self):
        for p in self.boxes:
            self.map[p[0]][p[1]] = "B"

    def checkStartDestSuperposition(self):
        for dest_pos in self.dest_point:
            if dest_pos == self.startPos:
                self.setXYOfMap(dest_pos, "+")

    def setXYOfMap(self, pos, val):
        self.map[pos[0]][pos[1]] = val

    def randomlyMove(self, box: list):
        direction = ["u", "d", "l", "r"]
        move_counter = 0

        curr_box = [box[0], box[1]]
        direct = None
        lastDirect = None

        self.putDest()
        self.putBoxes()

        while move_counter <= self.thresh:
            direct = random.choice(direction)

            # Not allow consecutive oppsite directions
            if lastDirect == "u" and direct == "d":
                continue
            elif lastDirect == "d" and direct == "u":
                continue
            elif lastDirect == "l" and direct == "r":
                continue
            elif lastDirect == "r" and direct == "l":
                continue

            if direct == "u" and curr_box[0] >= 3 and (
                    self.map[curr_box[0] - 1][curr_box[1]] == "#" or self.map[curr_box[0] - 1][curr_box[1]] == " "):

                self.map[curr_box[0] - 1][curr_box[1]] = " "
                self.map[curr_box[0] - 2][curr_box[1]] = " "
                curr_box = [curr_box[0] - 1, curr_box[1]]

                # make room for change push direction
                if lastDirect == "l":
                    self.map[curr_box[0]][curr_box[1] - 1] = " "
                    self.map[curr_box[0] + 1][curr_box[1] - 1] = " "
                elif lastDirect == "r":
                    self.map[curr_box[0]][curr_box[1] + 1] = " "
                    self.map[curr_box[0] + 1][curr_box[1] + 1] = " "

                move_counter += 1

            elif direct == "d" and curr_box[0] <= self.y - 3 and (
                    self.map[curr_box[0] + 1][curr_box[1]] == "#" or self.map[curr_box[0] + 1][curr_box[1]] == " "):
                self.map[curr_box[0] + 1][curr_box[1]] = " "
                self.map[curr_box[0] + 2][curr_box[1]] = " "
                curr_box = [curr_box[0] + 1, curr_box[1]]

                # make room for change push direction
                if lastDirect == "l":
                    self.map[curr_box[0]][curr_box[1] - 1] = " "
                    self.map[curr_box[0] - 1][curr_box[1] - 1] = " "
                elif lastDirect == "r":
                    self.map[curr_box[0]][curr_box[1] + 1] = " "
                    self.map[curr_box[0] - 1][curr_box[1] + 1] = " "

                move_counter += 1

            elif direct == "l" and curr_box[1] >= 3 and (
                    self.map[curr_box[0]][curr_box[1] - 1] == "#" or self.map[curr_box[0]][curr_box[1] - 1] == " "):
                self.map[curr_box[0]][curr_box[1] - 1] = " "
                self.map[curr_box[0]][curr_box[1] - 2] = " "
                curr_box = [curr_box[0], curr_box[1] - 1]
                move_counter += 1

                # make room for change push direction
                if lastDirect == "u":
                    self.map[curr_box[0] - 1][curr_box[1]] = " "
                    self.map[curr_box[0] - 1][curr_box[1] + 1] = " "
                elif lastDirect == "d":
                    self.map[curr_box[0] + 1][curr_box[1]] = " "
                    self.map[curr_box[0] + 1][curr_box[1] + 1] = " "

            elif direct == "r" and curr_box[1] <= self.x - 3 and (
                    self.map[curr_box[0]][curr_box[1] + 1] == "#" or self.map[curr_box[0]][curr_box[1] + 1] == " "):
                self.map[curr_box[0]][curr_box[1] + 1] = " "
                self.map[curr_box[0]][curr_box[1] + 2] = " "
                curr_box = [curr_box[0], curr_box[1] + 1]

                # make room for change push direction
                if lastDirect == "u":
                    self.map[curr_box[0] - 1][curr_box[1]] = " "
                    self.map[curr_box[0] - 1][curr_box[1] - 1] = " "
                elif lastDirect == "d":
                    self.map[curr_box[0] + 1][curr_box[1]] = " "
                    self.map[curr_box[0] + 1][curr_box[1] - 1] = " "

                move_counter += 1

            # self.printMap()
            # print()

            lastDirect = direct

            target_pos = None

        # print("Before connect boxes")
        # print("PlayerPos",self.player_pos)
        # print()

        if self.player_pos is not None:
            if direct == "u":
                target_pos = [curr_box[0] - 1, curr_box[1]]
            elif direct == "d":
                target_pos = [curr_box[0] + 1, curr_box[1]]
            elif direct == "l":
                target_pos = [curr_box[0], curr_box[1] - 1]
            elif direct == "r":
                target_pos = [curr_box[0], curr_box[1] + 1]

            # print("Target Pos:",target_pos)
            if direct == "r" and target_pos[1] > self.player_pos[1] and (
                    direct == "l" and target_pos[1] < self.player_pos[1]):
                if target_pos[1] > self.player_pos[1]:
                    while target_pos[1] > self.player_pos[1]:
                        self.player_pos = [self.player_pos[0], self.player_pos[1] + 1]
                        self.setXYOfMap(self.player_pos, " ")
                else:
                    while target_pos[1] < self.player_pos[1]:
                        self.player_pos = [self.player_pos[0], self.player_pos[1] - 1]
                        self.setXYOfMap(self.player_pos, " ")

                if target_pos[0] > self.player_pos[0]:
                    # make room for change direction

                    while target_pos[0] > self.player_pos[0]:
                        self.player_pos = [self.player_pos[0] + 1, self.player_pos[1]]
                        self.setXYOfMap(self.player_pos, " ")
                else:
                    while target_pos[0] < self.player_pos[0]:
                        self.player_pos = [self.player_pos[0] - 1, self.player_pos[1]]
                        self.setXYOfMap(self.player_pos, " ")
            else:
                if target_pos[0] > self.player_pos[0]:
                    # make room for change direction

                    while target_pos[0] > self.player_pos[0]:
                        self.player_pos = [self.player_pos[0] + 1, self.player_pos[1]]
                        self.setXYOfMap(self.player_pos, " ")
                else:
                    while target_pos[0] < self.player_pos[0]:
                        self.player_pos = [self.player_pos[0] - 1, self.player_pos[1]]
                        self.setXYOfMap(self.player_pos, " ")

                if target_pos[1] > self.player_pos[1]:
                    while target_pos[1] > self.player_pos[1]:
                        self.player_pos = [self.player_pos[0], self.player_pos[1] + 1]
                        self.setXYOfMap(self.player_pos, " ")
                else:
                    while target_pos[1] < self.player_pos[1]:
                        self.player_pos = [self.player_pos[0], self.player_pos[1] - 1]
                        self.setXYOfMap(self.player_pos, " ")
        # print("After connect boxes")
        # print("PlayerPos", self.player_pos)
        # print("TargetPos", target_pos)
        # print("CurrBox", curr_box)
        # self.printMap()
        # print()

        if direct == "u":
            self.map[curr_box[0] - 1][curr_box[1]] = " "
            self.player_pos = [curr_box[0] - 1, curr_box[1]]
        elif direct == "d":
            self.map[curr_box[0] + 1][curr_box[1]] = " "
            self.player_pos = [curr_box[0] + 1, curr_box[1]]
        elif direct == "l":
            self.map[curr_box[0]][curr_box[1] - 1] = " "
            self.player_pos = [curr_box[0], curr_box[1] - 1]
        elif direct == "r":
            self.map[curr_box[0]][curr_box[1] + 1] = " "
            self.player_pos = [curr_box[0], curr_box[1] + 1]

        # print("After update playerPos")
        # self.printMap()

        if self.first:
            self.startPos = [self.player_pos[0], self.player_pos[1]]
            self.setXYOfMap(self.startPos, "&")
            self.first = False
        # print("PlayerPos",self.player_pos)
        self.boxes.append(curr_box)

    def initBoxes(self):
        counter = 0

        while counter < self.z:
            useThisPoint = True

            bx = random.randint(2, self.x - 2)
            by = random.randint(2, self.y - 2)

            # exclude duplicate point
            for point in self.dest_point:
                if [by, bx] == point:
                    useThisPoint = False

            if useThisPoint:
                self.dest_point.append([by, bx])
                self.map[by][bx] = "."
                counter += 1
            else:
                continue


generating = True


def write_to_file(map, myFile):
    for line in map:
        for c in line:
            myFile.write(c)
        myFile.write("\n")

def get_critical_path(map, sol):
    the_board = bd.Board(map = map)
    initial_player = the_board.get_inital_player()[0]
    path = []
    for step in sol:
        path.append((initial_player[1] + action[step][1], initial_player[0] + action[step][0]))
        initial_player = (initial_player[0] + action[step][0], initial_player[1] + action[step][1])

    print(path)
    return path

def add_walls(map, path):
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j] == ' ' and (i, j) not in path:
                map[i][j] = '#'
    return map

map = None
sol = None

manager = multiprocessing.Manager()
return_dict = manager.dict()
while generating:
    MG = MapGenerator(12, 12, 8)

    badMap = True

    p = multiprocessing.Process(target=MG.genMap())
    p.start()
    p.join(10)
    # result = MG.genMap()
    if p.is_alive():
        p.terminate()
        print("Time exceed 60s")
        continue
    # if result == False:
    #     continue
    # map = MG.map
    map = MG.printMap()
    print("DEST POS", MG.dest_point)
    print("BOXES", MG.boxes)
    print("START POS", MG.startPos)

    the_board = bd.Board(map = map)
    sol = solver.aStarSearch(the_board)
    if sol == False:
        continue
    cost = sum(1 for c in sol if c.isupper())
    if cost > 30:
        myFile = open("generated1.txt", 'w')
        write_to_file(map, myFile)
        myFile.close()
        print("sol is", sol)
        generating = False
map = add_walls(map, get_critical_path(map, sol))
myFile = open("generated1.txt", 'w')
write_to_file(map, myFile)
myFile.close()