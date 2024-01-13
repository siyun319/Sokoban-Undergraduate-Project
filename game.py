import pygame
import sys
from pygame.locals import *
from Widgets import *
from threading import Thread
import re
import os, os.path

import level_generation
from os import listdir
from os.path import isfile, join

from tkinter import Tcl


import glob

import multiprocessing as mp

# 0 - Menu, 1 - Player choose level, 2 - Gaming, 3 - Finish Level,
# 4 - AI Solver choose level, 5 - AI Solver solving
# 7 - Generator input Page, 8 - Generating Page
LAYER = 0

FPS = 60

LEVEL_NUM = 13

LEVEL_SELECTED = -1


action = {'r': (1, 0),
          'l': (-1, 0),
          'd': (0, 1),
          'u': (0, -1),
          'R': (1, 0),
          'L': (-1, 0),
          'D': (0, 1),
          'U': (0, -1)}


import time

import board as bd

start_time = time.time()

import solver

# the whole board
global the_map
global level

global the_board

isGUI_on = False

is_Pulling = False

import numpy


from munkres import Munkres


def load_images():
    global wall_image
    wall_image = pygame.image.load('images/wall_2.png')
    global box_docked_image
    box_docked_image = pygame.image.load('images/box_docked.png')
    global box_image
    box_image = pygame.image.load('images/box.png')
    global dock_image
    dock_image = pygame.image.load('images/dock.png')
    global floor_image
    floor_image = pygame.image.load('images/floor.png')
    global worker_docker_image
    worker_docker_image = pygame.image.load('images/worker_dock.png')
    global worker_image
    worker_image = pygame.image.load('images/worker.png')

    wall_image = pygame.transform.scale(wall_image, (50, 50))
    box_docked_image = pygame.transform.scale(box_docked_image, (50, 50))
    box_image = pygame.transform.scale(box_image, (50, 50))
    dock_image = pygame.transform.scale(dock_image, (50, 50))
    floor_image = pygame.transform.scale(floor_image, (50, 50))
    worker_image = pygame.transform.scale(worker_image, (50, 50))
    worker_docker_image = pygame.transform.scale(worker_docker_image, (50, 50))

def scale_image(size):
    global wall_image
    global box_docked_image
    global box_image
    global dock_image
    global floor_image
    global worker_docker_image
    global worker_image
    wall_image = pygame.transform.scale(wall_image, (size, size))
    box_docked_image = pygame.transform.scale(box_docked_image, (size, size))
    box_image = pygame.transform.scale(box_image, (size, size))
    dock_image = pygame.transform.scale(dock_image, (size, size))
    floor_image = pygame.transform.scale(floor_image, (size, size))
    worker_image = pygame.transform.scale(worker_image, (size, size))
    worker_docker_image = pygame.transform.scale(worker_docker_image, (size, size))

def gaming(screen, level):
    global LAYER
    global FILE_CHOSEN

    print(f"Now playing level {level}")

    fileName = FILE_CHOSEN
    the_board = bd.Board(filename=fileName)

    state = bd.State(the_board)

    back_btn = Button("Back", (4 * SCREEN_X // 5, SCREEN_Y // 10), screen)

    # debug_btn = Button("DEBUG BUTTON", (SCREEN_X // 2, SCREEN_Y // 2), screen)

    # WIN FLAG
    win = False

    while LAYER == 2:
        if win:
            LAYER = 3
            return

        screen.fill(WHITE)
        back_btn.draw()

        # COMMENT THIS LINE IN THE RELEASE VERSION
        # debug_btn.draw()
        map_x = the_board.getSize()[0]
        map_y = the_board.getSize()[1]


        interval_x = (SCREEN_X - 200) // map_x
        interval_y = (SCREEN_Y - 200) // map_y


        size = min((SCREEN_X - 200) // map_x, (SCREEN_Y - 200) // map_y)
        scale_image(size)

        centre_x = 500
        centre_y = 450

        x = centre_x - map_x // 2 * size
        y = centre_y - map_y // 2 * size

        the_map = the_board.getMap()

        if_counter_wall = False
        last_first_wall = None
        current_first_wall = None
        print("box", sorted(state.boxes))
        print("goal",sorted(state.goals))

        for row in the_map:
            if_counter_wall = False
            last_first_wall = current_first_wall
            current_first_wall = None
            for char in row:
                if char == ' ':
                    if last_first_wall != None:
                        if x >= last_first_wall[0] and if_counter_wall:
                            screen.blit(floor_image, (x, y))
                elif char == '#':
                    if current_first_wall == None:
                        current_first_wall = (x,y)
                    if_counter_wall = True
                    screen.blit(wall_image, (x, y))
                elif char == '&':
                    screen.blit(worker_image, (x, y))
                elif char == 'B':
                    screen.blit(box_image, (x, y))
                elif char == 'X':
                    screen.blit(box_docked_image, (x, y))
                elif char == '.':
                    screen.blit(dock_image, (x, y))
                # worker on the goal
                elif char == '+':
                    screen.blit(worker_docker_image, (x, y))

                x = x + size
            x = centre_x - map_x // 2 * size
            y = y + size

        if state.is_end():
            win = True

        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    if (is_Pulling == False):
                             if state.is_legal_action(action['l'])[0]:
                                             state = state.update_state(action['l'])
                    else:
                             if state.is_legal_action_pull(action['l'])[0]:
                                             state = state.update_state_pull(action['l'])
                elif event.key == pygame.K_RIGHT:
                    if (is_Pulling == False):
                             if state.is_legal_action(action['r'])[0]:
                                             state = state.update_state(action['r'])
                    else:
                             if state.is_legal_action_pull(action['r'])[0]:
                                             state = state.update_state_pull(action['r'])
                elif event.key == pygame.K_DOWN:
                    if (is_Pulling == False):
                             if state.is_legal_action(action['d'])[0]:
                                             state = state.update_state(action['d'])
                    else:
                             if state.is_legal_action_pull(action['d'])[0]:
                                             state = state.update_state_pull(action['d'])
                elif event.key == pygame.K_UP:
                    if (is_Pulling == False):
                             if state.is_legal_action(action['u'])[0]:
                                             state = state.update_state(action['u'])
                    else:
                             if state.is_legal_action_pull(action['u'])[0]:
                                             state = state.update_state_pull(action['u'])
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if is_Pulling == False:
                    print(state.get_valid_moves())
                else:
                    print(state.get_valid_moves_pull())
                # print(state.toString())
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                if back_btn.button_rect.collidepoint(pygame.mouse.get_pos()):

                    LAYER = 9

        pygame.display.flip()
        clock.tick(FPS)



def player_choose_folder(screen):
    global LAYER
    global LEVEL_NUM
    global LEVEL_SELECTED
    global FOLDER_SELECTED

    title = Label("Choose and play", (SCREEN_X // 2, SCREEN_Y // 7), screen)

    curr_page = 0
    max_page = LEVEL_NUM // 9

    upPageBtn = Button("Page Up", (3 * SCREEN_X // 5, 4 * SCREEN_Y // 5), screen)
    downPageBtn = Button("Page Down", (3 * SCREEN_X // 5 + 170, 4 * SCREEN_Y // 5), screen)
    back_btn = Button("Back", (4 * SCREEN_X // 5, SCREEN_Y // 10), screen)

    while LAYER == 9:
        screen.fill(WHITE)

        level_btns = []

        level_count = len([name for name in glob.glob('levels/*[a-z]*[0-9].*')])

        onlyfolders = sorted([x[0][7:] for x in os.walk('levels/')])
        # print("only folers", onlyfolders)
        foler_count = len(onlyfolders) - 1
        # onlyfiles = sorted([f[0:f.index('.')] for f in glob.glob('levels/*[a-z][0-9].*')])
        # print("levels---", onlyfiles)
        # print("level_count", level_count)
        LEVEL_NUM = level_count
        for i in range(curr_page * 9, min(curr_page * 9 + 9, foler_count)):
            j = i % 9
            btn = Button(onlyfolders[i + 1], ((1 + j % 3) * SCREEN_X // 4, (1.3 + j // 3) * SCREEN_Y // 5), screen,
                         i + 1)
            btn.draw()
            level_btns.append(btn)

        title.draw()
        back_btn.draw()
        upPageBtn.draw()
        downPageBtn.draw()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                for btn in level_btns:
                    if btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                        LAYER = 1
                        FOLDER_SELECTED = btn.text

                if upPageBtn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    if curr_page > 0:
                        print("Page Up")
                        curr_page -= 1

                if downPageBtn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    if curr_page < max_page:
                        print("Page Down")
                        curr_page += 1

                if back_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    LAYER = 0

        pygame.display.flip()
        clock.tick(FPS)

def ai_choose_folder(screen):
    global LAYER
    global LEVEL_NUM
    global LEVEL_SELECTED
    global FOLDER_SELECTED

    title = Label("Choose and play", (SCREEN_X // 2, SCREEN_Y // 7), screen)

    curr_page = 0
    max_page = LEVEL_NUM // 9

    upPageBtn = Button("Page Up", (3 * SCREEN_X // 5, 4 * SCREEN_Y // 5), screen)
    downPageBtn = Button("Page Down", (3 * SCREEN_X // 5 + 170, 4 * SCREEN_Y // 5), screen)
    back_btn = Button("Back", (4 * SCREEN_X // 5, SCREEN_Y // 10), screen)

    while LAYER == 10:
        screen.fill(WHITE)

        level_btns = []

        level_count = len([name for name in glob.glob('levels/*[a-z]*[0-9].*')])

        onlyfolders = sorted([x[0][7:] for x in os.walk('levels/')])
        # print("only folers", onlyfolders)
        foler_count = len(onlyfolders) - 1
        # onlyfiles = sorted([f[0:f.index('.')] for f in glob.glob('levels/*[a-z][0-9].*')])
        # print("levels---", onlyfiles)
        # print("level_count", level_count)
        LEVEL_NUM = level_count
        for i in range(curr_page * 9, min(curr_page * 9 + 9, foler_count)):
            j = i % 9
            btn = Button(onlyfolders[i + 1], ((1 + j % 3) * SCREEN_X // 4, (1.3 + j // 3) * SCREEN_Y // 5), screen,
                         i + 1)
            btn.draw()
            level_btns.append(btn)

        title.draw()
        back_btn.draw()
        upPageBtn.draw()
        downPageBtn.draw()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                for btn in level_btns:
                    if btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                        LAYER = 4
                        FOLDER_SELECTED = btn.text

                if upPageBtn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    if curr_page > 0:
                        print("Page Up")
                        curr_page -= 1

                if downPageBtn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    if curr_page < max_page:
                        print("Page Down")
                        curr_page += 1

                if back_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    LAYER = 0

        pygame.display.flip()
        clock.tick(FPS)

def player_choose_level(screen):
    global LAYER
    global LEVEL_NUM
    global LEVEL_SELECTED
    global FOLDER_SELECTED
    global FILE_CHOSEN
    title = Label("Choose and play", (SCREEN_X // 2, SCREEN_Y // 7), screen)

    curr_page = 0


    upPageBtn = Button("Page Up", (3 * SCREEN_X // 5, 4 * SCREEN_Y // 5), screen)
    downPageBtn = Button("Page Down", (3 * SCREEN_X // 5 + 170, 4 * SCREEN_Y // 5), screen)
    back_btn = Button("Back", (4 * SCREEN_X // 5, SCREEN_Y // 10), screen)

    while LAYER == 1:
        screen.fill(WHITE)

        level_btns = []

        string = 'levels/' + FOLDER_SELECTED + '/*[a-z]*[0-9].*'
        level_count = len([name for name in glob.glob('levels/' + FOLDER_SELECTED + '/*[a-z]*[0-9].*')])


        onlyfiles = [os.path.basename(f)[0:os.path.basename(f).index('.')] for f in glob.glob('levels/' + FOLDER_SELECTED + '/*[a-z]*[0-9].*')]
        digit_index = re.search(r"\d", onlyfiles[0]).start()
        onlyfiles.sort(key=lambda s: int(s[digit_index:]))
        files_path = [f for f in glob.glob('levels/' + FOLDER_SELECTED + '/*[a-z]*[0-9].*')]
        digit_index = re.search(r"\d", files_path[0]).start()
        digit_end_index = re.search(r"\d", files_path[0]).end()
        files_path.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])
        # files_path.sort(key=lambda s: int(s[digit_index:digit_end_index]))
        # print(files_path)
        # print("levels---", onlyfiles)
        # print("level_count", level_count)
        LEVEL_NUM = level_count
        max_page = LEVEL_NUM // 9
        for i in range(curr_page * 9, min(curr_page * 9 + 9, LEVEL_NUM)):
            j = i % 9
            btn = Button(onlyfiles[i], ((1 + j % 3) * SCREEN_X // 4, (1.3 + j // 3) * SCREEN_Y // 5), screen,
                         i + 1)
            btn.draw()
            level_btns.append(btn)

        title.draw()
        back_btn.draw()
        upPageBtn.draw()
        downPageBtn.draw()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                for btn in level_btns:
                    if btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                        LAYER = 2
                        FILE_CHOSEN = files_path[btn.val - 1]
                        LEVEL_SELECTED = btn.val

                if upPageBtn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    if curr_page > 0:
                        print("Page Up")
                        curr_page -= 1

                if downPageBtn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    if curr_page < max_page:
                        print("Page Down")
                        curr_page += 1

                if back_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    LAYER = 9

        pygame.display.flip()
        clock.tick(FPS)


def menu_page(screen):
    global LAYER

    title = Label("Sokoban", (SCREEN_X // 2, SCREEN_Y // 7), screen)

    def startBtnOnclicked():
        global LAYER
        LAYER = 9

    start_btn = Button("Start the game", (SCREEN_X // 2, 3 * SCREEN_Y // 7), screen, callback=startBtnOnclicked)

    def aiSolverBtn():
        global LAYER
        LAYER = 10

    ai_solver_btn = Button("AI Solver", (SCREEN_X // 2, 4 * SCREEN_Y // 7), screen, callback=aiSolverBtn)

    def genMapBtn():
        global LAYER
        LAYER = 7

    gen_map_btn = Button("Generate new maps", (SCREEN_X // 2, 5 * SCREEN_Y // 7), screen, callback=genMapBtn)

    while LAYER == 0:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN and start_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                start_btn.onClick()
            elif event.type == MOUSEBUTTONDOWN and ai_solver_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                ai_solver_btn.onClick()
            elif event.type == MOUSEBUTTONDOWN and gen_map_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                gen_map_btn.onClick()

        screen.fill(WHITE)

        title.draw()

        start_btn.draw()
        ai_solver_btn.draw()
        gen_map_btn.draw()

        pygame.display.flip()
        clock.tick(FPS)


def winning_page(screen):
    global LAYER
    global LEVEL_SELECTED
    global FOLDER_SELECTED
    global FILE_CHOSEN
    files_path = sorted([f for f in glob.glob('levels/' + FOLDER_SELECTED + '/*[a-z]*[0-9].*')])

    info1 = Button("You've solved the Game!", (SCREEN_X // 2, 2 * SCREEN_Y // 5), screen)
    info2 = Button("Click to progress to the next level", (SCREEN_X // 2, 2 * SCREEN_Y // 5 + 50), screen)
    back_btn = Button("Back", (4 * SCREEN_X // 5, SCREEN_Y // 10), screen)

    while LAYER == 3:
        screen.fill(WHITE)
        info1.draw()
        info2.draw()
        back_btn.draw()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                if info1.button_rect.collidepoint(pygame.mouse.get_pos()) or \
                        info2.button_rect.collidepoint(pygame.mouse.get_pos()):
                    LEVEL_SELECTED += 1
                    FILE_CHOSEN = files_path[LEVEL_SELECTED - 1]


                    LAYER = 2
                    return

                if back_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    LAYER = 1

        pygame.display.flip()
        clock.tick(FPS)


def ai_solver_choose_level(screen):
    global LAYER
    global LEVEL_NUM
    global LEVEL_SELECTED
    global FOLDER_SELECTED
    global FILE_CHOSEN

    global paramter_dict
    paramter_dict = dict()

    title = Label("Choose and solve", (SCREEN_X // 2, SCREEN_Y // 7), screen)


    curr_page = 0
    max_page = LEVEL_NUM // 9

    upPageBtn = Button("Page Up", (3 * SCREEN_X // 5, 4 * SCREEN_Y // 5), screen)
    downPageBtn = Button("Page Down", (3 * SCREEN_X // 5 + 170, 4 * SCREEN_Y // 5), screen)
    back_btn = Button("Back", (4 * SCREEN_X // 5, SCREEN_Y // 10), screen)

    while LAYER == 4:
        screen.fill(WHITE)

        level_btns = []

        string = 'levels/' + FOLDER_SELECTED + '/*[a-z]*[0-9].*'
        level_count = len([name for name in glob.glob('levels/' + FOLDER_SELECTED + '/*[a-z]*[0-9].*')])

        onlyfiles = [os.path.basename(f)[0:os.path.basename(f).index('.')] for f in glob.glob('levels/' + FOLDER_SELECTED + '/*[a-z]*[0-9].*')]
        digit_index = re.search(r"\d", onlyfiles[0]).start()
        onlyfiles.sort(key=lambda s: int(s[digit_index:]))
        files_path = [f for f in glob.glob('levels/' + FOLDER_SELECTED + '/*[a-z]*[0-9].*')]
        digit_index = re.search(r"\d", files_path[0]).start()
        digit_end_index = re.search(r"\d", files_path[0]).end()
        files_path.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

        LEVEL_NUM = level_count
        max_page = LEVEL_NUM // 9
        for i in range(curr_page * 9, min(curr_page * 9 + 9, LEVEL_NUM)):
            j = i % 9
            btn = Button(onlyfiles[i], ((1 + j % 3) * SCREEN_X // 4, (1.3 + j // 3) * SCREEN_Y // 5), screen,
                         i + 1)
            btn.draw()
            level_btns.append(btn)

        title.draw()
        back_btn.draw()
        upPageBtn.draw()
        downPageBtn.draw()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                for btn in level_btns:
                    if btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                        LAYER = 11
                        FILE_CHOSEN = files_path[btn.val - 1]
                        LEVEL_SELECTED = btn.val

                if upPageBtn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    if curr_page > 0:
                        print("Page Up")
                        curr_page -= 1

                if downPageBtn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    if curr_page < max_page:
                        print("Page Down")
                        curr_page += 1

                if back_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    LAYER = 0

        pygame.display.flip()
        clock.tick(FPS)

def transition_page_algorithm_selection(screen):
    global LAYER

    info1 = Label("Please specify the algorithm you want to use", (500, SCREEN_Y // 7), screen)
    while LAYER == 11:
        screen.fill(WHITE)
        info1.draw()
        pygame.display.flip()
        print("please choose a algorithm: DFS/BFS/UCS/A_star/Greedy/Floor_fill")
        paramter_dict['algorithm'] = input()
        print("please choose a cost function: potential_pushing/encourages_pushing")
        paramter_dict['cost'] = input()
        print("please choose a type of  heuristic function: pre computed / dynamic")
        paramter_dict['heuristic'] = input()
        if paramter_dict['heuristic'] == "dynamic":
            print("do you want to use min-match algorithm?")
            if input().lower() == 'yes':
                paramter_dict['min_match'] = True
            else:
                paramter_dict['min_match'] = False


            print("linear conflicts?: yes/no")
            if input().lower() == 'yes':
                paramter_dict['linear_conflict'] = True
            else:
                paramter_dict['linear_conflict'] = False
        print("please tell if you want deadlock detection: yes/ no")
        if input().lower() == 'yes':
            paramter_dict['dd_detection'] = True
        else:
            paramter_dict['dd_detection'] = False

        print("moving ordering?: yes/no")
        if input().lower() == 'yes':
            paramter_dict['move_ordering'] = True
        else:
            paramter_dict['move_ordering'] = False
        print("you input are", paramter_dict)
        time.sleep(0.3)
        LAYER = 5



def solving_page(screen):
    global LAYER
    global LEVEL_SELECTED
    global FILE_CHOSEN

    global paramter_dict
    print("level_selected is", LEVEL_SELECTED)

    fileName = FILE_CHOSEN



    info1 = Label("Result:", (SCREEN_X // 5, SCREEN_Y // 7), screen)
    info2 = Label("Solving....", (SCREEN_X // 5 + 100, 2 * SCREEN_Y // 7), screen)
    back_btn = Button("Back", (4 * SCREEN_X // 5, SCREEN_Y // 10), screen)

    # solution_text = Label("Solution: XXXXXXXX", (SCREEN_X // 2, SCREEN_Y // 2), screen)

    debug_button = Button("DEBUG BUTTON", (SCREEN_X // 2, SCREEN_Y // 2), screen)

    solved = False

    sol = None

    sol_index = 0

    while LAYER == 5:
        screen.fill(WHITE)
        if solved:
            info1.draw()
            back_btn.draw()

            # For demostration

            map_x = the_board.getSize()[0]
            map_y = the_board.getSize()[1]

            interval_x = (SCREEN_X - 200) // map_x
            interval_y = (SCREEN_Y - 200) // map_y

            size = min((SCREEN_X - 200) // map_x, (SCREEN_Y - 200) // map_y)
            scale_image(size)

            centre_x = 500
            centre_y = 450

            x = centre_x - map_x // 2 * size
            y = centre_y - map_y // 2 * size

            the_map = initial_board.getMap()

            if_counter_wall = False
            last_first_wall = None
            current_first_wall = None

            row_index = 0
            for row in the_map:
                char_index = 0
                if_counter_wall = False
                last_first_wall = current_first_wall
                current_first_wall = None
                for char in row:
                    if char == ' ':
                        if last_first_wall != None:
                            if x >= last_first_wall[0] and if_counter_wall:
                                screen.blit(floor_image, (x, y))
                    elif char == '#':
                        if current_first_wall == None:
                            current_first_wall = (x, y)
                        if_counter_wall = True
                        screen.blit(wall_image, (x, y))
                    elif char == '&':
                        screen.blit(worker_image, (x, y))
                    elif char == 'B':
                        screen.blit(box_image, (x, y))
                    elif char == 'X':
                        screen.blit(box_docked_image, (x, y))
                    elif char == '.':
                        screen.blit(dock_image, (x, y))
                    # worker on the goal
                    elif char == '+':
                        screen.blit(worker_docker_image, (x, y))
                    char_index = char_index + 1
                    x = x + size
                x = centre_x - map_x // 2 * size
                y = y + size
                row_index = row_index + 1

            time.sleep(0.1)
            if sol_index < len(sol):
                state = state.update_state(action[sol[sol_index]])
                # player_pos = state.player[0]

            sol_index = sol_index + 1



            # ----For outputing the solution string
            # no_lines = len(sol) // 15
            # sol_dict = dict()
            # for i in range(no_lines):
            #     if i + 15 >= len(sol):
            #         line = str(sol[i * 15:len(sol)])
            #     else:
            #         line = str(sol[i * 15:i * 15 + 15])
            #     sol_dict[i] = line
            #     solution_text = small_Label(line, (SCREEN_X // 5 + 200, SCREEN_Y // 7 * 2 + i * 50), screen)
            #     solution_text.draw()
        else:
            info1.draw()
            info2.draw()
            back_btn.draw()
            pygame.display.flip()
            fileName = FILE_CHOSEN
            if sol == None:
                the_board = bd.Board(filename=fileName)
                state = bd.State(the_board)
                initial_board = bd.Board(filename=fileName)
                sol = solver.search(the_board, paramter_dict)

                # manager = mp.Manager()
                # sol1_dict = manager.dict()
                # sol2_dict = manager.dict()
                # solver1 = mp.Process(target=solver.aStarSearch, args=(the_board, sol1_dict))
                # solver2 = mp.Process(target=solver.aStarSearch_simple_herustic, args=(the_board,sol2_dict))
                #
                # solver1.start()
                # solver2.start()
                if sol == False:
                    info1 = Label("Result: Not found!", (SCREEN_X // 5, SCREEN_Y // 7), screen)
                    print(FILE_CHOSEN)
                    print(the_board.getMap())
                else:
                    # Draw the solution!
                    solved = True
                    initial_board = bd.Board(filename=fileName)
        # debug_button.draw()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                if debug_button.button_rect.collidepoint(pygame.mouse.get_pos()):
                    solved = True

                if back_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    LAYER = 4

        pygame.display.flip()
        clock.tick(FPS)


def gen_map_page(screen):
    global LAYER

    title = Label("Map Generator", (100 + SCREEN_X // 5, SCREEN_Y // 7), screen)
    back_btn = Button("Back", (4 * SCREEN_X // 5, SCREEN_Y // 10), screen)

    width_info = Label("width:", (SCREEN_X // 5, 2 * SCREEN_Y // 6), screen)
    height_info = Label("height:", (SCREEN_X // 5, 3 * SCREEN_Y // 6), screen)
    goal_info = Label("#goals:", (SCREEN_X // 5, 4 * SCREEN_Y // 6), screen)

    width_input = Edit((2 * SCREEN_X // 5, 2 * SCREEN_Y // 6), screen)
    height_input = Edit((2 * SCREEN_X // 5, 3 * SCREEN_Y // 6), screen)
    goal_input = Edit((2 * SCREEN_X // 5, 4 * SCREEN_Y // 6), screen)

    done_btn = Button("Done!", (4 * SCREEN_X // 5, 4 * SCREEN_Y // 6), screen)

    selected = -1

    inputList = [width_input, height_input, goal_input]
    textInputList = ["", "", ""]

    while LAYER == 7:
        screen.fill(WHITE)

        title.draw()
        back_btn.draw()

        width_info.draw()
        height_info.draw()
        goal_info.draw()

        width_input.draw()
        height_input.draw()
        goal_input.draw()

        done_btn.draw()

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:

                # Page exit
                if done_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    LAYER = 8
                    return textInputList

                if width_input.rect.collidepoint(pygame.mouse.get_pos()):
                    inputList[selected].setColor(GREY)
                    selected = 0
                    width_input.setColor(GREEN)
                elif height_input.rect.collidepoint(pygame.mouse.get_pos()):
                    inputList[selected].setColor(GREY)
                    selected = 1
                    height_input.setColor(GREEN)
                elif goal_input.rect.collidepoint(pygame.mouse.get_pos()):
                    inputList[selected].setColor(GREY)
                    selected = 2
                    goal_input.setColor(GREEN)
                else:
                    inputList[selected].setColor(GREY)
                    selected = -1

                if back_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    LAYER = 0
                    return [0, 0, 0]

            if selected != -1:
                if event.type == KEYDOWN:
                    if chr(event.key).isdigit():
                        textInputList[selected] += chr(event.key)
                    elif event.key == K_BACKSPACE:
                        textInputList[selected] = textInputList[selected][:-1]

        width_input.set_text(textInputList[0])
        height_input.set_text(textInputList[1])
        goal_input.set_text(textInputList[2])

        pygame.display.flip()
        clock.tick(FPS)

def transfer_to_playable_map(map, myFile):
    for line in map:
        for c in line:
            myFile.write(c)
        myFile.write("\n")

def generating_map_page(screen, width, height, goals):
    global LAYER
    global LEVEL_SELECTED

    info1 = Label("Generating...", (SCREEN_X // 5, SCREEN_Y // 7), screen)
    info2 = Label("Generated Map:", (SCREEN_X // 5 - 50, SCREEN_Y // 7), screen)
    back_btn = Button("Back", (4 * SCREEN_X // 5, SCREEN_Y // 10), screen)

    place_holder = Label(f"MAP HERE w:{width} h:{height} goalds:{goals}", (SCREEN_X // 2, SCREEN_Y // 2), screen)

    debug_button = Button("DEBUG BUTTON", (SCREEN_X // 2, SCREEN_Y // 2), screen)

    export_btn = Button("Export!", (4.5 * SCREEN_X // 5, SCREEN_Y // 10), screen)
    regen_btn = Button("Regenerate", (5.5 * SCREEN_X // 5, SCREEN_Y // 10), screen)

    generated = False

    map = None
    screen.fill(WHITE)
    info1.draw()

    pygame.display.flip()

    while LAYER == 8:
        screen.fill(WHITE)
        if generated:
            info2.draw()
            back_btn.draw()
            export_btn.draw()
            regen_btn.draw()

            the_board = bd.Board(map=map)
            map_x = the_board.getSize()[0]
            map_y = the_board.getSize()[1]

            interval_x = (SCREEN_X - 200) // map_x
            interval_y = (SCREEN_Y - 200) // map_y

            size = min((SCREEN_X - 200) // map_x, (SCREEN_Y - 200) // map_y)
            scale_image(size)

            centre_x = 500
            centre_y = 450

            x = centre_x - map_x // 2 * size
            y = centre_y - map_y // 2 * size

            the_map = the_board.getMap()

            for row in the_map:
                for char in row:
                    if char == ' ':
                        screen.blit(floor_image, (x, y))
                    elif char == '#':
                        screen.blit(wall_image, (x, y))
                    elif char == '&':
                        screen.blit(worker_image, (x, y))
                    elif char == 'B':
                        screen.blit(box_image, (x, y))
                    elif char == 'X':
                        screen.blit(box_docked_image, (x, y))
                    elif char == '.':
                        screen.blit(dock_image, (x, y))
                    # worker on the goal
                    elif char == '+':
                        screen.blit(worker_docker_image, (x, y))

                    x = x + size
                x = centre_x - map_x // 2 * size
                y = y + size

        else:
            info1.draw()
            back_btn.draw()
            debug_button.draw()
            map = level_generation.generating_map(int(width),int(height),int(goals))
            generated = True

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == MOUSEBUTTONDOWN:
                if debug_button.button_rect.collidepoint(pygame.mouse.get_pos()):
                    generated = True

                if export_btn.button_rect.collidepoint(pygame.mouse.get_pos()):

                    string = 'levels/Generated/*[a-z]*[0-9].*'
                    level_count = len([name for name in glob.glob('levels/Generated/*[a-z]*[0-9].*')])
                    myFile = open("levels/Generated/generated" + str(level_count + 1) + ".txt", 'w')
                    transfer_to_playable_map(map, myFile)
                    myFile.close()
                    LAYER = 0
                elif regen_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    print("Regen Map")
                    generated = False
                if back_btn.button_rect.collidepoint(pygame.mouse.get_pos()):
                    LAYER = 0

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y))
    clock = pygame.time.Clock()
    pygame.display.set_caption("Sokoban")

    load_images()

    map_width = 0
    map_height = 0
    map_goal = 0

    while True:
        # Menu Page
        if LAYER == 0:
            menu_page(screen)
        # Player Choose level
        elif LAYER == 1:
            player_choose_level(screen)
        # Gaming
        elif LAYER == 2:
            gaming(screen, LEVEL_SELECTED)
        # Wining page
        elif LAYER == 3:
            winning_page(screen)
        # AI choose level
        elif LAYER == 4:
            ai_solver_choose_level(screen)
        # AI solving page
        elif LAYER == 5:
            solving_page(screen)
        # Map Generator
        elif LAYER == 7:
            map_width, map_height, map_goal = gen_map_page(screen)
        elif LAYER == 8:
            generating_map_page(screen, map_width, map_height, map_goal)
        elif LAYER == 9:
            player_choose_folder(screen)
        elif LAYER == 10:
            ai_choose_folder(screen)
        elif LAYER == 11:
            transition_page_algorithm_selection(screen)


# To add:
# if the inital is like
#########
##    & #
####X####
# Then there is no solution!!!
# Needs some early check to get rid og this