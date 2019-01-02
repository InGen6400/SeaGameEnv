import random
from typing import List

import numpy as np
import gym
from gym import spaces, logger
from numpy.core.multiarray import ndarray

DIR_2_VECTOR: ndarray = np.array([
    [0, 10],
    [-10, 0],
    [0, -10],
    [10, 0],
    [0, 0]
])

RIGHT = 0
DOWN = 1
LEFT = 2
UP = 3
NOMOVE = 4


class Tank(object):
    def __init__(self, point, y, x):
        self.point = point
        self.pos = (y, x)


class Ship(object):
    pos: ndarray

    def __init__(self, name):
        self.name = name
        self.point = 0
        self.capture = 0
        self.pos = np.array([random.randrange(0, 256), random.randrange(0, 256)])

    def reset(self):
        self.point = 0
        self.capture = 0
        self.pos = np.array([random.randrange(0, 256), random.randrange(0, 256)])

    def move(self, moves: List[int]):
        self.pos = self.pos + DIR_2_VECTOR[moves[0]]
        self.pos = self.pos + DIR_2_VECTOR[moves[1]]
        # 0~256に収める
        if self.pos[0] < 0:
            self.pos[0] = self.pos[0]+256
        if self.pos[1] < 0:
            self.pos[1] = self.pos[1]+256
        self.pos[0] = self.pos[0] % 256
        self.pos[1] = self.pos[1] % 256


from ship_agent import ShipAgent
from rendering.rendering import Render


class SeaGameEnv(gym.core.Env):
    tank_map: ndarray
    ship_map: ndarray
    npc_list: List[ShipAgent]
    ship_list: List[Ship]
    tank_list: List[Tank]

    def __init__(self, nb_npc=5, max_step=None, npc_name='*', player_name='Admiral'):
        self.map = np.zeros((256, 512))
        self.ship_map = self.map[:, :256]
        self.tank_map = self.map[:, 256:]
        self.tank_list = []
        self.tank_plan = []
        self.tank_all = 0
        self.nb_step = 0
        if max_step:
            self.max_step = max_step
        else:
            self.max_step = random.randrange(1, 5)*60*2
        self.nb_npc = nb_npc
        self.screen = Render(256, 256, 2)
        self.ship_list = [Ship(player_name)] + [ShipAgent(npc_name + str(i + 1)) for i in range(nb_npc)]

        self.action_space = gym.spaces.Discrete(len(ACTION_MEANS))
        self.observation_space = gym.spaces.Box(low=0., high=1., shape=(512, 256), dtype=np.float64)
        self.obs = self.observe()

    def step(self, action: int):
        # AI移動
        moves = ACTION_MEANS[action]
        self.ship_list[0].move(moves)

        # NPC移動
        ship: ShipAgent
        for ship in self.ship_list[1:]:
            ship.decide_move(self.ship_map, self.tank_map)
            ship.move(ship.next_move)
            self.ship_map[ship.pos] = ship.point

        self.collide()

        done = False
        self.nb_step = self.nb_step + 1
        if self.nb_step >= self.max_step:
            done = True
        else:
            # 2ステップに一回タンクを生成
            if self.nb_step % 2 == 0:
                self.tank_list.append(Tank(self.tank_plan.pop(0), random.randrange(0, 256), random.randrange(0, 256)))

        self.mapping()
        reward = self.ship_list[0].capture/self.tank_all * 100
        self.ship_list[0].capture = 0
        self.obs = self.observe()
        return self.obs, reward, done, {}

    def collide(self):
        for ship in self.ship_list:
            for tank in self.tank_list:
                dx = tank.pos[1] - ship.pos[1]
                dy = tank.pos[0] - ship.pos[0]
                if dx > 128:
                    dx = dx - 256
                elif dx < -128:
                    dx = dx + 256
                if dy > 128:
                    dy = dy - 256
                elif dy < -128:
                    dy = dy + 256
                if dx*dx+dy*dy < 100:
                    ship.point = ship.point + tank.point
                    ship.capture = tank.point
                    self.tank_list.remove(tank)

    def mapping(self):
        self.ship_map.fill(0)
        for ship in self.ship_list:
            self.ship_map[ship.pos] = self.ship_map[ship.pos] + ship.point

        self.tank_map.fill(0)
        for tank in self.tank_list:
            self.tank_map[tank.pos] = self.tank_map[tank.pos] + tank.point

    def observe(self):
        x = self.ship_list[0].pos[1]
        y = self.ship_list[0].pos[0]
        ship_map = np.roll(self.ship_map, 128-x, axis=1)
        ship_map = np.roll(ship_map, 128-y, axis=0)
        tank_map = np.roll(self.tank_map, 128-x, axis=1)
        tank_map = np.roll(tank_map, 128-y, axis=0)
        return np.hstack((ship_map, tank_map))

    def reset(self):
        self.nb_step = 0
        self.map.fill(0)
        self.tank_list.clear()
        self.tank_plan.clear()
        for _ in range(self.max_step//2):
            self.tank_plan.append(random.randint(1, 4))
        self.tank_all = sum(self.tank_plan)
        for ship in self.ship_list:
            ship.reset()
        self.mapping()
        return self.observe()

    def render(self, mode='human'):
        self.screen.draw_ship(self.ship_list)
        self.screen.draw_tank(self.tank_list)
        self.screen.update()


ACTION_MEANS = [
    [RIGHT, RIGHT],
    [RIGHT, DOWN],
    [RIGHT, LEFT],
    [RIGHT, UP],
    [DOWN, RIGHT],
    [DOWN, DOWN],
    [DOWN, LEFT],
    [DOWN, UP],
    [LEFT, RIGHT],
    [LEFT, DOWN],
    [LEFT, LEFT],
    [LEFT, UP],
    [UP, RIGHT],
    [UP, DOWN],
    [UP, LEFT],
    [UP, UP],
]
