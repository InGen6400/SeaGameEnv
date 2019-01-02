import sys, pygame
from asyncio import sleep
from numpy.core.multiarray import ndarray
from typing import List

from pygame.locals import *

from sea_game import Ship, Tank


class Render(object):
    def __init__(self, width, height, scale, title='', ship_size=14, tank_size=10):
        pygame.init()
        self.size = (width*scale, height*scale)
        self.window = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title)
        self.game_surface = pygame.Surface((256*scale, 256*scale))
        self.ship_font = pygame.font.Font(None, 16*scale)
        self.tank_font = pygame.font.Font(None, 12*scale)
        self.scale = scale
        self.ship_size = ship_size*scale
        self.tank_size = tank_size*scale

    def update(self):
        # 上下反転
        self.window.blit(pygame.transform.flip(self.game_surface, False, True), (0, 0))
        pygame.display.update()
        self.game_surface.fill((0, 0, 255))
        for event in pygame.event.get():
            if event.type == QUIT:          # 閉じるボタンが押されたとき
                pygame.quit()
                sys.exit()
            if event.type == KEYDOWN:       # キーを押したとき
                if event.key == K_ESCAPE:   # Escキーが押されたとき
                    pygame.quit()
                    sys.exit()

    def draw_ship(self, ship_list: List[Ship]):
        for ship in ship_list:
            pygame.draw.ellipse(self.game_surface, (255, 0, 0), ((ship.pos[1]*self.scale-self.ship_size/2),
                                                                 (ship.pos[0]*self.scale-self.ship_size/2),
                                                                 self.ship_size, self.ship_size))
            text_view = self.ship_font.render(ship.name, True, (255, 255, 255))
            self.game_surface.blit(pygame.transform.flip(text_view, False, True),
                                   (ship.pos[1]*self.scale+self.ship_size/2, ship.pos[0]*self.scale))
            text_view = self.ship_font.render(str(ship.point), True, (0, 255, 0))
            self.game_surface.blit(pygame.transform.flip(text_view, False, True),
                                   (ship.pos[1]*self.scale+self.ship_size/2, ship.pos[0]*self.scale-self.ship_size))

    def draw_tank(self, tank_list: List[Tank]):
        for tank in tank_list:
            pygame.draw.ellipse(self.game_surface, (255, 255, 0), ((tank.pos[1]*self.scale-self.tank_size/2),
                                                                   (tank.pos[0]*self.scale-self.tank_size/2),
                                                                   self.tank_size, self.tank_size))
            text_view = self.tank_font.render(str(tank.point), True, (0, 0, 0))
            self.game_surface.blit(pygame.transform.flip(text_view, False, True),
                                   (tank.pos[1]*self.scale - text_view.get_rect().width/2,
                                    tank.pos[0]*self.scale - text_view.get_rect().height/2))
