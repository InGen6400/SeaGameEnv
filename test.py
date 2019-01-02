from time import sleep

from sea_game import SeaGameEnv
import numpy as np

env = SeaGameEnv()
env.reset()
env.render()
while True:
    obs, reward, done, _ = env.step(1)
    env.render()
    print(str(reward) + ': ' + str(env.ship_list[0].point) + '/' + str(env.tank_all))
    sleep(0.1)
    if done:
        env.reset()
