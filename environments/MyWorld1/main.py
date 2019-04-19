from environments.MyWorld1 import agent
from environments.MyWorld1 import world
from environments.MyWorld1.behaviours import random_behaviours

width, height = 400, 400

behaviour1 = random_behaviours.RandomBehaviour1(32, 6)
behaviour2 = random_behaviours.RandomBehaviour2(32, 6, width, height)
agent1 = agent.Agent(velocity=(1, 1), position=(width // 2, height // 2), r=30, behaviour=behaviour2)
world = world.World(width, height, [agent1])
world.run()
