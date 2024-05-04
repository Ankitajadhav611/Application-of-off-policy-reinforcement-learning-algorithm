import matplotlib
import matplotlib.pyplot as plt

# Import custom environment class
from cus_env import CustomEnv

# Open figure in new window (not necessary)
#matplotlib.use('QtAgg')


# generate class instance
env = CustomEnv()

# reset environment
state = env.reset()
done = False

# Simulation loop
while not done:
    # generate random action sample (normally this is done by the agent)
    action = env.action_space.sample()
    # do one simulation step
    observation, reward, done, info = env.step(action)
    # render new progress
    env.render()
    if done:
        # necessary to left finished plot open
        plt.show(block=True)
