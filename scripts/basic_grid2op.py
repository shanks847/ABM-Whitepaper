import grid2op
# create an environment
env_name = "rte_case14_realistic"  # for example, other environments might be usable
env = grid2op.make(env_name)

# create an agent
from grid2op.Agent import RandomAgent
my_agent = RandomAgent(env.action_space)

# proceed as you would any open ai gym loop
nb_episode = 10
for _ in range(nb_episode):
    # you perform in this case 10 different episodes
    obs = env.reset()
    reward = env.reward_range[0]
    done = False
    while not done:
        # here you loop on the time steps: at each step your agent receive an observation
        # takes an action
        # and the environment computes the next observation that will be used at the next step.
        act = my_agent.act(obs, reward, done)
        obs, reward, done, info = env.step(act)