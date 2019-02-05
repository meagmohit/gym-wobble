import gym
import gym_wobble
import time
import numpy as np
env = gym.make('WobbleNoFrameskip-v3')
env.reset()
action_idx = [0, 1]
actions = [1, -1]	# 0 means right i.e. 1, 1 means -1 i.e. left
p_err = 0.2
for _ in range(1):
	done = False
	env.reset()
	while not done:
		state = env.unwrapped.state
		[x_cursor, x_target] = state
		if x_cursor < x_target:	# cursor should move right with probability of 1-p_err
			action = np.random.choice(action_idx, p=[1-p_err, p_err])
		else:
			action = np.random.choice(action_idx, p=[p_err, 1-p_err])
		(obs, reward, done, info) =  env.step(action) # take a random action
		print actions[action], info['internal_state'], reward
		env.render()
		time.sleep(2.0)

print env.unwrapped.score
env.close()
