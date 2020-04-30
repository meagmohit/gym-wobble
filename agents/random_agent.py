import gym
import gym_wobble
import time
import numpy as np
env = gym.make('WobbleNoFrameskip-v4')
env.reset()
action_idx = [0, 1]
actions = [1, -1]	# 0 means right i.e. 1, 1 means -1 i.e. left
p_err = 0.2
speed = 1.0 # in seconds
total_play = 1
for _ in range(total_play):
	done = False
	env.reset()
	env.render()
	while not done:
		state = env.unwrapped._state
		[x_cursor, x_target] = state
		if x_cursor < x_target:	# cursor should move right with probability of 1-p_err
			action = np.random.choice(action_idx, p=[1-p_err, p_err])
		else:
			action = np.random.choice(action_idx, p=[p_err, 1-p_err])
		(obs, reward, done, info) =  env.step(action) # take a random action
		print actions[action], info['internal_state'], reward
		env.render()
		time.sleep(speed)

print env.unwrapped.score
env.close()
