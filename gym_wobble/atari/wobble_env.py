import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import matplotlib.pyplot as plt

# Libraires for sending external stimulations over TCP port
import sys
import socket
from time import time, sleep


class ALEInterface(object):
    def __init__(self):
      self.lives_left = 0

    def lives(self):
      return 0 #self.lives_left

class WobbleEnv(gym.Env):

    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second' : 50}

    def __init__(self, max_timesteps=100, max_dist=3, tcp_tagging=False, tcp_port=15361):

        # Atari-platform related parameters
        self.atari_dims = (210,160,3)		# Specifies standard atari resolution
        (self.atari_height, self.atari_width, self.atari_channels) = self.atari_dims

        #  Game-related paramteres
        self.grid_size = 20     # all codes are to force this
        self.block_size = self.atari_width/self.grid_size
        self.actions = [1, -1]
        self.score = 0.0
        self.time = 0
        self.max_dist = max_dist
        self.max_timesteps = max_timesteps

        # Gym-related variables [must be defined]
        self._action_set = np.array([3,4],dtype=np.int32)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.atari_height, self.atari_width, 3), dtype=np.uint8)
        self.viewer = None

        # Code for TCP Tagging
        self.tcp_tagging = tcp_tagging
        if (self.tcp_tagging):
            self.host = '127.0.0.1'
            self.port = tcp_port
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self.host, self.port))

        # Methods
        self.ale = ALEInterface()
        self.seed()
        self.reset()

    # Act by taking an action # return observation (object), reward (float), done (boolean) and info (dict)
    def step(self, action):
        if isinstance(action, np.ndarray):
          action = action[0]
        assert self.action_space.contains(action)   # makes sure the action is valid

        self.time = self.time + 1
        # Updating the state, state is hidden from observation
        [x_cursor, x_target] = self.state
        current_action = self.actions[action]
        x_cursor = min(max(x_cursor + current_action,0),19)
        # print current_action, x_cursor
        self.state = [x_cursor, x_target]

        # Generating the rewards
        reward = 0.0
        if x_cursor==x_target:
            reward = 1.0
            self.score = self.score + reward

            x_target = x_cursor + (1 + np.random.randint(self.max_dist))*np.random.choice([1,-1])
            if (x_target<0 or x_target>=20):
                x_cursor = 10
                x_target = x_cursor + (1 + np.random.randint(self.max_dist))*np.random.choice([1,-1])
            self.state = [x_cursor, x_target]

        # Generate the done (boolean)
        done = False
        if (self.time >= self.max_timesteps):
            done = True

        # Sending the external stimulation over TCP port
        if self.tcp_tagging:
            padding=[0]*8
            event_id = [0, 0, 0, 0, 0, x_target, x_cursor, action]
            timestamp=list(self.to_byte(int(time()*1000), 8))
            self.s.sendall(bytearray(padding+event_id+timestamp))

        return self._get_observation(), reward, done, {"ale.lives": self.ale.lives(), "internal_state": self.state}

    def reset(self):
        self.score = 0.0
        x_cursor = 10
        self.time = 0
        x_target = x_cursor + (1 + np.random.randint(self.max_dist))*np.random.choice([1,-1])    # picks b/w [-size to +size, except 0]
        self.state = [x_cursor, x_target]
        return self._get_observation()

    def _get_observation(self):
        img = np.zeros(self.atari_dims, dtype=np.uint8) # Black screen
        [x_cursor, x_target] = self.state
        row_mid = self.atari_height/2
        row_up = row_mid - self.block_size/2
        row_dn = row_mid + self.block_size/2
        img[row_up,:,:] = 255   # Upper white horizontal strip
        img[row_dn,:,:] = 255   # Bottom white horizontal strip
        # drawing vertical white strips
        for block_ind in xrange(self.grid_size):
            col_mid = block_ind*self.block_size + self.block_size/2
            col_left = col_mid - self.block_size/2
            img[row_up:row_dn,col_left,:] = 255
        img[row_up:row_dn,-1,:] = 255   # Rightmost white vertical strip

        # Drawing Target
        col_mid = x_target*self.block_size + self.block_size/2
        if x_target > x_cursor: # Red Square
            img[row_mid-1:row_mid+2,col_mid-1:col_mid+2,0]=255
        elif x_target < x_cursor: # Blue Square
            img[row_mid-1:row_mid+2,col_mid-1:col_mid+2,2]=255

        # Drawing Cursor
        # prev_val = img[row_mid-1:row_mid+2,col_mid-1:col_mid+2,:]
        col_mid = x_cursor*self.block_size + self.block_size/2
        img[row_mid-3:row_mid+4,col_mid-3:col_mid+4,1] = 255
        # img[row_mid-1:row_mid+2,col_mid-1:col_mid+2,:] = prev_val

        return img

    def render(self, mode='human', close=False):
        img = self._get_observation()
        if mode == 'rgb_array':
            return img
        #return np.array(...) # return RGB frame suitable for video
        elif mode is 'human':
            #... # pop up a window and render
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer(maxwidth=1920)

            self.viewer.imshow(np.repeat(np.repeat(img[80:131:,:,], 10, axis=0), 10, axis=1))
            return self.viewer.isopen
            #plt.imshow(img)
            #plt.show()
        else:
            super(WobbleEnv, self).render(mode=mode) # just raise an exception

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.tcp_tagging:
            self.s.close()

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]

    @property
    def _n_actions(self):
        return len(self._action_set)

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2**31
        return [seed1, seed2]

    # A function for TCP_tagging in openvibe
    # transform a value into an array of byte values in little-endian order.
    def to_byte(self, value, length):
        for x in range(length):
            yield value%256
            value//=256


    def get_keys_to_action(self):
        KEYWORD_TO_KEY = {
            'UP':      ord('w'),
            'DOWN':    ord('s'),
            'LEFT':    ord('a'),
            'RIGHT':   ord('d'),
            'FIRE':    ord(' '),
        }

        keys_to_action = {}

        for action_id, action_meaning in enumerate(self.get_action_meanings()):
            keys = []
            for keyword, key in KEYWORD_TO_KEY.items():
                if keyword in action_meaning:
                    keys.append(key)
            keys = tuple(sorted(keys))

            assert keys not in keys_to_action
            keys_to_action[keys] = action_id

        return keys_to_action


ACTION_MEANING = {
    0 : "NOOP",
    1 : "FIRE",
    2 : "UP",
    3 : "RIGHT",
    4 : "LEFT",
    5 : "DOWN",
    6 : "UPRIGHT",
    7 : "UPLEFT",
    8 : "DOWNRIGHT",
    9 : "DOWNLEFT",
    10 : "UPFIRE",
    11 : "RIGHTFIRE",
    12 : "LEFTFIRE",
    13 : "DOWNFIRE",
    14 : "UPRIGHTFIRE",
    15 : "UPLEFTFIRE",
    16 : "DOWNRIGHTFIRE",
    17 : "DOWNLEFTFIRE",
}
