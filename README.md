# gym-catch
A simple 1-D cursor-target catch game (with atari rendering) in the gym OpenAI environment

## Installation instructions
----------------------------

Requirements: gym with atari dependency

```shell
git clone https://github.com/meagmohit/gym-wobble
cd gym-wobble
python setup.py install
```

```python
import gym
import gym_wobble
env = gym.make('wobble-v0') # The other option is 'WobbleNoFrameskip-v4'
env.render()
```

## References
-------------
