# gym-catch
A simple 1-D cursor-target catch game (with atari rendering) in the gym OpenAI environment

<p align="center">
  <img src="extras/wobble_screenshot" width="600" title="Screenshot of Wobble Game">
</p>

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

## Environment Details
----------------------

* **wobble-v0 :** Default settings (`max_timesteps=100`, `max_dist=3`, `tcp_tagging=False`, `tcp_port=15361`)
* **WobbleNoFrameskip-v4 :** Default settings and `max_timesteps=100`
* **WobbleNoFrameskip-v3 :** Default settings and `max_timesteps=100` and `tcp_tagging=True`

## Agent Details
----------------

* `agents/random_agent.py` random agent plays game with given error probability to take actions (Perr)

## References
-------------
