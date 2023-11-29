import numpy as np  
import gym

from ding.envs import BaseEnv, BaseEnvTimestep
from ding.utils import ENV_REGISTRY

@ENV_REGISTRY.register('guess_the_number')
class GuessTheNumberEnv(BaseEnv):
    
    def __init__(self, cfg: dict = {}) -> None:

        self._cfg = cfg

        self.max_guesses = cfg.max_guesses
        self.max_number = cfg.action_space_size
        self.fixed_secret = cfg.fixed_secret
        self.continuous_rewards = cfg.continuous_rewards

        self._init_flag = False
        self._replay_path = None
#        self._observation_space = gym.spaces.Discrete(3, start=-1)
        obs_base = np.array([self.max_number, 3], dtype='int8')
        obs_shape = np.concatenate([obs_base for _ in range(self.max_guesses)])
        self._observation_space = gym.spaces.MultiDiscrete(obs_shape) 
#        print(self._observation_space.shape)
        self._action_space = gym.spaces.Discrete(self.max_number, start=1)
        self._action_space.seed(0)  # default seed
        self._reward_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1, ), dtype=np.float32)
        self._continuous = False
        

    def reset(self):
        self.episode_length = 0
        self.episode_return = 0

        #np.random.seed(0)

#        self.secret_number = self._action_space.sample()
        self.secret_number = np.random.randint(self.max_number)
        if self.fixed_secret:
            self.secret_number = 2 # testing

        self.history = np.full_like(self._observation_space, -1, 'int8')

        observation = self.history
        action_mask = np.ones(self.max_number, 'int8')
        to_play = -1

        obs_dict = {
            'observation': observation,
            'action_mask': action_mask,
            'to_play': -1,
            }

        return obs_dict

    def observe(self, action):
        if action < self.secret_number:
            observation = 0
        elif action > self.secret_number:
            observation = 1
        elif action == self.secret_number:
            observation = 2

        self.history[self.episode_length * 2 - 1] = observation

        observation = self.history
        action_mask = np.ones(self.max_number, 'int8')
        to_play = -1

        obs_dict = {
            'observation': observation,
            'action_mask': action_mask,
            'to_play': -1,
            }

        return obs_dict

    def step(self, action):
        self.episode_length += 1

        self.history[self.episode_length * 2 - 2] = action

        reward = 0
        done = False

        if action == self.secret_number:
            reward = 1.0
            done = True
        elif self.continuous_rewards:
            reward = ((abs(action - self.secret_number) + 1.)**-1 - 1)/self.max_guesses

        if self.episode_length >= self.max_guesses:
            done = True

        observation = self.observe(action)

        self.episode_return += reward

        info = {}

        if done:
            info['eval_episode_return'] = self.episode_return
            print((self.secret_number, observation))    
        
        return BaseEnvTimestep(observation, reward, done, info)

    def seed(self, seed: int, dynamic_seed: bool = True) -> None:
        self._seed = seed
        self._dynamic_seed = dynamic_seed
        np.random.seed(self._seed)

    def __repr__(self) -> str:
        return 'LightZero Guess the Number Env'

    def close(self) -> None:
        pass

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space
