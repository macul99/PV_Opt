#import gym
#from gym import spaces
import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np
from collections import OrderedDict

from dataclasses import dataclass
# as an example
@dataclass
class EnvConfig:
    render_mode = None
    seed = 42
    land_length = 36
    land_width = 36
    max_num_obstacle = 5
    max_obstacle_size = 3
    kernal_l = 3
    kernal_w = 2
    max_reward = 10 # terminate if reward > max_reward
    max_steps = 100


def topk_by_partition(input, k, axis=None, ascending=True):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis) # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis) # k non-sorted values

    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    if not ascending:
        input *= -1
    val = np.take_along_axis(input, ind_part, axis=axis) 
    return ind, val

def define_playground(length, width, num_obstacle, max_obstacle_size):
    # define playground with random obstable
    # width in axis 0, length in axis 1
    idx, _ = topk_by_partition(np.random.randn(width, length), num_obstacle)
    playground = np.zeros((width, length))
    
    for i in idx:
        ix, iy = i//length, i%length
        s = np.random.randint(max_obstacle_size)+1
        playground[ix:ix+s,iy:iy+s]=1
    #print(idx)
    #print(playground)
    return playground

def get_kernal_overlap_obstacle_area(kernal_l, kernal_w, x):
    # get obstacle area overlapped with kernal
    w, l = x.shape
    kernal_area = kernal_l * kernal_w
    #assert (l>=kernal_l) and (w>=kernal_w), 'kernal size too big'
    
    rst = np.zeros((w, l))
    for i in range(w):
        for j in range(l):
            rst[i,j]=np.sum(x[i:i+kernal_w,j:j+kernal_l])/kernal_area
    #print(rst)
    return rst

def gen_random_land(land_l, land_w, kernal_l, kernal_w):
    # generate a land with various irradiation
    #return torch.zeros((land_w, land_l)), torch.zeros((land_w, land_l))
    kernal_area = kernal_l * kernal_w
    land = np.random.random(land_l*land_w).reshape(land_w, land_l)
    land = np.where(land>0.8, land*2, land)/2
    land_panel_conv = np.zeros((land_w, land_l))
    for i in range(land_w):
        for j in range(land_l):
            land_panel_conv[i,j]=np.sum(land[i:i+kernal_w,j:j+kernal_l])/kernal_area
            
    return land, land_panel_conv
    
def prepare_input_data(cfg):
    land, land_panel_conv = gen_random_land(cfg.land_length, cfg.land_width, cfg.kernal_l, cfg.kernal_w)
    num_obstacle = np.random.randint(cfg.max_num_obstacle)+1
    land_obstacle = define_playground(cfg.land_length, cfg.land_width, num_obstacle, cfg.max_obstacle_size)
    land_obstacle_conv = get_kernal_overlap_obstacle_area(cfg.kernal_l, cfg.kernal_w, land_obstacle)
    d_in = np.stack([land_obstacle,land])
    d_out = np.stack([land_obstacle_conv,land_panel_conv])
    
    c=np.where(d_out[0:1,:,:]>0, 0, d_out[1:,:,:])
    d_max=np.zeros_like(c).reshape([c.shape[0],-1])
    d_max[np.arange(0,c.shape[0]),np.argmax(c.reshape([c.shape[0],-1]), axis=1)]=1
    d_max=d_max.reshape(c.shape)
    
    return d_in, d_out, d_max

class PanelGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        try:
            import pygame
        except:
            pygame = None
            render_mode = None
            
        self.cfg = EnvConfig()
        self.land_length = int(self.cfg.land_length)  # The size of the square grid
        self.land_width = int(self.cfg.land_width)  # The size of the square grid
        self.max_steps = self.cfg.max_steps  # The size of the PyGame window
        self.current_step = 0

        # Observations are dictionaries with the agent's and the target's location.
        # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
        self.observation_space = spaces.Dict(
            {
                'panel': spaces.Box(0, 255, shape=(1, self.land_width,self.land_length), dtype=np.uint8),
                'obstruction': spaces.Box(0, 255, shape=(1, self.land_width,self.land_length), dtype=np.uint8),
                'land_irr': spaces.Box(0, 255, shape=(1, self.land_width,self.land_length), dtype=np.uint8),
            }
        ) # panel installation status, obstraction, land_irr
        self._obs_space = spaces.Box(0, 1, shape=(self.land_width,self.land_length), dtype=float)

        # Actions: move to a location, and install/uninstall a panel
        self.action_space = spaces.Discrete(self.land_width*self.land_length) #spaces.Discrete(2*self.land_width*self.land_length)

        assert self.cfg.render_mode is None or self.cfg.render_mode in self.metadata["render_modes"]
        self.render_mode = self.cfg.render_mode

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.window = None
        self.clock = None
        
        self.data_raw, self.data_conv, self.data_max_irr = prepare_input_data(self.cfg)

    def _get_obs(self):
        return OrderedDict(
            [
                ('panel', (self._panel_installation_area[np.newaxis,:].copy()*255).astype(np.uint8)),
                ('obstruction', (self.data_conv[0:1].copy()*255).astype(np.uint8)),
                ('land_irr', (self.data_conv[1:].copy()*255).astype(np.uint8)),
            ]
        )

    def _get_info(self):
        return {
            "panel_installation_reward": self._reward_panel_installation,
            "obstruction_penalty": self._penalty_obstruction, # panel installed at obstruction area
            "data_raw": self.data_raw,
            "data_conv": self.data_conv,
            "data_max_irr": self.data_max_irr,
        }

    def _panel_installation_reward(self):
        # no reward for overlapping panels
        
        # define panel coverage based on kernal_l, kernal_w, ref point is left-top conner
        panel_coverage= []
        for j in range(self.cfg.kernal_w):
            panel_coverage = panel_coverage + [i+self.land_length*j for i in range(self.cfg.kernal_l)]
        panel_coverage = np.array(panel_coverage)

        # get panel placement summary, panel_placement has shape (land_width*land_length, land_width*land_length)
        # for grid with panel, value set to 1, otherwise set to 0
        # for example, land_length=4,land_width=4,kernal_l=3,kernal_w=2 and r[0]==1, then
        # panel_placement[0] = [1,1,1,0, 1,1,1,0, 0,0,0,0, 0,0,0,0]
        # for example, land_length=4,land_width=4,kernal_l=3,kernal_w=2 and r[1]==1, then
        # panel_placement[1] = [0,1,1,1, 0,1,1,1, 0,0,0,0, 0,0,0,0]
        a = []
        r = self._panel_installation.reshape(-1)
        for i in range(self.land_length*self.land_width):
            x = np.zeros(self.land_length*self.land_width)
            if r[i] and i%self.land_length<=(self.land_length-self.cfg.kernal_l) and i//self.land_length<=(self.land_width-self.cfg.kernal_w):
                x[panel_coverage+i] = 1
            a.append(x)
        panel_placement = np.stack(a)

        # find non-overlap grid, shape (land_width*land_length, land_width*land_length), 1 for non-overlap grid, 0 for overlap
        non_overlap_grid=(panel_placement.sum(axis=0)==1).astype(int)
        non_overlap_grid=np.tile(non_overlap_grid.reshape([-1,non_overlap_grid.shape[0]]),(panel_placement.shape[0],1))
        # find non_overlap panel, non-overlap panel should have non_overlap_cnt == kernal_l*kernal_w
        non_overlap_cnt=(panel_placement*non_overlap_grid)
        non_overlap_cnt=non_overlap_cnt.sum(axis=1)

        return self.data_conv[1].reshape(-1)[np.where(non_overlap_cnt==self.cfg.kernal_l*self.cfg.kernal_w)].sum()
    
    def _obstacle_penalty(self, scale=1.0):
        # logits and obstacle have shape (Batch, w, l), have values in range (0, 1)
        r = self._panel_installation.astype(float)
        loss = np.where(self.data_conv[0]>0,1,0) * r
        #print(r, logits, obstacle, loss)
        return -scale*loss.sum()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.current_step = 0
        
        # default panel installation are all 0
        self._panel_installation = np.zeros([self.land_width, self.land_length]) #self._obs_space.sample()>0.9
        self._panel_installation_area = np.zeros([self.land_width, self.land_length])

        observation = self._get_obs()
        
        self._reward_panel_installation = self._panel_installation_reward() # positive
        self._penalty_obstruction = self._obstacle_penalty() # negative
        
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action
        #self._panel_installation[action.astype(bool)] = 1 - self._panel_installation[action.astype(bool)]
        rm = action%(self.land_width*self.land_length)
        self._panel_installation[(rm//self.land_length, rm%self.land_length)] = 1 - self._panel_installation[(rm//self.land_length, rm%self.land_length)]
        if self._panel_installation[(rm//self.land_length, rm%self.land_length)]:
            self._panel_installation_area[rm//self.land_length:rm//self.land_length+self.cfg.kernal_w,
                                          rm%self.land_length:rm%self.land_length+self.cfg.kernal_l] += 1/(self.cfg.kernal_l*self.cfg.kernal_w)
        else:
            self._panel_installation_area[rm//self.land_length:rm//self.land_length+self.cfg.kernal_w,
                                          rm%self.land_length:rm%self.land_length+self.cfg.kernal_l] -= 1/(self.cfg.kernal_l*self.cfg.kernal_w)
        observation = self._get_obs()
        
        reward_panel_installation = self._panel_installation_reward() # positive
        penalty_obstruction = self._obstacle_penalty() # negative
        
        reward = reward_panel_installation + penalty_obstruction - (self._reward_panel_installation + self._penalty_obstruction)
        
        reward += -0.001 
        
        self._reward_panel_installation = reward_panel_installation
        self._penalty_obstruction = penalty_obstruction
        
        # An episode is done iff the agent has reached the target
        terminated = (self._reward_panel_installation + self._penalty_obstruction) > self.cfg.max_reward
        info = self._get_info()

        self.current_step += 1
        truncated = self.current_step >= self.max_steps
        
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
