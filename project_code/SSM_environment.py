import os
from scipy import signal
import numpy as np
import pathlib
import math
import gym
from gym import spaces
import matplotlib.pyplot as plt
import control
import control.matlab as cnt


class SSMBasic(gym.Env):
    """
    Custom Environment that follows gym interface
    Description:
        A simple Heli model

    Source:
        Simulink model from IAT-Institute

    State:

    Info:

    Actions:


    States:

    Reward:

    Starting State:

    Episode Termination:


    """
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    def __init__(self,
                 env_dict: dict = None
                 ):
        """
        :param env_dict: All env kwargs
        """
        super(SSMBasic, self).__init__()

        self.env_dict = env_dict

        # set default params
        self.default_env_dict = {
            'input_bounds': np.array([[-150, 150],
                                      [-150, 150]]),
            'output_bounds': np.array([[-10, 10],
                                       [-10, 10]]),
            'penalty_output_bounds': np.array([[-10, 10],
                                               [-10, 10]]),
            'initial_output_bounds': np.array([[-10, 10],
                                               [-10, 10]]),
            'reference_output_bounds': np.array([[-10, 10],
                                                 [-10, 10]]),
            'active_output': np.array([1, 1]),
            'controlled_output': np.array([1, 1]),
            'int_e': False,
            'IT_e': False,
            'T_episode': 10,
            'Ts': 0.1,
            'substeps': 10,
            'render': True,
            'render_step': None,
            'save_fig': False,
            'full_screen': False,
            'plot/column': 3,
            'y_0': 'random',
            'debug': False
        }

        # write default params
        for key in self.default_env_dict:
            self.env_dict.setdefault(key, self.default_env_dict[key])

        
        self.env_name = 'SSM'
        self.n_input_model = 2
        self.n_output_model = 2
        self.n_state_model = 2
        self.input_name = ['u_1', 'u_2']
        self.output_name = ['y_1', 'y_2']

        self.index = np.where(self.env_dict['active_output'] == 0)
        #print(self.index)
        # Define action and observation spaces:
        # action dimensions
        self.n_actions = self.n_input_model
        # state dimensions
        self.n_observations = np.sum(self.env_dict['active_output']) \
                              + (1+self.env_dict['int_e']+self.env_dict['IT_e']) \
                              * np.sum(self.env_dict['controlled_output'])
        # distance between boundaries
        self.input_d = self.env_dict['input_bounds'][:, 1] - self.env_dict['input_bounds'][:, 0]
        self.output_d = self.env_dict['output_bounds'][:, 1] - self.env_dict['output_bounds'][:, 0]

        # define the observation space (space.Box for continuous spaces in gym environments)
        # outside of the claas the limits are [-1, 1]
        self.observation_space = spaces.Box(low=-self.bounds(self.n_observations),
                                            high=self.bounds(self.n_observations))
        # define the action space / limits are [-1, 1] because most agents can only output actions in this range
        self.action_space = spaces.Box(low=-self.bounds(self.n_actions),
                                       high=self.bounds(self.n_actions))

        # SSM
        # Time vector vor lsim
        self.T_v = np.linspace(0, self.env_dict['Ts'], self.env_dict['substeps'])
        #print(self.T_v)
        # SSM states
        self.state = np.zeros([self.n_state_model])
        #print(self.state)
        # SSM parameter
        self.a = np.array([[-6.0, 4.0], [8.0, -10.0]])
        self.b = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.c = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.d = np.array([[0.0, 0.0], [0.0, 0.0]])
        # SSM
        self.sys = signal.StateSpace(self.a, self.b, self.c, self.d)
        #print(self.sys1)
        #control system 
        #self.sys2 = cnt.ss(self.a, self.b, self.c, self.d)
        #print(self.sys2)
        # Inputvector vor SSM lsim
        self.u_v = np.zeros([self.env_dict['substeps'], self.n_input_model])
        #print(self.u_v)

        # IT params
        T = 2
        self.num = 1
        self.den = [T, 1, 0]
        self.state_IT = np.zeros([len(self.den)-1, sum(self.env_dict['controlled_output'])])
        #print(self.state_IT)
        self.IT = signal.TransferFunction(self.num, self.den)
        #print(self.IT)
        self.IT_e = np.zeros([sum(self.env_dict['controlled_output'])])
        #print(self.IT_e)
        self.e_v = np.zeros([self.env_dict['substeps'], sum(self.env_dict['controlled_output'])])
        #print(self.e_v)

        # Define counter and synthetic observations:
        # count the steps in one trajectory
        self.i_step = 0
        self.action = 0
        # save control error
        self.e = np.zeros([sum(self.env_dict['controlled_output'])])
        self.int_e = np.zeros([self.n_output_model])

        self.observation_name = list(self.input_name)
        self.action_name = list(self.output_name)
        
        # Initialize data saving vectors:
        # initialize simulation time vector
        self.n_max = int(self.env_dict['T_episode'] / self.env_dict['Ts'])+1
        #print(self.n_max)
        self.x = np.linspace(0, self.env_dict['T_episode'], self.n_max)
        #print(self.x)
        # initialize state trajectory vector
        self.y = np.zeros([self.n_output_model, self.n_max])
        self.y_ref = np.zeros([self.n_output_model, self.n_max])
        self.debug = np.zeros([self.n_max])
        # Fill the reference vector with the correct values. Needed for plotting
        self.generate_reference()
        # initialize action vector
        self.u = np.zeros([self.n_input_model, self.n_max])
        # initialize reward vector
        self.r = np.zeros(self.n_max)
        if self.env_dict['render_step'] is None:
            self.env_dict['render_step'] = self.n_max
        # set figure arguments
        self.fig = None
        self.output_ax = None
        self.output_line = None
        self.output_line_r = None
        self.input_ax = None
        self.input_line = None
        self.reward_ax = None
        self.reward_line = None
        self.debug_ax = None
        self.debug_line = None
        self.force_render = False
        # Initialize plots
        if self.env_dict['render']:
            self.init_render(self.env_dict['full_screen'])

    def init_render(self, full_screen: bool = False):
        self.env_dict['render'] = True
        plt.ion()
        self.fig = plt.figure(figsize=(12, 12))
        n_plots = self.n_input_model + self.n_output_model + 1 + self.env_dict['debug']
        n_columns = math.ceil(n_plots/self.env_dict['plot/column'])
        # state plots
        self.output_ax = [None] * self.n_output_model
        self.output_line = [None] * self.n_output_model
        self.output_line_r = [None] * self.n_output_model
        index_offset = 1
        for i in range(self.n_output_model):
            self.output_ax[i] = self.fig.add_subplot(self.env_dict['plot/column'], n_columns, index_offset + i)
            self.output_line[i] = self.output_ax[i].plot(self.x, self.y[i, :], 'r-')
            self.output_line_r[i] = self.output_ax[i].plot(self.x, self.y_ref[i, :], 'b-')
            self.output_ax[i].set_title('output of state space')
            self.output_line[i][0].set_label('$Y$')
            self.output_line_r[i][0].set_label('$y_{ref}$')
            self.output_ax[i].set_ylim(self.env_dict['output_bounds'][i, :])
            self.output_ax[i].legend(loc='upper right')
            self.output_ax[i].set_ylabel(self.output_name[i],labelpad=1)

        # action plots
        self.input_ax = [None] * self.n_input_model
        self.input_line = [None] * self.n_input_model
        index_offset += self.n_output_model
        for i in range(self.n_input_model):
            self.input_ax[i] = self.fig.add_subplot(self.env_dict['plot/column'], n_columns, index_offset + i)
            self.input_line[i] = self.input_ax[i].plot(self.x, self.u[i, :], 'r-')
            self.input_ax[i].set_title('input to state space(action)')
            self.input_line[i][0].set_label('$u$')
            self.input_ax[i].set_ylim(self.env_dict['input_bounds'][i, :])
            self.input_ax[i].set_ylabel(self.input_name[i],labelpad=1)

        # plot reward
        index_offset += self.n_input_model
        self.reward_ax = self.fig.add_subplot(self.env_dict['plot/column'], n_columns, index_offset)
        self.reward_line, = self.reward_ax.plot(self.x, self.r, 'g-')
        self.reward_ax.set_title('Generated reward')
        self.reward_ax.set_ylim(-1, 1)
        self.reward_ax.set_ylabel('$r$',labelpad=1)
        self.reward_ax.set_xlabel('$t$ [s]')

        # plot debug signal
        if self.env_dict['debug']:
            index_offset += 1
            self.debug_ax = self.fig.add_subplot(self.env_dict['plot/column'], n_columns, index_offset)
            self.debug_line, = self.debug_ax.plot(self.x, self.r, 'g-')
            self.debug_ax.set_ylabel('$debug_signal$')
            self.debug_ax.set_xlabel('$t$ [s]')

        # set figure parameter
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)
        if full_screen:
            mng = plt.get_current_fig_manager()
            mng.window.state('zoomed')

    def step(self, action):
        """
        Take one simulation step
        :param action: np.array([], dtype=np.float32) with dimensions [1,]
        :return:
        """
        # done flag is set to false and gets overwritten if the episode ends
        done = False
        #print(self.n_max)
        # info is initialized empty
        info = {}
        self.action = action
        # transform (last) action range [-1, 1] to simulink model input range
        input_plant = self.input_d / 2 * (self.action + 1) + self.env_dict['input_bounds'][:, 0]
        for i in range(self.env_dict['substeps']):
            self.u_v[i, :] = input_plant
        #print(self.u_v)
        tout, y, x = signal.lsim(self.sys, self.u_v, self.T_v, self.state)
        #y, tout, x = cnt.lsim(self.sys2, self.u_v, self.T_v, self.state)
        #tout, y, x = control.matlab.lsim(self.sys, self.u_v, self.T_v, self.state)
        self.state = x[-1, :]
        output = y[-1, :]
        #print(output)
        # compute observation vector
        #print(self.y_ref)
        observation = self.calc_observation(output)
        #print(observation)
        #observation = np.concatenate((self.state, np.reshape(self.y_ref[:,0], [1, 2])[0]), 0)
        # here the reward function can be designed
        reward = self.reward_function(output)

        #print(self.i_step)
        # log state, action and reward in the corresponding vector
        self.y[:, self.i_step] = output
        self.u[:, self.i_step] = input_plant
        self.r[self.i_step] = reward
        self.debug[self.i_step] = self.IT_e[0]

        # check if the maximum trajectory length is reached
        if self.i_step >= self.n_max:
            self.force_render = True
            self.save_figure()

        # Count simulation steps
        self.i_step = self.i_step + 1
        if done:
            print(done)
        self.render()

        return observation, reward, done, info

    def reward_function(self, output):
        error_sum = 0
        n_control = 0
        for i in range(self.n_output_model):
            if not np.isnan(self.y_ref[i, self.i_step]):
                error_sum += abs((self.y_ref[i, self.i_step] - output[i])/self.output_d[i])
                n_control += 1
        reward = 1 - 1 * (np.clip(1/n_control * error_sum, 0, 1)) ** (1/2)

        #for i in range(self.n_output_model):
        #    if not (self.env_dict['penalty_output_bounds'][i, 0] < output[i] < self.env_dict['penalty_output_bounds'][i, 1]):
        #        reward = -1

        return reward

    def render(self):
        """
        renders the Env
        :param mode: mode='human'
        :return:
        """
        # to increase performance render only every "render_step"
        if (self.i_step % self.env_dict['render_step'] == 0 or self.force_render) and self.env_dict['render']:
            self.force_render = False
            # update state data in plot
            for i in range(self.n_output_model):
                self.output_line[i][0].set_ydata(self.y[i, :])
            # update action data in plot
            for i in range(self.n_input_model):
                self.input_line[i][0].set_ydata(self.u[i, :])
            # update reward data in plot
            self.reward_line.set_ydata(self.r)
            # rescale reward plot
            # self.reward_ax.relim()
            # self.reward_ax.autoscale()
            if self.env_dict['debug']:
                self.debug_line.set_ydata(self.debug)
                self.debug_ax.relim()
                self.debug_ax.autoscale()


            # update figure
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def reset(self):
        """
        Resets the environment
        :return:
        """
        # reset sim step counter to 0
        self.i_step = 0
        self.action = 0
        self.e = np.zeros([self.n_output_model])
        self.int_e = np.zeros([self.n_output_model])
        self.state = np.zeros([self.n_output_model])
        self.IT_e = np.zeros([sum(self.env_dict['controlled_output'])])
        self.state_IT = np.zeros([len(self.den)-1, sum(self.env_dict['controlled_output'])])
        self.generate_reference()
        self.generate_initial_states()
        # state = np.reshape(self.state, [self.n_state_model, 1])
        output = np.matmul(self.c, self.state)
        #print(output)
        # reset state vector
        self.y = np.zeros([self.n_output_model, self.n_max])
        # reset action vector
        self.u = np.zeros([self.n_input_model, self.n_max])
        # reset action vector
        self.r = np.zeros([self.n_max])
        self.debug = np.zeros([self.n_max])

        # compute observation vector
        observation = self.calc_observation(output)
        #print(observation)
        #observation = np.concatenate((self.state, np.reshape(self.y_ref[:,0], [1, 2])[0]), 0)
        # update plots
        if self.env_dict['render']:
            for i in range(self.n_output_model):
                self.output_line[i][0].set_ydata(self.y[i, :])
                self.output_line_r[i][0].set_ydata(self.y_ref[i, :])
            for i in range(self.n_input_model):
                self.input_line[i][0].set_ydata(self.u[i, :])
            self.reward_line.set_ydata(self.r)
            # self.reward_ax.relim()
            # self.reward_ax.autoscale()
            # self.fig.canvas.draw()
            # self.fig.canvas.flush_events()

        return observation

    def close(self):
        """
        Close function normally closes the viewer for rendering
        :return:
        """
        # close figure
        plt.close()

    def calc_observation(self, output):
        self.calc_e(output)
        #print(output)
        # create observation vector
        hidden_output = np.delete(output, self.index, 0)
        #print("hidden output ",hidden_output)
        hidden_output_d = np.delete(self.output_d, self.index, 0)
        #print("hidden output d ",hidden_output_d)
        observation = np.array(np.divide(hidden_output, hidden_output_d), dtype=np.float32)
        #print("obse ",observation)
        # for i in range(self.n_output_model):
        #     if not np.isnan(self.y_ref[i, 0]):
        #         self.e[i] = self.y_ref[i, 0] - output[i]
        #         self.int_e[i] += self.env_dict['Ts'] * math.tanh(self.e[i] / self.output_d[i] * 100)
        #
        #         observation = np.append(observation, self.e[i] / self.output_d[i])
        #         if self.env_dict['int_e']:
        #             observation = np.append(observation, self.int_e[i] / self.env_dict['T_episode'])
        #             # observation = np.append(observation, self.y_ref[i, 0] / self.output_d[i])
        # if self.env_dict['IT_e']:
        #     observation = np.append(observation, self.IT_e / self.env_dict['T_episode'])
        #     print(self.IT_e)

        observation = np.append(observation, self.e)
        #print("obse ",observation)
        #print(self.e)

        if self.env_dict['int_e']:
            observation = np.append(observation, self.int_e / self.env_dict['T_episode'])
        if self.env_dict['IT_e']:
            observation = np.append(observation, self.IT_e / self.env_dict['T_episode'])
        #     print(self.IT_e)

        return observation

    def calc_e(self, output):
        n = 0
        j = 0 
        # for i in range(self.n_output_model):
        #     k= 250
        #     if not np.isnan(self.y_ref[i, j+250]):
        #         self.e[n] = (self.y_ref[i, j+250] - output[i]) / self.output_d[i]
        #         #print(output[i])
        #         n += 1
        for i in range(self.n_output_model):
            if not np.isnan(self.y_ref[i, self.i_step]):
                self.e[n] = (self.y_ref[i, self.i_step] - output[i]) / self.output_d[i]
                n += 1
        #print(self.y_ref)
        if self.env_dict['int_e']:
            for i in range(sum(self.env_dict['controlled_output'])):
                self.int_e[i] += self.env_dict['Ts'] * math.tanh(self.e[i] * 100)
        if self.env_dict['IT_e']:
            for n in range(self.env_dict['substeps']):
                self.e_v[n, :] = self.e
            for i in range(sum(self.env_dict['controlled_output'])):
                tout, y, x = signal.lsim(self.IT, self.e_v[:, i], self.T_v, self.state_IT[:, i])
                self.state_IT[:, i] = x[-1, :]
                self.IT_e[i] = y[-1]

    # def calc_IT_e(self, e):
    #     for n in range(self.env_dict['substeps']):
    #         self.e_v[n, :] = e
    #     for i in range(sum(self.env_dict['controlled_output'])):
    #         tout, y, x = signal.lsim(self.IT, self.e_v[:, i], self.T_v, self.state_IT[:, i])
    #         self.state_IT[:, i] = x[-1, :]
    #         self.IT_e[i] = y[-1]

    def save_figure(self):
        if self.env_dict['render'] and self.env_dict['save_fig']:
            for i in range(self.n_input_model):
                self.input_ax[i].relim()
                self.input_ax[i].autoscale()
            for i in range(self.n_output_model):
                self.output_ax[i].relim()
                self.output_ax[i].autoscale()
            self.reward_ax.relim()
            self.reward_ax.autoscale()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            directory = pathlib.Path(__file__).parent.resolve().__str__() + '/figures'
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(directory + '/eval_plot.png', format="png")
            for i in range(self.n_input_model):
                self.input_ax[i].set_ylim(self.env_dict['input_bounds'][i, :])
            for i in range(self.n_output_model):
                self.output_ax[i].set_ylim(self.env_dict['output_bounds'][i, :])

    def generate_reference(self):
        for i in range(self.n_output_model):
            if self.env_dict['controlled_output'][i] == 0:
                self.y_ref[i, :] = None
            else:
                k= 25
                for j in range(100):
                    self.y_ref[i][j*k:j*k+k]= np.random.uniform(self.env_dict['reference_output_bounds'][i, 0],
                                                             self.env_dict['reference_output_bounds'][i, 1],
                                                             size=(1,))
                #self.y_ref[i, :] = np.random.uniform(self.env_dict['reference_output_bounds'][i, 0],
                #                                     self.env_dict['reference_output_bounds'][i, 1],
                 #                                    size=(1,))

                    
                    

    def generate_initial_states(self):
        for i in range(self.n_state_model):
            self.state[i] = np.random.uniform(low=self.env_dict['initial_output_bounds'][i, 0],
                                              high=self.env_dict['initial_output_bounds'][i, 1],
                                              size=(1,)).astype(np.float32)
        

    @staticmethod
    def bounds(n):
        """
        Generate np.array with ones in size of n
        :param n: vector size (int)
        :return bonds_v: (np.array)
        """
        bounds_v = np.ones((n,), dtype=np.float32)
        return bounds_v
