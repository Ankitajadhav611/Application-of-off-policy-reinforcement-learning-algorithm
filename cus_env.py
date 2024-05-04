import gymnasium as gym
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from scipy import signal
import pandas as pd



class CustomEnv(gym.Env):

        metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}
        
        def __init__(self):
            super(CustomEnv,self).__init__()

            #observation space - o/p of controller & y_ref value
            self.obs_dim = 4
            self.y_ref_dim = 2 
            obs_low = np.array([-10,-10,-10,10]).astype(np.float32)
            obs_high = np.array([10,10,10,10]).astype(np.float32)
            self.observation_space = spaces.Box(low= obs_low, high= obs_high)

            #action space - i/p of controller
            act_low = np.array([-150,-150]).astype(np.float32)
            act_high = np.array([150,150]).astype(np.float32)
            self.act_dim = 2
            self.action_space = spaces.Box(low = act_low, high = act_high)

            #poles
            self.poles = [-4,-4]

            #time series
            self.start_time = 0 

            #initial state 
            self.state_dim = 2
            #self.state = np.zeros(self.state_dim = 2)
            self.state  = np.array([ 0.0, 0.0]) 

            #initialize action 
            #self.action = np.random.rand(1,2)[0]
            
            #initialize time series
            self.T  = np.array([ 0.0])

            #counter 
            self.i_step = 0

            #sample & episode duration
            self.ep_timesteps = 10
            self.Ts = 0.1
            
            #system parameters
            self.n_output_model = 2
            self.n_input_model = 2
            
            # system parameter 
            self.a = np.array([[-6.0, 4.0], [8.0, -10.0]])
            self.b = np.array([[1.0, 0.0], [0.0, 1.0]])
            self.c = np.array([[1.0, 0.0],[0.0, 1.0]])
            self.d = np.array([[0.0,0.0],[0.0,0.0]])
            
            self.sys = signal.StateSpace(self.a, self.b, self.c, self.d)
            
            # initialize simulation time vector

            self.n_max = int(self.ep_timesteps / self.Ts) + 1

            self.t = np.linspace(0, self.ep_timesteps, self.n_max)

            # initialize state trajectory vector
            #state vector
            self.y = np.zeros([self.n_output_model, self.n_max])

            self.y_ref = np.zeros([self.n_output_model, self.n_max])
            for i in range(self.n_output_model):
                self.y_ref[i,:] = np.random.uniform(-10,10,size=(1,))
            # initialize action vector

            self.u = np.zeros([self.n_input_model, self.n_max])

            # initialize reward vector
            self.r = np.zeros(self.n_max)
            
            self.action  = np.array([[0.0, 0.0]])

            #render
            self.render_step = 10
            #self.render = False
            
            self.fig = plt.figure()
            self.output_ax1 = self.fig.add_subplot(4, 1, 1)
            self.output_ax2 = self.fig.add_subplot(4, 1, 2)
            self.input_ax = self.fig.add_subplot(4, 1, 3)
            self.reward = self.fig.add_subplot(4, 1, 4)

            self.output_ax1.set_ylim(-10,10)
            self.output_ax2.set_ylim(-10,10)
            self.input_ax.set_ylim(-150,150)
            self.reward.set_ylim(-10,10)

            self.output_line_1, = self.output_ax1.plot(self.t, self.y[0,:])
            self.output_line_1r, = self.output_ax1.plot(self.t, self.y_ref[0,:])
            self.output_line_2,  = self.output_ax2.plot(self.t, self.y[1,:])
            self.output_line_2r, = self.output_ax2.plot(self.t, self.y_ref[1,:])
            
            self.input_line_1, = self.input_ax.plot(self.t,self.u[0,:])
            self.input_line_2, = self.input_ax.plot(self.t,self.u[1,:])

            self.reward_line, = self.reward.plot(self.t,self.r)



        def reward_function(self, y_out):
            e1 = self.y_ref[0, 0] - y_out[0]
            e2 = self.y_ref[1, 0] - y_out[1]
            reward = 1 - (0.5*(e1+e2))
            return reward

        def reset(self):
            #reset the step counter
            self.i_step = 0
            done = False 
            # reset state vector
            self.y = np.zeros([self.n_output_model, self.n_max])
            # reset action vector
            self.u = np.zeros([self.n_input_model, self.n_max])
            # reset reward vector
            self.r = np.zeros([self.n_max])
            # initial state,time series, reward
            for i in range(self.n_output_model):
                self.y_ref[i,:] = np.random.uniform(-10,10,size=(1,))
                self.state[i] = np.random.uniform(-10,10,size=(1,))
            self.T = np.array([0.0])
            self.action  = np.array([[0.0, 0.0]]) 

            self.output_line_1.set_ydata(self.y[0,:])
            self.output_line_2.set_ydata(self.y[1,:])
            self.input_line_1.set_ydata(self.u[0,:])
            self.input_line_2.set_ydata(self.u[1,:])
            self.reward_line.set_ydata(self.r)

            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


            observation = np.concatenate((self.state, np.reshape(self.y_ref[:,0], [1, 2])[0]), 0)
            
            return observation
        

        def step(self, action):
            done = False 
            self.T = np.array([self.T[-1], self.T[-1]+ 0.1])
            self.action = np.array([self.action[-1], action])
            tout, y, x = signal.lsim(self.sys, self.action, self.T, self.state)
            self.state = x[-1]
            y_out = y[-1]
            #print(y_out)
            r = self.reward_function(y_out)
            self.r[self.i_step] = r
            self.y[0][self.i_step] = y_out[0]
            self.y[1][self.i_step] = y_out[1]
            self.u[0][self.i_step] = action[0]
            self.u[1][self.i_step] = action[1]
            self.i_step +=1
            if self.i_step >= self.n_max - 1:
                done = True
            observation = np.concatenate((self.state, np.reshape(self.y_ref[:,0], [1, 2])[0]), 0)
    
            return observation , r ,done , {}
        
        def render(self):
            if self.i_step % self.render_step == 0:
                self.output_line_1.set_ydata(self.y[0,:])
                self.output_line_2.set_ydata(self.y[1,:])
                self.input_line_1.set_ydata(self.u[0,:])
                self.input_line_2.set_ydata(self.u[1,:])
                self.reward_line.set_ydata(self.r)

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

 
        







        


            
        






            
            


