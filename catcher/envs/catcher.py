import pymunk as pm
import numpy as np
from pymunk.vec2d import Vec2d as v2
import pygame as pg 
import gym 
from gym import spaces, error, utils 
from gym.utils import seeding 

collision_ball_type = 2
collision_bar_type = 1 

def vec2np(vec):
    return np.array([vec.x, vec.y])
def flip_pg(pos): 
    return [pos[0],  1. - pos[1]]



class Ball: 

	def __init__(self, scale, max_distance = 0.3, radius = 0.02): 

		x_pos = np.random.uniform(0.5-max_distance,0.5 + max_distance)
		y_pos = np.random.uniform(0.7,0.95)
		pos = np.array([x_pos, y_pos])
		# pos = np.array([0.5, 0.95])
				
		self.radius = radius*scale 
		self.scale = scale

		self.body = pm.Body(10,100)
		self.body.position = pos*scale 
		self.shape = pm.Circle(self.body, self.radius, (0,0))
		self.shape.collision_type = collision_ball_type

	def get_physics(self): 
		return self.body, self.shape 

	def get_position_and_speed(self): 

		speed = vec2np(self.body.velocity).copy()/self.scale
		position = vec2np(self.body.position).copy()/self.scale

		return position, speed

class Robot: 

	def __init__(self, nb_joints, joints_length, scale, space, max_torque = 0.1): 

		self.nb_joints = nb_joints
		self.joints_length = joints_length
		self.scale = scale
		self.space = space
		self.max_torque = max_torque

		self.base_pos = np.array([0.5,0.1])
		# self.angles = np.ones((self.nb_joints))*np.pi/2.
		self.angles = np.random.uniform(0., np.pi, (self.nb_joints))
		self.bar_rotation = 0.

		self.get_joints_pos()
		self.create_bar()

	def create_bar(self): 

		effector_pos = self.get_joints_pos()[-1]
		self.bar = Bar(effector_pos, self.bar_rotation, self.scale, self.space)

	def get_bar(self):

		return self.bar

	def step(self, a): 

		a = np.clip(a, -self.max_torque, self.max_torque)
		self.angles += a[:-1]
		self.angles = self.angles%(np.pi*2.)
		self.bar_rotation += a[-1]
		self.bar_rotation = self.bar_rotation%(np.pi*2.)

		self.create_bar()

	def get_joints_pos(self): 

		joints = np.hstack([np.cos(self.angles).reshape(-1,1), np.sin(self.angles).reshape(-1,1)])
		joints *= self.joints_length
		joints = np.vstack([self.base_pos, joints])
		joints = np.cumsum(joints, 0)

		return joints

class Bar: 

	def __init__(self, pos, rot, scale, space, length = 0.15): 

		self.body = pm.Body(body_type = pm.Body.STATIC)
		position = np.vstack([pos.reshape(-1), pos.reshape(-1)])

		decal_x = np.array([1.,-1.])*length*0.5*np.cos(rot)
		decal_y = np.array([1.,-1.])*length*0.5*np.sin(rot)
		position[:,0] += decal_x
		position[:,1] += decal_y

		self.rot = rot 
		self.length = length

		self.phy_position = position*scale

		self.draw_position = position 

		self.shape = pm.Segment(space.static_body, self.phy_position[0], self.phy_position[1], 1)
		self.shape.collision_type = collision_bar_type

	def get_physics(self):
		return self.shape

	def get_draw_infos(self): 
		return self.draw_position.copy(), self.rot, self.length



class World(gym.Env):

	metadata = {'render.modes':['human']}

	def __init__(self, nb_joints = 3, joints_length = 0.2, max_steps = 500, world_scale = 100.):

		super().__init__()

		self.render_ready = False 

		self.nb_joints = nb_joints
		self.joints_length = joints_length
		self.max_steps = max_steps

		self.max_torque = 0.1
		self.scale = world_scale

		self.initialize_world()

	def seed(self, seed): 

		return 

	def from_params(self, nb_joints, joints_length): 

		self.nb_joints = nb_joints
		self.joints_length = joints_length

		self.initialize_world()

	def initialize_world(self): 

		self.space = pm.Space()
		self.space.gravity = 0.,-10.
		self.steps = 0 
		

		self.first_contact = False 
		self.initialize_collisions()
		self.robot = Robot(self.nb_joints, self.joints_length, self.scale, self.space, self.max_torque)
		self.add_ball()
		self.add_bar()

		low_ac = -self.max_torque*np.ones((self.nb_joints+1))
		high_ac = self.max_torque*np.ones((self.nb_joints+1))

		# joints + bar rotation + ball position and speed + first contact 
		low_obs = -50.*np.ones((self.nb_joints + 1 + 4 + 1))
		high_obs = -50.*np.ones((self.nb_joints + 1 + 4 + 1))

		self.observation_space = spaces.Box(low_obs, high_obs, dtype = np.float)
		self.action_space = spaces.Box(low_ac, high_ac, dtype = np.float)

	def initialize_collisions(self): 

		self.collision_handler = self.space.add_collision_handler(collision_bar_type, collision_ball_type)
		self.collision_handler.post_solve = self.observe_contact

	def observe_contact(self, arbiter, space, data): 

		self.first_contact = True

	def create_new_ball(self):

		self.remove_ball()
		self.add_ball()

	def add_ball(self): 

		max_distance = 0.9*self.nb_joints*self.joints_length
		self.ball = Ball(self.scale, max_distance = max_distance)
		self.space.add(self.ball.get_physics())

	def remove_ball(self): 

		ball_phy = self.ball.get_physics()
		for b_phy in ball_phy: 
			self.space.remove(b_phy)

	def add_bar(self): 

		self.bar = self.robot.get_bar()
		self.space.add(self.bar.get_physics())

	def remove_bar(self): 

		self.space.remove(self.bar.get_physics())

	def update_bar(self): 

		self.remove_bar()
		self.add_bar()

	def step(self, action): 

		self.robot.step(action)
		self.update_bar()
		self.space.step(0.02)

		self.steps += 1
		
		return self.observe() 

	def random_step(self): 

		action = np.random.uniform(-self.max_torque,self.max_torque, (self.nb_joints+1))
		action[-1] = np.random.uniform(-0.05,0.05)
		self.robot.step(action)
		self.update_bar()
		self.space.step(0.02)

		self.steps += 1 
		
		return self.observe() 		

	def observe(self): 

		done = False 
		reward = 0. 
		info = {}

		robot_angles = self.robot.angles.copy()
		ball_pos, ball_speed = self.ball.get_position_and_speed()

		state = robot_angles.tolist() + \
			    [self.robot.bar_rotation] + \
			    ball_pos.tolist() + \
			    ball_speed.tolist() + \
			    [1 if self.first_contact else 0.]

		if(self.first_contact): 
			reward = ball_pos[1]

		if(ball_pos[1] <= 0.): 
			done = True 
			reward = -1.

		if(self.steps > self.max_steps): 
			done = True 
			reward = 1.

		return state, reward, done, info  

	def reset(self): 

		self.create_new_ball()
		self.steps = 0 
		self.first_contact = False 

		return self.observe()[0]

	def render(self): 

		if not self.render_ready: 
			self.render_init()

		time = 60 
		self.clock.tick(time)
		self.screen.fill((0,0,0))

		self.draw()
		pg.display.flip()

	def render_init(self, size = [700., 700.]): 

		pg.init()
		self.size = np.array(size)
		self.screen = pg.display.set_mode(np.array(size).astype(int))
		self.clock = pg.time.Clock()

		self.render_ready = True

	def draw(self): 

         # DRAW ROBOT

		joints_position = self.robot.get_joints_pos().copy()
		joints_position[:,1] = 1.- joints_position[:,1]
		joints_position *= self.size 

		for jp in joints_position: 
			pg.draw.circle(self.screen, (250, 250, 40), [int(jp[0]), int(jp[1])], 5)
		for i in range(1,len(joints_position)): 
			pg.draw.aaline(self.screen, (0, 250, 40), joints_position[i-1,:], joints_position[i,:], 1)


		bar_params = self.bar.get_draw_infos()[0]
		bar_params[:,1] = 1.-bar_params[:,1]

		pg.draw.aaline(self.screen, (230,50,150), bar_params[0,:]*self.size, bar_params[1,:]*self.size, 1)

        # DRAW BALL


		ball_pos,_  = self.size*self.ball.get_position_and_speed()
		ball_pos[1] = self.size[1] - ball_pos[1]
		pg.draw.circle(self.screen, (250, 15, 0), [int(ball_pos[0]), int(ball_pos[1])], 15, )
		pg.draw.circle(self.screen,  (250, 250, 250), [int(ball_pos[0]), int(ball_pos[1])], 10,)
		pg.draw.circle(self.screen,  (250, 15, 0), [int(ball_pos[0]), int(ball_pos[1])], 5,)

	def __repr__(self): 

		return 'Catcher with obs_space {}, action_space {}'.format(self.observation_space.shape[0],self.action_space.shape[0])

	def __str__(self): 
		return '\t Catcher with obs_space {}, action_space {}'.format(self.observation_space.shape[0],self.action_space.shape[0])

# world = World()

# for i in range(2000): 

#     # state, r, done, info  = world.random_step()
#     state, r, done, info  = world.random_step()
#     # world.step()
#     # print(state)
#     # print(r)
#     input(state)
#     world.render()
#     if done: world.reset()