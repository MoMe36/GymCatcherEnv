from gym.envs.registration import register

register(id = 'catcher-v0', 
	entry_point = 'catcher.envs:World',)

register(id = 'catcher-v1', 
	entry_point = 'catcher.envs:WorldWithSpeed',)

