from gym.envs.registration import register

register(id = 'catcher-v0', 
	entry_point = 'catcher.envs:World',)

register(id = 'catcher-v1', 
	entry_point = 'catcher.envs:WorldWithSpeed',)

register(id = 'catcher-v2', 
	entry_point = 'catcher.envs:JacoWorld',)

register(id = 'catcher-v3', 
	entry_point = 'catcher.envs:JacoWithSpeed',)
