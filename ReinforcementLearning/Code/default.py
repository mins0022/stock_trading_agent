import gym
env = gym.make("CartPole-v1")
observation = env.reset()
for _ in range(1000):
  env.render()
  action = env.action_space.sample() # your agent here (this takes random actions)
  #print(action)
  observation, reward, done, info = env.step(action)
  #print(observation, reward, info)
   
  if done:
    observation = env.reset()
env.close()
