#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 23:39:13 2017

@author: sezan92
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 23:39:13 2017

@author: sezan92
"""
# In[1] Libraries
import gym
import numpy as np
from collections import deque
import random
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
# In[2]
env = gym.make('MountainCar-v0')
# In[Constants]
legal_actions=env.action_space.n
actions = [0,1]
gamma =0.95
lr =0.5
num_episodes =1000
epsilon =1
epsilon_decay =0.995
memory_size =1000
batch_size=100
show=True
action_size=env.action_space.shape[0]
state_size=env.observation_space.shape[0]
# In[Memory]
factor=[1,100]
memory=deque(maxlen=memory_size)
s=env.reset()
s = s.reshape((1,-1))
s = s*factor
a=env.action_space.sample()
new_s,r,d,_ =env.step(a)
new_s = new_s.reshape((1,-1))
new_s = new_s*factor
experience=(s,r,a,new_s)
memory.append(experience)
s = new_s
for _ in range(memory_size):
    a=env.action_space.sample()
    new_s,r,d,_ =env.step(a)
    new_s = new_s.reshape((1,-1))
    new_s = new_s*factor
    if show:
        env.render()
    if d:
        r=-100
        experience =(s,r,a,new_s)
        s=env.reset()
        s = s.reshape((1,-1))
    else:    
        experience =(s,r,a,new_s)
    memory.append(experience)
    s = new_s
env.close()  
# In[Deep Q learning]
model = Sequential()
model.add(Dense(100,activation='relu',input_shape=(1,state_size)))
model.add(Dense(100,activation='relu'))
model.add(Dense(action_size,activation='linear'))
model.compile(loss='mse',optimizer=Adam(lr=0.01),)
model.summary()
# In[4]
if ep_list is None and reward_list is None:
    ep_list =[]
    reward_list =[] 
index=0 
for ep in range(num_episodes):
    s= env.reset()
    s=s.reshape((1,-1))
    s = s*factor
    rAll =0
    d = False
    j = 0
    for j in range(200):
        #time.sleep(0.01)
        #epsilon greedy. to choose random actions initially when Q is all zeros
        if np.random.random()< epsilon:
            a = np.random.randint(0,legal_actions)
            #epsilon = epsilon*epsilon_decay
        else:
            Q = model.predict(s.reshape(-1,s.shape[0],s.shape[1]))
            a =np.argmax(Q)
        new_s,r,d,_ = env.step(a)
        new_s = new_s.reshape((1,-1))
        new_s = new_s*factor
        rAll=rAll+r
        if show:
            env.render()
        if d:
            if rAll<-199:
                r =-100
                experience = (s,r,a,new_s)
                memory.append(experience)
                print("Episode %d, Failed! Reward %d"%(ep,rAll))
                #break
            elif rAll<-110 and rAll>-199:
                r=-10
                experience = (s,r,a,new_s)
                memory.append(experience)
                print("Episode %d, Better! Reward %d"%(ep,rAll))
            elif rAll>=-110:
                r=100
                experience = (s,r,a,new_s)
                memory.append(experience)

                print("Episode %d, Passed! Reward %d"%(ep,rAll))
            ep_list.append(ep)
            reward_list.append(rAll)
            break
        
        experience = (s,r,a,new_s)
        memory.append(experience)
        if j==199:
            print("Reward %d after full episode"%(rAll))
            
        s = new_s
    batches=random.sample(memory,batch_size)
    #batches= list(memory)[index:index+batch_size]
    states= np.array([batch[0] for batch in batches])
    rewards= np.array([batch[1] for batch in batches])
    actions= np.array([batch[2] for batch in batches])
    new_states= np.array([batch[3] for batch in batches])
    Qs =model.predict(states)
    new_Qs = model.predict(new_states)
    for i in range(len(rewards)):
        if rewards[i]==-100 or rewards[i]==-10:
            Qs[i][0][actions[i]]=Qs[i][0][actions[i]]+ lr*(rewards[i]-Qs[i][0][actions[i]])
        else:
            Qs[i][0][actions[i]]= Qs[i][0][actions[i]]+ lr*(rewards[i]+gamma*np.max(new_Qs[i])-Qs[i][0][actions[i]])
    model.fit(states,Qs,verbose=0)
    epsilon=epsilon*epsilon_decay
    index=index+batch_size
    if index>=len(memory):
        index=0
env.close()       
# In[moving average]

def moving_average (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma
# In[plot]
rewards = moving_average(reward_list,10)
plt.plot(rewards)
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Reward vs Episodes for Deep Q Learning')
plt.savefig('Reward_vs_Episode_DL_lr_%f_eps_%d.jpg'%(lr,ep+1))