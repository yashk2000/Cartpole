import gym 
env  = gym.make('CartPole-v0')

state1 = env.reset()
action = env.action_space.sample()
state, reward, done, info = env.step(action)

import numpy as np
import torch 

l1 = 4
l2 = 150 
l3 = 2

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2), 
    torch.nn.LeakyReLU(), 
    torch.nn.Linear(l2, l3), 
    torch.nn.Softmax()
)

learning_rate = 0.0009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

pred = model(torch.from_numpy(state1).float())
action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())
state2, reward, done, info = env.step(action)

def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma, torch.arange(lenr).float()) * rewards
    disc_return /= disc_return.max()
    return disc_return

def loss_fn(preds, r):
    return -1 * torch.sum(r * torch.log(preds))

def running_mean(x, N=50):
    kernel = np.ones(N)
    conv_len = x.shape[0]-N
    y = np.zeros(conv_len)
    for i in range(conv_len):
        y[i] = kernel @ x[i:i+N]
        y[i] /= N
    return y

MAX_DUR = 200
MAX_EPISODES = 500
gamma = 0.99
score = []
for episodes in range(MAX_EPISODES):
    curr_state  = env.reset()
    done = False
    transitions = []

    for t in range(MAX_DUR): 
        act_prob = model(torch.from_numpy(curr_state).float()) 
        action = np.random.choice(np.array([0,1]), p=act_prob.data.numpy()) 
        prev_state = curr_state
        curr_state, _, done, info = env.step(action) 
        transitions.append((prev_state, action, t+1))
        env.render()
        if done: 
            break

    ep_len = len(transitions) 
    score.append(ep_len)
    reward_batch = torch.Tensor([r for (s,a,r) in transitions]).flip(dims=(0,)) 
    disc_returns = discount_rewards(reward_batch) 
    state_batch = torch.Tensor([s for (s,a,r) in transitions]) 
    action_batch = torch.Tensor([a for (s,a,r) in transitions])
    pred_batch = model(state_batch) 
    prob_batch = pred_batch.gather(dim=1,index=action_batch.long().view(-1,1)).squeeze() 
    loss = loss_fn(prob_batch, disc_returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

score = np.array(score)
avg_score = running_mean(score, 50)

plt.figure(figsize=(10,7))
plt.ylabel("Episode Duration",fontsize=22)
plt.xlabel("Training Epochs",fontsize=22)
plt.plot(avg_score, color='green')