# Cartpole

REINFORCE is a simple RL algorithm(perhaps too simple), which uses stochatic policy gradient function to find out which action would have the highest probability to give the maximum reward in a particular state. 

A policy gradient method is a reinforcement learning approach that tries to directly learn a policy by generally using a parameterized function as a policy function
(e.g., a neural network) and training it to increase the probability of actions based on the observed rewards.

REINFORCE is the simplest implementation of a policy gradient method; it essentially maximizes the probability of an action times the observed reward after tak-
ing that action, such that each action’s probability (given a state) is adjusted according to the size of the observed reward. REINFORCE is an effective and very simple way of training a policy function, but it’s a little too simple. For CartPole it works very well, since the state space is very small and there are only two actions. If we’re dealing with an environment with many more possible actions, reinforcing all of them each episode and hoping that on average it will only reinforce the good actions becomes less and less reliable.

## Training examples: 

### Initial Stages 

![ezgif com-gif-maker](https://user-images.githubusercontent.com/41234408/111776519-20d3d880-88d8-11eb-9b24-28ea86c63edf.gif)

Here the pole keeps falling of pretty soon when the training is in the initial stages. 

### After training

![ezgif com-gif-maker (1)](https://user-images.githubusercontent.com/41234408/111776810-790ada80-88d8-11eb-9e05-2c781d87077a.gif)

Now the cart is able to travel further without the pole dropping. This distance can be increased by making the network a bit deeper(I have used a very simple sequential network), and by increasing the number of maximum steps from 200 to a greater number. For this you can edit the `MAX_DUR` variable. 
