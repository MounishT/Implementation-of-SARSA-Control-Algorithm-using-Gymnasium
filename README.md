# Implementation-of-SARSA-Control-Algorithm-using-Gymnasium

## Aim

To implement the **SARSA control algorithm** using the Gymnasium `FrozenLake-v1` environment and learn an action-value function that helps the agent select better actions for reaching the goal state while avoiding holes.

---
## Problem Statement

Implement the **SARSA (State-Action-Reward-State-Action)** reinforcement learning control algorithm using the Gymnasium `FrozenLake-v1` environment.

The agent must learn an optimal policy by interacting with the environment and updating its Q-values based on the action actually selected in the next state.

The learned Q-table, state-value function, policy, average reward, and learning curve are used to evaluate the performance of the agent.



## Software Requirements

* Python 3.x
* Jupyter Notebook / Google Colab
* Gymnasium
* NumPy
* Matplotlib


## Environment Description


The environment used for this SARSA agent training is the `FrozenLake-v1` from the Gymnasium library. Here's a breakdown of its characteristics:

*   **Grid World:** It's a 4x4 grid where each cell represents a state.
*   **States:** There are 16 states in total, indexed from 0 to 15.
    *   **Start (S):** State 0, typically at the top-left corner.
    *   **Frozen (F):** Safe frozen surfaces, where the agent can move.
    *   **Hole (H):** Dangerous holes. Stepping into a hole ends the episode, and the agent receives no reward.
    *   **Goal (G):** State 15, typically at the bottom-right corner. Reaching the goal ends the episode and provides a reward of +1.
*   **Actions:** The agent can perform four discrete actions:
    *   0: Left
    *   1: Down
    *   2: Right
    *   3: Up
*   **Slippery Surface (`is_slippery=True`):** This is a crucial aspect of this environment. When `is_slippery` is set to `True`, the agent's actions are stochastic. This means that if the agent chooses to move in a certain direction, there's a chance (usually 1/3 for the intended direction and 1/3 for each of the two perpendicular directions) that it will slide to an adjacent tile instead of the intended one. This makes the environment more challenging and necessitates learning a robust policy that accounts for uncertainty.
*   **Reward Function:**
    *   +1 for reaching the Goal state (G).
    *   0 for reaching any other state (Frozen or Hole).
*   **Episode Termination:** An episode ends when the agent reaches the Goal, falls into a Hole, or exceeds a maximum number of steps (defined during training, `max_steps_per_episode`).

## Theory

SARSA stands for:

$$
S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1}
$$

It updates the Q-value using the action actually selected in the next state.

The SARSA update rule is:

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha
\left[
R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)
\right]
$$

Where:

| Symbol | Meaning |
|---|---|
| $S_t$ | Current state |
| $A_t$ | Current action |
| $R_{t+1}$ | Reward received after taking action $A_t$ |
| $S_{t+1}$ | Next state |
| $A_{t+1}$ | Next action selected using the current policy |
| $\alpha$ | Learning rate |
| $\gamma$ | Discount factor |
| $Q(s,a)$ | Action-value function |

---

## Epsilon-Greedy Policy

SARSA uses an epsilon-greedy policy for action selection.

With probability $\epsilon$, the agent explores by selecting a random action.

With probability $1-\epsilon$, the agent exploits by selecting the action with the highest Q-value.

$$
a =
\begin{cases}
\text{random action}, & \text{with probability } \epsilon \\
\arg\max_a Q(s,a), & \text{with probability } 1-\epsilon
\end{cases}
$$

---


## Algorithm

1. Create the `FrozenLake-v1` environment.
2. Initialize the Q-table with zeros.
3. Set the learning rate `α`.
4. Set the discount factor `γ`.
5. Initialize epsilon for the epsilon-greedy policy.
6. For every training episode:

   * Reset the environment.
   * Select the initial action using the epsilon-greedy policy.
7. For every step:

   * Execute the selected action.
   * Observe the next state and reward.
   * If the episode has terminated, update the Q-value using the reward.
   * Otherwise, select the next action using the epsilon-greedy policy.
   * Apply the SARSA update rule.
   * Move to the next state and action.
8. Decrease epsilon after each episode.
9. Repeat until all training episodes are completed.
10. Calculate the state-value function from the learned Q-table.
11. Extract the learned policy using the action with the highest Q-value for every state.
12. Calculate the average reward over the last 1000 episodes.
13. Plot the learning curve.



## Python Program

```python

# -------------------------------------------------
# SARSA Training
# -------------------------------------------------
episode_rewards = []

for episode in range(num_episodes):
    state, _ = env.reset() # Initial state
    action = epsilon_greedy_action(state, epsilon) # Initial action

    done = False
    truncated = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        next_state, reward, done, truncated, _ = env.step(action)
        next_action = epsilon_greedy_action(next_state, epsilon) # Select next action using epsilon-greedy

        # SARSA Update Rule
        Q[state, action] = Q[state, action] + alpha * \
                           (reward + gamma * Q[next_state, next_action] - Q[state, action])

        state = next_state
        action = next_action
        rewards_current_episode += reward

        if done or truncated:
            break

    # Epsilon decay
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    episode_rewards.append(rewards_current_episode)


# Derive state-value function and policy from the learned Q-table
state_values = np.max(Q, axis=1)
learned_policy = np.argmax(Q, axis=1)
```
---

## Output

Final Q-table:

<img width="243" height="295" alt="image" src="https://github.com/user-attachments/assets/0642a0cd-d78b-43ab-b19f-191b0638c197" />




Estimated State-Value Function:


<img width="286" height="105" alt="image" src="https://github.com/user-attachments/assets/35534b25-0d1a-4956-959b-4704bbf8f7e8" />





Learned Policy:


<img width="202" height="97" alt="image" src="https://github.com/user-attachments/assets/9a4690d0-472e-498d-a7e0-f21e1b49859b" />




Average reward over last 1000 episodes: 


<img width="372" height="27" alt="image" src="https://github.com/user-attachments/assets/558b31eb-90b6-4134-a4ae-551973fd404a" />

SARSA Learning Curve:

<img width="816" height="528" alt="image" src="https://github.com/user-attachments/assets/b7a32fb4-0d38-47a7-be19-2ef599199ab9" />


---

## Result

The SARSA control algorithm was successfully implemented using the Gymnasium FrozenLake-v1 environment. The agent learned the Q-values through repeated interaction with the environment using the SARSA update rule and an epsilon-greedy policy. The learned Q-table was then used to obtain the state-value function and the learned policy.

---

## Inference
SARSA learns by updating Q-values using the action actually selected in the next state. The epsilon-greedy policy allows the agent to explore different actions initially and gradually choose better actions as learning progresses. The final Q-table and learned policy show the agent's learned action preferences for reaching the goal while avoiding holes.


---

