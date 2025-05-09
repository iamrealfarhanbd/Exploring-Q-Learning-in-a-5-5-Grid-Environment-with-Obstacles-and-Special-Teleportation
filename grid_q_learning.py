import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
import os
import collections

class GridWorld:
    def __init__(self):
        self.rows, self.cols = 5, 5

        self.start = (1, 0)
        self.terminal = (4, 4)
        self.jump_start = (1, 3)
        self.jump_dest = (3, 3)
        self.obstacles = [(2, 2), (2, 3), (2, 4), (3, 2)]

        self.actions = {
            0: (-1, 0), # Up
            1: (+1, 0), # Down
            2: (0, +1), # Right
            3: (0, -1)  # Left
        }
        self.num_actions = len(self.actions)

    def reset(self):
        self.pos = self.start
        return self.pos

    def step(self, action):
        if action not in self.actions:
            raise ValueError(f"Invalid action: {action}")

        current_pos = self.pos
        r, c = current_pos

        if current_pos == self.jump_start:
            self.pos = self.jump_dest
            return self.pos, 5, True # Reward for jump, ends episode

        dr, dc = self.actions[action]
        next_r, next_c = r + dr, c + dc
        next_pos = (next_r, next_c)

        if not (0 <= next_r < self.rows and 0 <= next_c < self.cols) or next_pos in self.obstacles:
            next_pos = current_pos # Stay in current position if hit boundary or obstacle

        self.pos = next_pos

        reward = -1
        done = False

        if self.pos == self.terminal:
            reward = 10
            done = True

        return self.pos, reward, done

def run_experiment(alpha, max_episodes=100, early_stop_avg_reward=10, early_stop_window=30, gamma=0.9, epsilon=1.0, epsilon_decay=0.95, epsilon_min=0.01):
    env = GridWorld()

    Q = np.zeros((env.rows, env.cols, env.num_actions))

    rewards = []
    eps_history = []
    avg_abs_td_errors = []

    snapshots = {}
    snapshot_episodes_list = [10, 50]

    recent_rewards = collections.deque(maxlen=early_stop_window)
    stopped_early = False

    current_epsilon = epsilon

    for ep in range(1, max_episodes + 1):
        # This block is important for ensuring that if early stopping occurs,
        # the lists are padded to `max_episodes` length for consistent plotting.
        if stopped_early and len(rewards) < max_episodes:
            rewards.extend([rewards[-1]] * (max_episodes - len(rewards)))
            eps_history.extend([eps_history[-1]] * (max_episodes - len(eps_history)))
            avg_abs_td_errors.extend([avg_abs_td_errors[-1]] * (max_episodes - len(avg_abs_td_errors)))
            break # Break the loop after padding

        state = env.reset()
        done = False
        cum_r = 0
        episode_td_errors = []

        while not done:
            if random.uniform(0, 1) < current_epsilon:
                action = np.random.randint(env.num_actions)
            else:
                state_row, state_col = state
                q_values = Q[state_row, state_col, :]
                max_q = np.max(q_values)
                best_actions = np.where(q_values == max_q)[0]
                action = random.choice(best_actions)

            old_q = Q[state[0], state[1], action]
            next_state, reward, done = env.step(action)

            next_state_row, next_state_col = next_state
            max_future_q = np.max(Q[next_state_row, next_state_col, :]) # Max Q for next state
            td_target = reward + gamma * max_future_q
            td_error = td_target - old_q
            Q[state[0], state[1], action] += alpha * td_error # Q-learning update

            episode_td_errors.append(abs(td_error))

            state = next_state
            cum_r += reward

        rewards.append(cum_r)
        eps_history.append(current_epsilon)
        avg_abs_td_errors.append(np.mean(episode_td_errors) if episode_td_errors else 0)

        current_epsilon = max(epsilon_min, current_epsilon * epsilon_decay) # Epsilon decay

        recent_rewards.append(cum_r)
        if ep >= early_stop_window and np.mean(recent_rewards) > early_stop_avg_reward:
            print(f"Early stopping for alpha={alpha} at episode {ep} due to average reward exceeding {early_stop_avg_reward}")
            stopped_early = True

        # Capture snapshots of the Q-table at specified episodes
        if ep in snapshot_episodes_list or ep == max_episodes or (stopped_early and ep < max_episodes):
             if ep not in snapshots: # Avoid re-saving if already captured by early stopping
                snapshots[ep] = Q.copy()

    # Ensure lists are exactly max_episodes long, even if early stopped
    while len(rewards) < max_episodes:
        rewards.append(rewards[-1])
        eps_history.append(eps_history[-1])
        avg_abs_td_errors.append(avg_abs_td_errors[-1])

    return Q, rewards, eps_history, avg_abs_td_errors, snapshots

def plot_q_heatmap(q_table, env, title, filename, vmin=None, vmax=None):
    state_values = np.max(q_table, axis=2) # Get the maximum Q-value for each state
    rows, cols = env.rows, env.cols

    fig, ax = plt.subplots(figsize=(cols + 1, rows + 1))
    im = ax.imshow(state_values, cmap='viridis', origin='upper', vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label='Max Q-Value')
    ax.set_title(title)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(1, cols + 1))
    ax.set_yticklabels(np.arange(1, rows + 1))
    ax.grid(False)

    # Draw grid lines
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color='gray', lw=1)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color='gray', lw=1)

    # Mark obstacles
    for obs_r, obs_c in env.obstacles:
        ax.add_patch(Rectangle((obs_c - 0.5, obs_r - 0.5), 1, 1, color='black', alpha=0.7, ec='gray'))

    # Mark start state
    start_r, start_c = env.start
    ax.add_patch(Circle((start_c, start_r), 0.35, color='red', zorder=5)) # Center of the cell is (c, r)

    # Mark terminal state
    terminal_r, terminal_c = env.terminal
    ax.add_patch(Rectangle((terminal_c - 0.5, terminal_r - 0.5), 1, 1, color='cyan', alpha=0.7, ec='gray'))
    ax.text(terminal_c, terminal_r, '★', ha='center', va='center', color='yellow', fontsize=16, weight='bold')


    # Mark jump state and destination
    jump_start_r, jump_start_c = env.jump_start
    jump_dest_r, jump_dest_c = env.jump_dest
    ax.add_patch(FancyArrowPatch((jump_start_c, jump_start_r), (jump_dest_c, jump_dest_r),
                                         connectionstyle="arc3,rad=0.3", arrowstyle="-|>",
                                         mutation_scale=20, lw=2, color='magenta', zorder=5))
    ax.text(jump_start_c + 0.3, jump_start_r - 0.3, '+5', ha='left', va='bottom', color='purple', fontsize=10, weight='bold')


    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig) # Close the figure after saving to prevent display

def plot_policy_map(Q, env, title, filename):
    best_actions = np.argmax(Q, axis=2) # Get the index of the best action for each state
    rows, cols = env.rows, env.cols

    fig, ax = plt.subplots(figsize=(cols + 1, rows + 1))
    ax.imshow(np.ones((rows, cols)), cmap='gray', origin='upper', vmin=0, vmax=1) # Background
    ax.set_title(title)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(np.arange(1, cols + 1))
    ax.set_yticklabels(np.arange(1, rows + 1))
    ax.grid(False)

    # Draw grid lines
    for r in range(rows + 1):
        ax.axhline(r - 0.5, color='gray', lw=1)
    for c in range(cols + 1):
        ax.axvline(c - 0.5, color='gray', lw=1)

    action_arrows = { # Unicode arrows for directions
        0: '↑', # Up
        1: '↓', # Down
        2: '→', # Right
        3: '←'  # Left
    }

    for r in range(rows):
        for c in range(cols):
            cell = (r, c)
            color = 'white'
            if cell in env.obstacles: color = 'black'
            elif cell == env.terminal: color = 'cyan'
            elif cell == env.jump_start: color = 'purple'
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=color, edgecolor='gray'))

            if cell in env.obstacles:
                plt.text(c, r, 'X', ha='center', va='center', color='white', fontsize=12, weight='bold')
            elif cell == env.terminal:
                 plt.text(c, r, '★', ha='center', va='center', color='yellow', fontsize=16, weight='bold')
            elif cell == env.start:
                 ax.add_patch(Circle((c, r), 0.35, color='red', zorder=5))
            elif cell == env.jump_start:
                 plt.text(c, r, 'Jump', ha='center', va='bottom', color='white', fontsize=8, weight='bold')
            else:
                arrow = action_arrows[best_actions[r, c]]
                plt.text(c, r, arrow, ha='center', va='center', color='blue', fontsize=12, weight='bold')

    # Mark jump state and destination with arrow
    jump_start_r, jump_start_c = env.jump_start
    jump_dest_r, jump_dest_c = env.jump_dest
    ax.add_patch(FancyArrowPatch((jump_start_c, jump_start_r), (jump_dest_c, jump_dest_r),
                                         connectionstyle="arc3,rad=0.3", arrowstyle="-|>",
                                         mutation_scale=20, lw=2, color='magenta', zorder=5))
    ax.text(jump_dest_c + 0.3, jump_dest_r + 0.3, '+5', ha='left', va='top', color='purple', fontsize=10, weight='bold') # Reward label


    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close(fig) # Close the figure after saving to prevent display

def plot_cumulative_reward(episode_rewards_dict, title, filename):
    plt.figure(figsize=(10, 6))
    for alpha, rewards in episode_rewards_dict.items():
        plt.plot(range(1, len(rewards) + 1), rewards, label=f'α={alpha}')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_epsilon_decay(epsilon_values, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(epsilon_values) + 1), epsilon_values)
    plt.xlabel('Episode')
    plt.ylabel('Epsilon ($\epsilon$)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_average_td_error(avg_td_errors_dict, title, filename):
     plt.figure(figsize=(10, 6))
     for alpha, errors in avg_td_errors_dict.items():
         plt.plot(range(1, len(errors) + 1), errors, label=f'α={alpha}')
     plt.xlabel('Episode')
     plt.ylabel('Average Absolute TD Error')
     plt.title(title)
     plt.legend()
     plt.grid(True)
     plt.tight_layout()
     plt.savefig(filename)
     plt.close()

def plot_average_final_reward_bar_chart(learning_rates, average_rewards, title, early_stop_window, filename):
    plt.figure(figsize=(8, 6))
    bars = plt.bar([str(lr) for lr in learning_rates], average_rewards, color=['blue', 'green', 'red'])
    plt.xlabel('Learning Rate (α)')
    plt.ylabel(f'Average Cumulative Reward (Last {early_stop_window} Episodes)')
    plt.title(title)
    plt.grid(axis='y', linestyle='--')

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.2f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

if __name__ == "__main__":
    env = GridWorld()

    learning_rates = [0.1, 0.3, 0.7]

    results = {}

    max_episodes = 100
    early_stop_window = 30
    early_stop_threshold = 10

    v_max_q = 10.0
    v_min_q = -10.0

    # Create results directory if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        print(f"Created directory: {results_dir}")
    else:
        print(f"Directory already exists: {results_dir}")


    all_average_final_rewards = []
    for alpha in learning_rates:
        print(f"Running experiment with learning rate alpha = {alpha}...")
        Q, rewards, eps_hist, avg_td_errors, snapshots = run_experiment(
            alpha=alpha,
            max_episodes=max_episodes,
            early_stop_avg_reward=early_stop_threshold,
            early_stop_window=early_stop_window,
            gamma=0.9,
            epsilon=1.0,
            epsilon_decay=0.95,
            epsilon_min=0.01
        )
        results[alpha] = {
            'Q': Q,
            'rewards': rewards,
            'eps_history': eps_hist,
            'avg_td_errors': avg_td_errors,
            'snapshots': snapshots,
        }
        actual_episodes_run = len(rewards)
        if actual_episodes_run >= early_stop_window:
             avg_final_reward = np.mean(rewards[actual_episodes_run - early_stop_window : actual_episodes_run])
        else:
             avg_final_reward = np.mean(rewards) # If fewer episodes than window, average all available

        all_average_final_rewards.append(avg_final_reward)
        print(f"Finished alpha={alpha}. Actual episodes run: {actual_episodes_run}. Average reward over final period: {avg_final_reward:.2f}")

    print("\nGenerating plots...")

    plot_cumulative_reward({alpha: res['rewards'] for alpha, res in results.items()},
                              'Cumulative Reward vs Episode for Different Learning Rates',
                              os.path.join(results_dir, 'cumulative_reward.png'))

    first_alpha = learning_rates[0] # Epsilon decay is the same for all runs
    plot_epsilon_decay(results[first_alpha]['eps_history'],
                       'Epsilon ($\epsilon$) Decay Schedule over Episodes',
                       os.path.join(results_dir, 'epsilon_decay.png'))

    plot_average_td_error({alpha: res['avg_td_errors'] for alpha, res in results.items()},
                          'Average Absolute TD Error vs Episode for Different Learning Rates',
                          os.path.join(results_dir, 'average_td_error.png'))

    plot_average_final_reward_bar_chart(learning_rates, all_average_final_rewards,
                                        f'Average Cumulative Reward by Learning Rate (Last {early_stop_window} Episodes)',
                                        early_stop_window,
                                        os.path.join(results_dir, 'average_final_reward_bar_chart.png'))

    all_snapshot_episodes = sorted(list(set([ep for alpha_results in results.values() for ep in alpha_results['snapshots'].keys()])))

    print("\nGenerating Q-Table and Policy snapshot plots...")
    for alpha in learning_rates:
         print(f"  For alpha = {alpha}:")
         sorted_snapshots = sorted(results[alpha]['snapshots'].items()) # Ensure snapshots are processed in order

         for episode, Q_snap in sorted_snapshots:
             print(f"    Plotting snapshot for Episode {episode}")
             temp_env = GridWorld() # Re-initialize env to ensure clean slate for plotting (though not strictly necessary here)

             plot_q_heatmap(Q_snap, temp_env,
                            f'Q-Table Heatmap: α = {alpha}, Episode {episode}',
                            os.path.join(results_dir, f'q_heatmap_alpha{str(alpha).replace(".", "_")}_ep{episode}.png'),
                            vmin=v_min_q, vmax=v_max_q)

             plot_policy_map(Q_snap, temp_env,
                             f'Greedy Policy Map: α = {alpha}, Episode {episode}',
                             os.path.join(results_dir, f'policy_map_alpha{str(alpha).replace(".", "_")}_ep{episode}.png'))

    print("\nAll plots generated and saved in the 'results' folder.❤️")