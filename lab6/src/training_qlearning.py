import minigrid
import gymnasium as gym
import matplotlib.pyplot as plt
from learning_agents import QLearningAgent, train_generic_qlearning
from learning_agents import SARSA, train_generic_sarsa
import numpy as np
from four_rooms_wrapper import FourRoomsWrapper, minigrid_state_processor
import random


def train_qlearning(params, episodes, fixed_pos=True, fixed_agent=(6, 6), fixed_goal=(2, 2)):
    raw_env = gym.make("MiniGrid-FourRooms-v0", max_episode_steps=100)
    env = FourRoomsWrapper(raw_env, actions_map=[0, 1, 2], fixed_pos=fixed_pos, agent_start=fixed_agent, goal_pos=fixed_goal)

    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=params['beta'],
        discount=params['gamma'],
        epsilon=params['epsilon']
    )

    history = train_generic_qlearning(
        env=env,
        agent=agent,
        state_processor_func=minigrid_state_processor,
        episodes=episodes
    )

    return history


def train_sarsa(params, episodes, fixed_pos=True, fixed_agent=(6, 6), fixed_goal=(2, 2)):
    raw_env = gym.make("MiniGrid-FourRooms-v0", max_episode_steps=100)
    env = FourRoomsWrapper(raw_env, actions_map=[0, 1, 2], fixed_pos=fixed_pos, agent_start=fixed_agent, goal_pos=fixed_goal)

    agent = SARSA(
        n_actions=env.action_space.n,
        learning_rate=params['beta'],
        discount=params['gamma'],
        epsilon=params['epsilon']
    )

    history = train_generic_sarsa(
        env=env,
        agent=agent,
        state_processor_func=minigrid_state_processor,
        episodes=episodes
    )

    return history


def qlearning_experiments():
    experiments = [
    {'beta': 0.1, 'gamma': 0.99, 'epsilon': 0.1, 'label': 'Standard'},
    {'beta': 0.5, 'gamma': 0.9, 'epsilon': 0.3, 'label': 'Agresywny (Wysokie Beta/Epsilon)'},
    {'beta': 0.05, 'gamma': 0.99, 'epsilon': 0.05, 'label': 'Ostrożny (Niskie Beta/Epsilon)'}
    ]

    plt.figure(figsize=(10, 6))

    random.seed(42)
    np.random.seed(42)

    for exp in experiments:
        print(f"Trenowanie: {exp['label']}...")
        # Wygładzanie wyników (średnia ruchoma) dla czytelności wykresu
        history = train_qlearning(exp, episodes=1000)
        window = 50
        smoothed_history = np.convolve(history, np.ones(window) / window, mode='valid')
        plt.plot(smoothed_history, label=f"{exp['label']} ($gamma$={exp['gamma']}, $\\beta$={exp['beta']}, $epsilon$={exp['epsilon']})")

    plt.title("Porównanie zbieżności Q-learning w FourRooms")
    plt.xlabel("Epizod")
    plt.ylabel("Średnia nagroda (Moving Avg)")
    plt.legend()
    plt.grid(True)
    plt.show()


def sarsa_experiments():
    experiments = [
    {'beta': 0.1, 'gamma': 0.99, 'epsilon': 0.1, 'label': 'Standard'},
    {'beta': 0.5, 'gamma': 0.9, 'epsilon': 0.3, 'label': 'Agresywny (Wysokie Beta/Epsilon)'},
    {'beta': 0.05, 'gamma': 0.99, 'epsilon': 0.05, 'label': 'Ostrożny (Niskie Beta/Epsilon)'}
    ]

    plt.figure(figsize=(10, 6))

    random.seed(42)
    np.random.seed(42)

    for exp in experiments:
        print(f"Trenowanie: {exp['label']}...")
        # Wygładzanie wyników (średnia ruchoma) dla czytelności wykresu
        history = train_sarsa(exp, episodes=1000)
        window = 50
        smoothed_history = np.convolve(history, np.ones(window) / window, mode='valid')
        plt.plot(smoothed_history, label=f"{exp['label']} ($gamma$={exp['gamma']}, $\\beta$={exp['beta']}, $epsilon$={exp['epsilon']})")

    plt.title("Porównanie zbieżności SARSA w FourRooms")
    plt.xlabel("Epizod")
    plt.ylabel("Średnia nagroda (Moving Avg)")
    plt.legend()
    plt.grid(True)
    plt.show()


def sarsa_vs_qlearning():
    experiments = [
    {'beta': 0.1, 'gamma': 0.99, 'epsilon': 0.1, 'label': 'Standard'},
    {'beta': 0.5, 'gamma': 0.9, 'epsilon': 0.3, 'label': 'Agresywny (Wysokie Beta/Epsilon)'},
    {'beta': 0.05, 'gamma': 0.99, 'epsilon': 0.05, 'label': 'Ostrożny (Niskie Beta/Epsilon)'}
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    window = 50

    for i, exp in enumerate(experiments):
        ax = axes[i]
        print(f"Porównanie zestawu: {exp['label']}...")
        # Wygładzanie wyników (średnia ruchoma) dla czytelności wykresu

        random.seed(42)
        np.random.seed(42)
        history_sarsa = train_sarsa(exp, episodes=1000)
        history_q = train_qlearning(exp, episodes=1000)

        smooth_sarsa = np.convolve(history_sarsa, np.ones(window) / window, mode='valid')
        smooth_q = np.convolve(history_q, np.ones(window) / window, mode='valid')

        ax.plot(smooth_sarsa, label='SARSA', color='blue', linestyle='-')
        ax.plot(smooth_q, label='Q-Learning', color='orange', linestyle='--')

        ax.set_title(f"{exp['label']}\n($gamma$={exp['gamma']}, $\\beta$={exp['beta']}, $epsilon$={exp['epsilon']})")
        ax.set_xlabel("Epizod")
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel("Średnia nagroda (Moving Avg)")
    plt.suptitle("Bezpośrednie prównanie: SARSA vs Q-Learning")
    plt.tight_layout()
    plt.show()


def main():
    print("=================Testing QLearning================")
    qlearning_experiments()
    print("\n=================Testing SARSA================")
    sarsa_experiments()
    print("\n=================Testing SARSA vs QLearning================")
    sarsa_vs_qlearning()


if __name__ == "__main__":
    main()
