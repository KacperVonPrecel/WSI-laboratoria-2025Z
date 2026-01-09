import minigrid
import gymnasium as gym
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent, train_generic
import numpy as np
from four_rooms_wrapper import FourRoomsWrapper, minigrid_state_processor
import random


def train(params, episodes, fixed_pos=True, fixed_agent=(6, 6), fixed_goal=(2, 2)):
    raw_env = gym.make("MiniGrid-FourRooms-v0", max_episode_steps=100)
    env = FourRoomsWrapper(raw_env, actions_map=[0, 1, 2], fixed_pos=fixed_pos, agent_start=fixed_agent, goal_pos=fixed_goal)

    agent = QLearningAgent(
        n_actions=env.action_space.n,
        learning_rate=params['beta'],
        discount=params['gamma'],
        epsilon=params['epsilon']
    )

    history = train_generic(
        env=env,
        agent=agent,
        state_processor_func=minigrid_state_processor,
        episodes=episodes
    )

    return history


def main():
    experiments = [
    {'beta': 0.1, 'gamma': 0.99, 'epsilon': 0.1, 'label': 'Standard'},
    {'beta': 0.5, 'gamma': 0.9, 'epsilon': 0.3, 'label': 'Agresywny (Wysokie Beta/Epsilon)'},
    {'beta': 0.05, 'gamma': 0.99, 'epsilon': 0.05, 'label': 'Ostrożny (Niskie Beta/Epsilon)'}
    ]

    plt.figure(figsize=(10, 6))

    random.seed("1234")

    for exp in experiments:
        print(f"Trenowanie: {exp['label']}...")
        # Wygładzanie wyników (średnia ruchoma) dla czytelności wykresu
        history = train(exp, episodes=1000)
        window = 50
        smoothed_history = np.convolve(history, np.ones(window) / window, mode='valid')
        plt.plot(smoothed_history, label=f"{exp['label']} ($gamma$={exp['gamma']}, $\\beta$={exp['beta']}, $epsilon$={exp['epsilon']})")

    plt.title("Porównanie zbieżności Q-learning w FourRooms")
    plt.xlabel("Epizod")
    plt.ylabel("Średnia nagroda (Moving Avg)")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
