import minigrid
import gymnasium as gym
import matplotlib.pyplot as plt
from qlearning_agent import QLearningAgent, train_generic
import numpy as np
from action_wrapper import ResrtrictedActionWrapper, minigrid_state_processor


def train(params, episodes):
    raw_env = gym.make("MiniGrid-FourRooms-v0", max_episode_steps=100, goal_pos=(2, 2), agent_pos=(7, 7))
    env = ResrtrictedActionWrapper(raw_env, actions_map=[0, 1, 2])

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
