import torch
import numpy as np

class BanditModel:
    def __init__(self, bandits):
        self.bandits = bandits
        self.num_bandits = len(bandits)
        self.weights = torch.ones(self.num_bandits, requires_grad=True)
        self.total_reward = np.zeros(self.num_bandits)

    def pull_bandit(self, bandit):
        result = np.random.randn(1)
        if result > bandit:
            return 1
        else:
            return -1

    def train(self, learning_rate=0.5, epsilon=5, max_iterations=10000, max_iterations_without_change=10000):
        optimizer = torch.optim.SGD([self.weights], lr=learning_rate)
        episode = 0
        correct_choice = False
        iterations_without_change = 0
        while episode < max_iterations and not correct_choice and iterations_without_change < max_iterations_without_change:
            print(f"Начало итерации №{episode + 1}:")
            if np.random.rand() < epsilon:
                action = np.random.randint(self.num_bandits)
            else:
                action = self.choose_action(epsilon)
            print("Действие агента:", action + 1)
            reward = self.pull_bandit(self.bandits[action])
            print("Его награда:", reward)
            loss = self.compute_loss(action, reward)
            self.update_weights(optimizer, loss)
            self.total_reward[action] += reward
            if episode % 50 == 0:
                print(f"Эпизод: {episode}, Лучший ход: {action}, Оценка: {np.mean(self.total_reward)}")
            episode += 1
            if self.check_correct_choice():
                correct_choice = True
            else:
                iterations_without_change += 1
        if iterations_without_change >= max_iterations_without_change:
            print("Превышено максимальное количество итераций без изменения правильного выбора.")
        self.print_result()

    def choose_action(self, epsilon):
        if np.random.rand(1) < epsilon:
            return np.random.randint(self.num_bandits)
        else:
            return torch.argmax(self.weights).item()

    def compute_loss(self, action, reward):
        return -torch.log(self.weights[action]) * reward

    def update_weights(self, optimizer, loss):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def print_result(self):
        best_bandit = torch.argmax(self.weights).item() + 1
        print(f"Агент думает, что бандит №{best_bandit} идеален.")
        if best_bandit == np.argmax(-np.array(self.bandits)):
            print("И это совершенно верно!")
        else:
            print(f"Увы, но правильный ответ: {np.argmax(-np.array(self.bandits))}")

    def check_correct_choice(self):
        best_bandit = torch.argmax(self.weights).item() + 1
        return best_bandit == np.argmax(-np.array(self.bandits))

def main():
    bandits = [9, -3, 2, -5, 6, -1, 1, -2, 3]
    model = BanditModel(bandits)
    model.train(max_iterations=10000)

if __name__ == "__main__":
    main()
