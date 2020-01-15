import numpy as np
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import tensorflow as tf
import matplotlib.pyplot as plt


# Taken from https://stackoverflow.com/questions/47840527/using-tensorflow-huber-loss-in-keras
def huber_loss(y_true, y_pred, clip_delta=1.0):
    error = y_true - y_pred
    cond = tf.keras.backend.abs(error) < clip_delta

    squared_loss = 0.5 * tf.keras.backend.square(error)
    linear_loss = clip_delta * (tf.keras.backend.abs(error) - 0.5 * clip_delta)

    return tf.where(cond, squared_loss, linear_loss)

# To support parallel vector operations
# x = {reward, done, max_action, gamma}
def apply_next_state_value(x):
    y = x[0]
    if x[1] == 0:
        y += x[3] * x[2]
    return y

# Updating the target values to be fed into the ANN
# X is [State State ... State Action]
# Done so that batches can be updated in parallel
def update_model_batch(x):
    y = x[0:3]
    action = int(x[4])
    y[action] = x[3]
    return y


class DeepQLearner(object):
    def __init__(self, state_space, action_space, replay_memory_size, c_updates,
                 epsilon, epsilon_decay, min_epsilon, gamma, alpha, replay_batch_size):
        self.state_space = state_space
        self.action_space = action_space
        self.replay_memory = deque(maxlen=replay_memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.alpha = alpha
        self.replay_batch_size = replay_batch_size
        self.q_model = self.construct_neural_net()
        self.q_hat_model = self.construct_neural_net()
        # make sure the weights are the same
        self.q_hat_model.set_weights(self.q_model.get_weights())
        self.c_updates = c_updates
        self.counter = 0
        self.train_counter = 0

    def construct_neural_net(self):
        feature_size = self.state_space
        action_size = self.action_space
        alpha = self.alpha

        # Form ANN, input layer size of feature size
        # 2 hidden layers, 64 nodes each, dense
        # one output layer action size
        # these numbers and layers are all hyper-parameters - tune them to see what happens
        neural_net = Sequential()
        neural_net.add(Dense(32, activation='relu', input_dim=feature_size))
        neural_net.add(Dense(32, activation='relu'))
        neural_net.add(Dense(action_size, activation='linear'))
        # TODO this loss function should be our own - also use Huber loss like DeepMind did
        # TODO see https://stackoverflow.com/questions/47840527/using-tensorflow-huber-loss-in-keras
        neural_net.compile(loss='mse', optimizer=Adam(lr=alpha))

        return neural_net

    def determine_action(self, state_vector):
        # use epsilon greedy here
        # notice that we use the ACTUAL weights of the ANN to output an action
        if np.random.random() < self.epsilon:
            action = np.random.randint(low=0, high=self.action_space)
        else:
            q_state_action = self.q_model.predict(state_vector) # use Q here with real weights
            action = np.argmax(q_state_action)
            # print("State value: " + str(np.max(q_state_action)))

        if self.epsilon > self.min_epsilon:
            self.epsilon = self.epsilon * self.epsilon_decay

        return action

    def store_experience(self, state, action, state_prime, reward, done):
        self.replay_memory.append((state, action, state_prime, reward, done))

    def train(self):
        if len(self.replay_memory) > 10 * self.replay_batch_size:
            experience_batch = random.sample(self.replay_memory, self.replay_batch_size)
            # go through the whole batch and train on it
            state_batch = np.zeros((self.replay_batch_size, self.state_space), dtype=np.float64)
            state_prime_batch = np.zeros((self.replay_batch_size, self.state_space), dtype=np.float64)
            target_batch = np.zeros((self.replay_batch_size, 4), dtype=np.float64)
            action_batch = np.zeros((self.replay_batch_size, 1), dtype=np.float64)

            count = 0
            for state, action, state_prime, reward, done in experience_batch:
                state_batch[count, :] = state
                action_batch[count, 0] = action
                state_prime_batch[count, :] = state_prime
                target_batch[count, 0] = reward # add model prediction to this later if done flag is false
                target_batch[count, 1] = done
                count += 1

            next_state_values = self.q_hat_model.predict(state_prime_batch)
            maximizing_actions = np.apply_along_axis(lambda x: np.amax(x), 1, next_state_values)
            target_batch[:, 2] = maximizing_actions
            target_batch[:, 3] = self.gamma
            target_full_batch = np.apply_along_axis(apply_next_state_value, 1, target_batch)
            target_full_batch = np.reshape(target_full_batch, [self.replay_batch_size, 1])

            model_prediction_batch = self.q_model.predict(state_batch, verbose=0)
            y_batch_temp = np.concatenate((model_prediction_batch, target_full_batch, action_batch), axis=1)

            y_batch = np.apply_along_axis(update_model_batch, 1, y_batch_temp)
            self.q_model.fit(state_batch, y_batch, verbose=0, use_multiprocessing=True)

            self.counter += 1
            if self.counter > self.c_updates:
                # it is time to transfer the q model to the target model
                self.counter = 0
                self.transfer_q_to_q_hat()

    def transfer_q_to_q_hat(self):
        # to hold the TD difference steady
        self.q_hat_model.set_weights(self.q_model.get_weights())
        # print("Transferring weights from q to q hat")

    def anneal_learning_rate(self):
        self.alpha *= 0.6


def run_simple_environment():
    env = gym.make('Acrobot-v1')
    np.random.seed(12345)
    env.seed(12345)
    EPISODES = 5000

    # make our agent and randomly decide a lot of critical hyper-parameters lol
    dqlearner = DeepQLearner(state_space=env.observation_space.shape[0],
                             action_space=env.action_space.n,
                             replay_memory_size=32000, c_updates=32,
                             epsilon=1.0, epsilon_decay=0.999995, min_epsilon=0.01, gamma=0.985,
                             alpha=0.0001, replay_batch_size=32)

    # load weights if they exist
    try:
        dqlearner.q_model.load_weights("q_model_latest.h5")
        dqlearner.q_hat_model.load_weights("q_model_latest.h5")
    except:
        print("No weights to load")

    average_reward = deque(maxlen=100)
    graph_data = np.zeros((EPISODES, 3))
    for episode in range(0, EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, env.observation_space.shape[0]])
        total_reward = 0
        step = 0
        while True:
            # env.render()
            action = dqlearner.determine_action(state)
            state_prime, reward, done, info = env.step(action)
            state_prime = np.reshape(state_prime, [1, env.observation_space.shape[0]])
            dqlearner.store_experience(state, action, state_prime, reward, done)
            step += 1
            if step % 4 == 0:
                dqlearner.train()
            state = state_prime
            total_reward += reward
            if done: # TODO break early in this case for added stability
                average_reward.append(total_reward)
                print("Episode: " + str(episode) + "\tReward: " + str(total_reward) + "\tEpsilon: " + str(dqlearner.epsilon) +
                      "\t100 Episode Reward: " + str(sum(average_reward)/len(average_reward)))
                graph_data[episode, 0] = total_reward
                graph_data[episode, 1] = sum(average_reward)/len(average_reward)
                graph_data[episode, 2] = dqlearner.epsilon * 100
                dqlearner.q_model.save_weights("q_model_latest.h5", overwrite=True)
                break
    env.close()
    json_rep = dqlearner.q_model.to_json()
    with open("q_model.json" , "w") as json_file:
        json_file.write(json_rep)

    plt.figure()
    plt.plot(range(0, EPISODES), graph_data[:, 0], label="Episode Reward")
    plt.plot(range(0, EPISODES), graph_data[:, 1], label="Average Reward")
    plt.plot(range(0, EPISODES), graph_data[:, 2], label="Epsilon Percent")
    plt.ylabel("Reward")
    plt.xlabel("Episode")
    plt.title("Reward During Training for DQN Agent")
    plt.legend()
    plt.savefig('Training_Agent.png')
    plt.close()


def main():
    run_simple_environment()


if __name__ == "__main__":
    main()