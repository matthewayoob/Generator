import pandas as pd
import math
import numpy as np
import sys
import os

# Set up file paths
#base_data_path = FILEPATH
# data_log = FILEPATH2

class QLearningAgent:
    def __init__(self, num_states, num_actions, gamma, learning_rate):
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.Q_values = np.zeros((num_states, num_actions))

    def update_q_values(self, state, action, reward, next_state):
        max_next_state_value = np.max(self.Q_values[next_state, :])
        self.Q_values[state, action] += self.learning_rate * (reward + (self.gamma * max_next_state_value) - self.Q_values[state, action])

    def write_policy_to_file(self, filename):
        with open(filename, 'w') as file:
            best_actions = np.argmax(self.Q_values, axis=1)
            for action in best_actions:
                file.write(str(action + 1) + '\n')

def train_q_learning_agent(data_filepath, agent):
    data = pd.read_csv(data_filepath, delimiter=',')
    num_states = int(data['s'].max())
    num_actions = int(data['a'].max())
    agent.num_states = num_states
    agent.num_actions = num_actions

    for _, row in data.iterrows():
        state = int(row['s']) - 1
        action = int(row['a']) - 1
        reward = row['r']
        next_state = int(row['s_prime']) - 1

        agent.update_q_values(state, action, reward, next_state)

def generate_q_learning_submission(input_filepath, output_filepath):
    agent = QLearningAgent(num_states=0, num_actions=0, gamma=0.95, learning_rate=0.1)
    train_q_learning_agent(input_filepath, agent)
    agent.write_policy_to_file(output_filepath)

def generate_random_submission(input_logs):
    random_submissions = []

    for i, file_path in enumerate(input_logs):
        df = pd.read_csv(file_path)
        print('File {} read'.format(i))
        df = df.drop_duplicates(subset='session_id', keep='first', inplace=False)

        for _, row in df.iterrows():
            session_length = row['session_length']
            partial_length = math.ceil(session_length / 2)
            current_preds = np.random.choice([0, 1], size=(partial_length,))
            random_submissions.append(current_preds)

    return random_submissions

def save_submission_to_file(submission_data, output_path):
    with open(output_path, "w") as file:
        for line in submission_data:
            formatted_line = ''.join(map(str, line))
            file.write(formatted_line + '\n')
    print('Submission saved to {}'.format(output_path))

def main():
    if len(sys.argv) != 2:
        raise Exception("Error: Insufficient Arguments Entered.")

    random_submission_list = generate_random_submission(test_input_logs)
    save_submission_to_file(random_submission_list, os.path.join(submission_path, 'random_submission.txt'))

    input_file = os.path.join(base_data_path, 'test_set', sys.argv[1] + '.csv')
    output_file = os.path.join(base_data_path, 'test_set', sys.argv[1] + '.txt')
    generate_q_learning_submission(input_file, output_file)

if __name__ == '__main__':
    main()
