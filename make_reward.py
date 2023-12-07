import pandas as pd
import sys

def calculate_reward_transitions(file_paths):
    """
    Calculate reward transitions based on input files.

    Parameters:
    - file_paths (list): List of file paths containing rows of (s, a, r, sp).

    Returns:
    - pd.DataFrame: Reward transition model with columns (s, a, reward).
    """
    total_counts = pd.DataFrame(columns=['s', 'a', 'skip1', 'skip2', 'skip3', 'not_skip'])

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        df = pd.read_csv(file_path)
        df = df.groupby(['s', 'a']).agg({'skip1': 'sum', 'skip2': 'sum', 'skip3': 'sum', 'not_skip': 'sum'}).reset_index()
        
        total_counts = pd.concat([total_counts, df])
        total_counts = total_counts.groupby(['s', 'a']).sum().reset_index()

    total_counts['max_skip'] = total_counts[['skip1', 'skip2', 'skip3', 'not_skip']].idxmax(axis=1)

    skip_mapping = {'skip1': 0, 'skip2': 3, 'skip3': 5, 'not_skip': 10}
    total_counts['reward'] = total_counts['max_skip'].map(skip_mapping)

    result_df = total_counts[['s', 'a', 'reward']]
    return result_df

def write_to_file(data_frame, outfile_path):
    """
    Write DataFrame to a CSV file.

    Parameters:
    - data_frame (pd.DataFrame): DataFrame to be written.
    - outfile_path (str): Output file path.
    """
    data_frame.to_csv(outfile_path, index=False)

def main():
    infile_path = "data/clustered/mini/test_set"
    outfile_path = "data/clustered/mini/test_set"
    n_files = 2

    if len(sys.argv) == 3:
        infile_path = sys.argv[1]
        outfile_path = sys.argv[2]
        n_files = 31

    outfile_path = f'{outfile_path}/final_rewards.csv'

    # Assuming files are named [test1.csv], [test2.csv], etc.
    files = [f'{infile_path}/test{i}.csv' for i in range(1, n_files + 1)]
    
    reward_df = calculate_reward_transitions(files)
    write_to_file(reward_df, outfile_path)

if __name__ == '__main__':
    main()
