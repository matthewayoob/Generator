import pandas as pd
import math
import numpy as np

# Define data paths
data_path = 'data/mini/'  # TODO: Point this to your data folder
submission_path = 'submissions/'
training_path = data_path + 'training_set/'
input_logs = [training_path + "/log_mini.csv"]
# input_logs = sorted(glob.glob(training_path + "log_mini*.csv")) # TODO: Point this to a subset of CSVs in your training set

def get_ground_truth(input_logs):
    """
    Extract ground truth data from the second half of sessions in input log files.

    Parameters:
    - input_logs (list): List of file paths containing log data in CSV format.

    Returns:
    - list: List of lists representing ground truth skip information (0 or 1) for session segments.

    Reads each CSV file in the input_logs list, extracts relevant columns for the second
    half of each session, and processes the data to create a list of ground truth skips.
    Each inner list corresponds to the skip information for a session segment.
    """
    ground_truths = []     
    for i, file_path in enumerate(input_logs):
        df = pd.read_csv(file_path)

        # Keep only the relevant columns of the second half of the session for saving the ground truth
        df = df[['session_id', 'skip_2', 'session_position', 'session_length']].loc[df['session_position'] * 2 > df['session_length']]
        df = df.reset_index()

        current_index = 0

        # Process each session, saving a list containing the ground truth skips
        while current_index < len(df):
            partial_length = df['session_length'].iloc[current_index] - df['session_position'].iloc[current_index] + 1
            session_skips = list(df.loc[current_index:current_index + partial_length - 1, 'skip_2'])
            ground_truths.append(session_skips)
            current_index += partial_length 

    return ground_truths

def generate_random_submission(input_logs):
    """
    Generate a random submission based on the lengths of sessions in input log files.

    Parameters:
    - input_logs (list): A list of file paths containing log data in CSV format.

    Returns:
    - list: A list of lists, where each inner list represents random binary predictions
            (0 or 1) for a portion of the sessions in the input log files.

    The function reads each CSV file in the input_logs list, drops duplicate sessions based on
    'session_id', and generates random binary predictions for each session. The length
    of the predictions is determined by half of the session length, rounded up if necessary.
    """
    output = []
    for i, file_path in enumerate(input_logs):
        df = pd.read_csv(file_path)
        print('File {} read'.format(i))

        # For this random submission, we just need to know the length of each session
        df = df.drop_duplicates(subset='session_id', keep='first', inplace=False)

        # For each session, generate a random prediction of the required length
        for j in df.index:
            session_length = df.loc[j, 'session_length']
            partial_length = math.ceil(session_length / 2)
            current_preds = np.random.choice([0, 1], size=(partial_length,))
            output.append(current_preds)

    return output

def evaluate(submission, ground_truth):
    """
    Evaluate the performance of a submission against ground truth data.

    Parameters:
    - submission (list): List of lists representing binary predictions (0 or 1).
    - ground_truth (list): List of lists representing ground truth skip information (0 or 1).

    Returns:
    - tuple: A tuple containing two metrics:
        - Average Precision (ap): A measure of the precision-recall trade-off.
        - First Prediction Accuracy (first_pred_acc): Proportion of correct first predictions.

    The function compares each prediction in the submission to the corresponding ground truth,
    calculates the Average Precision for each pair, and computes the overall Average Precision
    and First Prediction Accuracy for the entire evaluation set.
    """
    ap_sum = 0.0
    first_pred_acc_sum = 0.0
    counter = 0

    for sub, tru in zip(submission, ground_truth):
        if len(sub) != len(tru):
            raise Exception(f'Line {counter + 1} should contain {len(tru)} predictions, '
                            f'but instead contains {len(sub)}')

        ap_sum += average_precision(sub, tru, counter)
        first_pred_acc_sum += sub[0] == tru[0]
        counter += 1

    ap = ap_sum / counter
    first_pred_acc = first_pred_acc_sum / counter
    return ap, first_pred_acc

def average_precision(submission, ground_truth, counter):
    """
    Calculate Average Precision for a single submission against ground truth data.

    Parameters:
    - submission (list): List of binary predictions (0 or 1).
    - ground_truth (list): List of ground truth skip information (0 or 1).
    - counter (int): Line number used for error reporting.

    Returns:
    - float: Average Precision (AP) for the given submission and ground truth.

    The function calculates the Average Precision for a single pair of submission and ground truth.
    It checks for valid predictions (0 or 1) and raises an exception if an invalid prediction is found.
    The counter parameter is used for error reporting to indicate the line number in the input data.
    """
    s = 0.0
    t = 0.0
    c = 1.0

    for x, y in zip(submission, ground_truth):
        if x != 0 and x != 1:
            raise Exception(f'Invalid prediction in line {counter}, should be 0 or 1')

        if x == y:
            s += 1.0
            t += s / c

        c += 1

    return t / len(ground_truth)

def main():
    ground_truth = get_ground_truth(input_logs)
    submission = generate_random_submission(input_logs)

    ap, first_pred_acc = evaluate(submission, ground_truth)

    print(f'Average Precision: {ap}')
    print(f'First Prediction Accuracy: {first_pred_acc}')  

if __name__ == '__main__':
    main()
