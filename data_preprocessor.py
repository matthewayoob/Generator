# Importing necessary libraries
import pandas as pd
import csv
import sys
import os

# Setting up file paths
#submission_filepath = #file name

# ID_COLUMN = 'track_id'
# CSV: state, action, reward, state_prime
# [[s1, s2, s3], s4, reward integer, [s2, s3, s4]]

def compute_reward(consecutive_set):
    # Example: Calculate reward based on your conditions
    # Extract values for 'id2' row
    skip_1_id2 = consecutive_set.iloc[1]['skip_1']
    skip_2_id2 = consecutive_set.iloc[1]['skip_2']
    skip_3_id2 = consecutive_set.iloc[1]['skip_3']
    not_skipped_id2 = consecutive_set.iloc[1]['not_skipped']

    # Extract values for 'id3' row
    skip_1_id3 = consecutive_set.iloc[2]['skip_1']
    skip_2_id3 = consecutive_set.iloc[2]['skip_2']
    skip_3_id3 = consecutive_set.iloc[2]['skip_3']
    not_skipped_id3 = consecutive_set.iloc[2]['not_skipped']

    # Example: Calculate reward based on conditions for 'id2' and 'id3'
    reward_id2 = 0
    if skip_1_id2:
        reward_id2 += 0
    if skip_2_id2:
        reward_id2 += 3
    if skip_3_id2:
        reward_id2 += 5
    if not not_skipped_id2:
        reward_id2 += 10

    reward_id3 = 0
    if skip_1_id3:
        reward_id3 += 0
    if skip_2_id3:
        reward_id3 += 3
    if skip_3_id3:
        reward_id3 += 5
    if not not_skipped_id3:
        reward_id3 += 10

    return reward_id2, reward_id3

def preprocess(input_filepath):
    # List to store processed data
    processed_data = []

    # Read CSV file into a DataFrame
    df = pd.read_csv(input_filepath)

    # Group by 'session_id'
    grouped_df = df.groupby('session_id')

    # Iterate over groups
    for session_id, group_data in grouped_df:
        # Extract consecutive sets of 3 songs
        for i in range(len(group_data) - 3):
            # Extract the current set of 4 songs
            current_set = group_data.iloc[i:i+4]

            # Example: Convert the consecutive set to lists and append to the processed_data
            state_id1 = current_set.iloc[0]['track_id']
            state_id2 = current_set.iloc[1]['track_id']
            state_id3 = current_set.iloc[2]['track_id']
            
            # Calculate reward based on conditions for 'id2' and 'id3'
            reward_id2, reward_id3 = compute_reward(current_set)

            # Append the processed data to the output list
            processed_data.append([state_id1, state_id2, reward_id2])
            processed_data.append([state_id2, state_id3, reward_id3])

    return processed_data

def write_to_csv_file(processed_data, output_filepath):
    # Write the data to the CSV file
    with open(output_filepath, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # Write the header line
        csvwriter.writerow(['state', 'action', 'reward', 'state_prime'])

        # Iterate over rows in the processed data
        for row in processed_data:
            # Append a fourth value that is the same as the second value
            row.append(row[1])

            formatted_row = []

            # Iterate over elements in each row
            for element in row:
                # Ensure each element is formatted appropriately
                if isinstance(element, list):
                    formatted_row.append("[" + ', '.join(map(str, element)) + "]")  # Convert list to string without enclosing in quotes
                elif isinstance(element, int):
                    formatted_row.append(str(element))  # Convert int to string
                else:
                    formatted_row.append(element)

            # Write the formatted row to the CSV file
            csvwriter.writerow(formatted_row)

def process_data_folder(input_folder, output_folder):
    # Check if the input folder exists
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        sys.exit(1)

    # Check if the output folder exists, create it if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process each CSV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            input_filepath = os.path.join(input_folder, filename)
            
            # Generate unique output file name based on the input file name
            output_filename = f"processed_{filename}"
            output_filepath = os.path.join(output_folder, output_filename)

            # Run the preprocessing function
            processed_data = preprocess(input_filepath)

            # Write the processed data to a CSV file
            write_to_csv_file(processed_data, output_filepath)

def main_program():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python script.py input_folder output_folder")
        sys.exit(1)

    # Get input and output folder paths from command-line arguments
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]

    # Process all CSV files in the input folder
    process_data_folder(input_folder, output_folder)

if __name__ == '__main__':
    # Run the main function when the script is executed
    main_program()
