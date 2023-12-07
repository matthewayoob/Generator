import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import sys

def load_csv_to_df(file_path):
    """
    Load a CSV file into a DataFrame.

    Parameters:
    - file_path (str): Path to the CSV file.

    Returns:
    - pd.DataFrame: Loaded DataFrame.
    """
    df = pd.read_csv(file_path)
    return df

def k_means(df, num_clusters):
    """
    Perform k-means clustering on the DataFrame.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - num_clusters (int): Number of clusters.

    Returns:
    - pd.DataFrame: DataFrame with added 'cluster' column.
    - np.ndarray: Array of cluster centroids.
    """
    # Convert 'major' and 'minor' values to 0 and 1, respectively
    df.loc[df['mode'] == 'major', 'mode'] = 0
    df.loc[df['mode'] == 'minor', 'mode'] = 1

    # Select columns for vectors
    vector_columns = df.columns[4:]

    # Convert each row into a vector
    vectors = df[vector_columns].values

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(vectors)

    centroids = kmeans.cluster_centers_

    result_clusters = df[['track_id', 'cluster']]

    return result_clusters, centroids

def write_to_files(clusters, centroids, outfile_suffix, num_clusters):
    """
    Write clusters and centroids to files.

    Parameters:
    - clusters (pd.DataFrame): DataFrame containing 'track_id' and 'cluster' columns.
    - centroids (np.ndarray): Array of cluster centroids.
    - outfile_suffix (str): Suffix for output file names.
    - num_clusters (int): Number of clusters.
    """
    filepath = 'clusters/'
    clusters_filepath = f"{filepath}clusters_{outfile_suffix}_{num_clusters}"
    centroids_filepath = f"{filepath}centroids_{outfile_suffix}_{num_clusters}"

    # Save clusters
    clusters.to_csv(clusters_filepath, index=False)

    # Save centroids
    np.savetxt(centroids_filepath, centroids, fmt='%.18e', delimiter='\t')

def parse_centroids(clust_1_fp, clust_2_fp, centr_fp, clusters_fp):
    """
    Parse and combine centroid files.

    Parameters:
    - clust_1_fp (str): File path for the first cluster file.
    - clust_2_fp (str): File path for the second cluster file.
    - centr_fp (str): File path for the final centroid file.
    - clusters_fp (str): File path for the final cluster file.
    """
    # Read in both centroid files
    clust_1 = pd.read_csv(clust_1_fp)
    clust_2 = pd.read_csv(clust_2_fp)

    clust_2['cluster'] += 1000

    clusters = pd.concat([clust_1, clust_2])

    clusters.to_csv(clusters_fp, index=False)

def song_to_cluster(infile_path, outfile_path):
    """
    Convert track IDs to cluster IDs.

    Parameters:
    - infile_path (str): Input file path.
    - outfile_path (str): Output file path.
    """
    mapping = pd.read_csv('clusters/clusters_final')
    mapping.rename(columns={'track_id': 'track_id'}, inplace=True)
    mapping_dict = dict(zip(mapping['track_id'], mapping['cluster']))

    file_data = pd.read_csv(infile_path)

    file_data['track_id'] = file_data['track_id_clean'].map(mapping_dict)

    selected_columns = ['track_id', 'session_id', 'session_position', 'session_length', 'skip_1', 'skip_2', 'skip_3', 'not_skipped']
    final_file_data = file_data[selected_columns]

    final_file_data.to_csv(outfile_path, index=False)

def create_track_features():
    data_set = pd.read_csv('clusters/centroids_final', sep='\t', header=None)
    headers_data = pd.read_csv('data/mini/track_features/tf_mini.csv')

    data_set.columns = headers_data.columns[4:]

    # Add track IDs
    track_ids = [i for i in range(len(data_set))]
    data_set.insert(0, 'track_id', track_ids)

    data_set.to_csv('data/clustered/track_features/tf.csv', index=False)

def main():
    if sys.argv[1] == 'k_means':
        if len(sys.argv) < 4:
            infile_path = 'data/mini/track_features/tf_mini.csv'
            outfile_suffix = "mini"
            num_clusters = 3
        else:
            infile_path = sys.argv[2]
            outfile_suffix = sys.argv[3]
            num_clusters = int(sys.argv[4])

        print("Loading DataFrame")
        df = load_csv_to_df(infile_path)

        print("Performing k-means clustering")
        clusters, centroids = k_means(df, num_clusters)

        print("Writing to files")
        write_to_files(clusters, centroids, outfile_suffix, num_clusters)

    if sys.argv[1] == 'combine':
        clust_1_fp = 'clusters/clusters_00_1000'
        clust_2_fp = 'clusters/clusters_01_1000'
        centr_fp = 'clusters/centroids_final'
        clusters_fp = 'clusters/clusters_final'

        parse_centroids(clust_1_fp, clust_2_fp, centr_fp, clusters_fp)

    if sys.argv[1] == 'convert':
        infile_path = sys.argv[2]
        outfile_path = sys.argv[3]

        song_to_cluster(infile_path, outfile_path)

    if sys.argv[1] == 'generate_tf':
        create_track_features()

if __name__ == '__main__':
    main()
