import pandas as pd

# Function to fuse the two datasets based on closest timestamps and matching labels
def fuse_datasets(cyber_df, physical_df):
    fused_data = []

    # Iterate over each row in the physical dataset
    for _, physical_row in physical_df.iterrows():
        physical_timestamp = physical_row['timestamp_p']
        physical_class = physical_row['class']

        # Find the closest timestamp in the cyber dataset
        closest_cyber_row = cyber_df.iloc[(cyber_df['timestamp_c'] - physical_timestamp).abs().argsort()[:1]]

        # Check if the labels match
        if closest_cyber_row['class'].values[0] == physical_class:
            # Concatenate rows (timestamp_c, timestamp_p, and all other columns)
            combined_row = pd.concat([
                pd.Series([closest_cyber_row['timestamp_c'].values[0], physical_timestamp], index=['timestamp_c', 'timestamp_p']),
                closest_cyber_row.drop(columns=['timestamp_c', 'class']).iloc[0],  # Drop timestamp_c and class from cyber
                physical_row.drop(labels=['timestamp_p', ])  # Drop timestamp_p and class from physical
            ])

            # Append the combined row to the list
            fused_data.append(combined_row)

    # Convert the list of fused rows into a DataFrame
    fused_df = pd.DataFrame(fused_data)
    return fused_df


if __name__ == '__main__':
    cyber_df = pd.read_csv('/data/cyber_ready.csv')
    physical_df = pd.read_csv('/data/physical_ready.csv')
    target_path = '/data/fuse.csv'

    # Apply the fusion function
    fused_df = fuse_datasets(cyber_df, physical_df)

    # translate to csv
    fused_df.to_csv(target_path, index=False)
