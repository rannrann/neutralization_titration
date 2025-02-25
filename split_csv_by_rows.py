import pandas as pd

def split_csv_by_rows(input_file, output_file_prefix, split_index, start_file_index):
    # Read the entire CSV into a DataFrame
    df = pd.read_csv(input_file)

    # Calculate the total number of splits
    num_splits = len(split_index)

    # Split the DataFrame and save each part
    for i in range(num_splits):
        start_row = split_index[i]
        if i != num_splits - 1:
            end_row = split_index[i + 1]
        else:
            end_row = len(df)

        # Slice the DataFrame
        chunk = df[start_row:end_row]

        # Save each chunk to a separate file
        output_file = f"{output_file_prefix}{start_file_index + i + 1}.csv"
        chunk.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

# Example usage
input_file = 'files/sample175-194.csv'
split_index = [125, 1071, 1878, 2629, 3341, 3632, 4119, 4719, 5893, 6652, 7088, 7814, 8494, 8866, 9162, 10313, 10816, 11279, 11765, 12592]
split_index = [i - 1 for i in split_index]
output_file_prefix = "files/sample"
start_file_index = 174
split_csv_by_rows(input_file, output_file_prefix, split_index, start_file_index)
 