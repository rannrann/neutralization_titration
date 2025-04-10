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
        output_file = f"{output_file_prefix}{start_file_index + i}.csv"
        chunk.to_csv(output_file, index=False)
        print(f"Saved: {output_file}")

# Example usage
input_file = 'files/sample214-233.csv'
split_index = [74, 995, 1706, 2137, 2978, 3792, 4304, 4875, 5893, 6535, 7340, 7949, 8462, 9137, 9866, 10632, 11161, 11821, 12362, 12975]
split_index = [i - 1 for i in split_index]
output_file_prefix = "files/sample"
start_file_index = 214
split_csv_by_rows(input_file, output_file_prefix, split_index, start_file_index)
 