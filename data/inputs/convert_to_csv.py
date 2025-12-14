import pandas as pd
import os

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))



# Loop through each excel file
for filename in os.listdir(current_directory):
    if not filename.endswith('.xlsx'): continue
    try:
        # Read the excel file (first sheet by default)
        df = pd.read_excel(filename)

        # Construct the output csv filename
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        csv_filename = os.path.join(current_directory, f"{base_filename}.csv")

        # Write the dataframe to csv
        df.to_csv(csv_filename, index=False, encoding='utf-8')

        print(f"Successfully converted '{os.path.basename(filename)}' to '{os.path.basename(csv_filename)}'")
        

    except Exception as e:
        print(f"Error converting file '{os.path.basename(filename)}': {e}")

print("Conversion process finished.")