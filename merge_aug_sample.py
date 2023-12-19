import pandas as pd
import numpy as np

combined_csv = pd.DataFrame()

# Iterate through the file numbers and concatenate them into the combined_csv DataFrame
for i in range(0,512):
    file_name = f'aug_sample/concept_{i}.csv'
    # Read the current file
    current_csv = pd.read_csv(file_name)
    # Concatenate the current file's data to the combined DataFrame
    combined_csv = pd.concat([combined_csv, current_csv], ignore_index=True)

# Save the combined data to a new CSV file
combined_csv.to_csv('aug_sample/concatenated_csv.csv', index=False)

combined_csv.head()  # Displaying the first few rows of the combined CSV for verification


images = []

for i in range(0,512):
    file_name = f'aug_sample/concept_{i}.npy'
    image_data = np.load(file_name)
    print(image_data.shape)
    images.append(image_data)
    
concatenated_images = np.concatenate(images, axis=0)
print(concatenated_images.shape)
np.save("aug_sample/concatenated_images.npy", concatenated_images)