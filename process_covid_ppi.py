import pandas as pd
import numpy as np
import os

# Path to the interactions file
interactions_file = '/home/olarinoyem/Project/PPI_GNN1/my_new_dataset/raw/BIOGRID-PROJECT-covid19_coronavirus_project-4.4.227/BIOGRID-PROJECT-covid19_coronavirus_project-INTERACTIONS-4.4.227.tab3.txt'
output_dir = '/home/olarinoyem/Project/PPI_GNN1/my_new_dataset/processed'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Read the BioGRID interactions file
print("Reading interactions file...")
df = pd.read_csv(interactions_file, sep='\t')

# Display basic info about the dataset
print(f"Total interactions: {len(df)}")
print(df.columns.tolist())

# Extract the essential columns for PPI prediction using the correct column names
ppi_data = df[['Official Symbol Interactor A', 'Official Symbol Interactor B', 
               'Organism ID Interactor A', 'Organism ID Interactor B', 
               'Experimental System']]
print(f"Extracted columns. Sample data:")
print(ppi_data.head())

# Create positive examples (interactions)
positive_interactions = ppi_data[['Official Symbol Interactor A', 'Official Symbol Interactor B']].values
labels = np.ones(len(positive_interactions))

# Create a list of unique proteins
unique_proteins = set(ppi_data['Official Symbol Interactor A'].tolist() + 
                      ppi_data['Official Symbol Interactor B'].tolist())
print(f"Number of unique proteins: {len(unique_proteins)}")

# Save the processed data
interactions = np.column_stack((positive_interactions, labels))
np.save(os.path.join(output_dir, 'covid_ppi_interactions.npy'), interactions)

# Save protein list for sequence retrieval
with open(os.path.join(output_dir, 'covid_protein_list.txt'), 'w') as f:
    for protein in unique_proteins:
        f.write(f"{protein}\n")

print("Processing completed. Files saved to processed directory.")