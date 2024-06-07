import pandas as pd
from scripts.dna_encoding import encode_data, decode_data
from scripts.dna_visualization import plot_dna_helix, animate_dna
from scripts.training import train_model
from scripts.reinforcement_learning import train_agent
from scripts.performance_visualization import animate_performance


file_path = 'data/BestSeller_Books_of_Amazon.csv'
df = pd.read_csv(file_path)
output_dir = 'output'


encoded_dataset = df.apply(encode_data, axis=1).apply(pd.Series)


# Display and animate the DNA structure for each encoded book name
for index, row in encoded_dataset.iterrows():
    example_dna_sequence = row['Book Name']
    plot_dna_helix(example_dna_sequence, output_dir)
    animate_dna(example_dna_sequence, output_dir)

# Train the model
model_losses = train_model(encoded_dataset)
train_agent()

animate_performance(model_losses)