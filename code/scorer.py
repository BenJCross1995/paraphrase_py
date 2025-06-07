import seaborn as sns
import matplotlib.pyplot as plt
import torch
from parascore import ParaScorer
from parascore.utils import diverse

class ParaphraseScorer:
    def __init__(self, score_type, model_type='bert-base-uncased', num_layers=None):
        self.score_type = score_type
        self.model_type = model_type
        self.num_layers = num_layers
        self.paraScorer = None
        self.initialize_parascore()
    
    def initialize_parascore(self):
        """Initialize the ParaScorer model based on the score type"""
        if self.score_type in ['parascore', 'parascore_free']:
            self.paraScorer = ParaScorer(lang="en", model_type=self.model_type, num_layers=self.num_layers)
        else:
            raise ValueError(f"Score type '{self.score_type}' is not supported.")

    def add_parascore(self, df):
        """Calculate the standard ParaScore for the given DataFrame"""
        cands = df['rephrased'].tolist()
        refs = df['original'].tolist()
        P, R, F1 = self.paraScorer.score(cands, refs, batch_size=16)
        df['parascore'] = F1
        return df

    def add_parascore_free(self, df, parascore_diversity_weight=0.05):
        """Calculate the reference-free version of ParaScore for the given DataFrame"""
        cands = df['rephrased'].tolist()
        refs = df['original'].tolist()

		# The free_score function weighs sim and diversity but also incorrect
        similarity = self.paraScorer.score(cands, refs, verbose=False, batch_size=64, return_hash=False)
        diversity = diverse(cands, refs)
		# Convert the diversity from a list to a tensor
        div  = torch.tensor(diversity, device=similarity[0].device)
		
        P, R, F1 = [sim + parascore_diversity_weight * div for sim in similarity]
        # P, R, F1 = self.paraScorer.free_score(cands, refs, batch_size=16)
		
        df['similarity_score'] = similarity[2].tolist()
        df['diversity_score'] = diversity
        df['diversity_weighting'] = parascore_diversity_weight
        df['parascore_free'] = F1
		
        return df

    def calculate_score(self, df, parascore_diversity_weight=0.05):
        """Calculate the selected score for the given DataFrame based on initialized score_type"""
        if self.score_type == 'parascore':
            return self.add_parascore(df)
        elif self.score_type == 'parascore_free':
            return self.add_parascore_free(df, parascore_diversity_weight)
        else:
            raise ValueError(f"Score type '{self.score_type}' is not supported.")
    
    def plot_density(self, df, grouping_column=None):
        """Plot the density of the score grouped by a specified column."""
        score_column = None
        if self.score_type == 'parascore':
            score_column = 'parascore'
        elif self.score_type == 'parascore_free':
            score_column = 'parascore_free'
        else:
            raise ValueError(f"Score type '{self.score_type}' is not supported for plotting.")
        
        # Ensure the score column exists
        if score_column not in df.columns:
            raise ValueError(f"The DataFrame must contain a '{score_column}' column.")
        
        # Plot the density plot using seaborn
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df, x=score_column, hue=grouping_column, fill=True)
        
        # Set plot title and labels
        title = f'Density Plot of {self.score_type.replace("_", " ").capitalize()}'
        if grouping_column:
            title += f' Grouped by {grouping_column}'
        plt.title(title)
        plt.xlabel(self.score_type.replace('_', ' ').capitalize())
        plt.ylabel('Density')
        
        # Show the plot
        plt.show()

# Example usage:
# Initialize with the score type 'parascore_free'
# tool = ParaphraseScorer(score_type='parascore_free', model_type='bert-base-uncased')

# Pass the DataFrame when calculating the score
# df_with_score = tool.calculate_score(df)

# Plot the density with grouping by 'chunk_id'
# tool.plot_density(df_with_score, grouping_column='chunk_id')

# Plot the density without any grouping
# tool.plot_density(df_with_score)