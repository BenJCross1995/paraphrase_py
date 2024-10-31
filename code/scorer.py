import seaborn as sns
import matplotlib.pyplot as plt
from parascore import ParaScorer

class ParaphraseScorer:
    def __init__(self, score_type, model_type='bert-base-uncased'):
        self.score_type = score_type
        self.model_type = model_type
        self.paraScorer = None
        self.initialize_parascore()
    
    def initialize_parascore(self):
        """Initialize the ParaScorer model based on the score type"""
        if self.score_type in ['parascore', 'parascore_free']:
            self.paraScorer = ParaScorer(lang="en", model_type=self.model_type)
        else:
            raise ValueError(f"Score type '{self.score_type}' is not supported.")

    def add_parascore(self, df):
        """Calculate the standard ParaScore for the given DataFrame"""
        cands = df['rephrased'].tolist()
        refs = df['original'].tolist()
        P, R, F1 = self.paraScorer.score(cands, refs, batch_size=16)
        df['parascore'] = F1
        return df

    def add_parascore_free(self, df):
        """Calculate the reference-free version of ParaScore for the given DataFrame"""
        cands = df['rephrased'].tolist()
        refs = df['original'].tolist()
        P, R, F1 = self.paraScorer.free_score(cands, refs, batch_size=16)
        df['parascore_free'] = F1
        return df

    def calculate_score(self, df):
        """Calculate the selected score for the given DataFrame based on initialized score_type"""
        if self.score_type == 'parascore':
            return self.add_parascore(df)
        elif self.score_type == 'parascore_free':
            return self.add_parascore_free(df)
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