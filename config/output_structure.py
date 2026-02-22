"""
Output Directory Structure Configuration

Defines the file organization for all pipeline outputs.
"""
from pathlib import Path

OUTPUT_DIR = Path('outputs')


class OutputStructure:
    """Manages the output directory tree."""

    def __init__(self, base_dir: Path = OUTPUT_DIR):
        self.base_dir = base_dir

        self.figures = {
            'base': base_dir / 'figures',
            'data_exploration': base_dir / 'figures' / '01_data_exploration',
            'model_training': base_dir / 'figures' / '02_model_training',
            'model_evaluation': base_dir / 'figures' / '03_model_evaluation',
            'uncertainty': base_dir / 'figures' / '04_uncertainty_quantification',
            'interpretability': base_dir / 'figures' / '05_interpretability',
            'process_optimization': base_dir / 'figures' / '06_process_optimization',
            'advanced_analysis': base_dir / 'figures' / '07_advanced_analysis',
            'supplementary': base_dir / 'figures' / '08_supplementary',
        }

        self.results = {
            'base': base_dir / 'results',
            'models': base_dir / 'results' / 'model_params',
            'features': base_dir / 'results' / 'feature_analysis',
            'prim': base_dir / 'results' / 'prim_analysis',
            'metrics': base_dir / 'results' / 'metrics',
        }

    def create_all_directories(self):
        """Create the full output directory tree."""
        for folder in self.figures.values():
            folder.mkdir(parents=True, exist_ok=True)
        for folder in self.results.values():
            folder.mkdir(parents=True, exist_ok=True)
        print("Output directory structure created.")

    def get_figure_path(self, category: str, filename: str) -> Path:
        """Get the full path for a figure file."""
        if category in self.figures:
            return self.figures[category] / filename
        return self.figures['base'] / filename

    def get_result_path(self, category: str, filename: str) -> Path:
        """Get the full path for a result file."""
        if category in self.results:
            return self.results[category] / filename
        return self.results['base'] / filename


# Global instance
output_structure = OutputStructure()
