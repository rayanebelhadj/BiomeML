"""
Feature loader module for IBD experiments.
Loads clinical and other features based on experiment configuration.
"""

import pandas as pd
from pathlib import Path


def load_features(experiment_config, output_dir):
    """
    Load additional features based on experiment configuration.
    
    Parameters
    ----------
    experiment_config : dict
        Experiment configuration dictionary
    output_dir : Path
        Output directory path containing the data
        
    Returns
    -------
    dict
        Dictionary containing:
        - use_clinical: bool - Whether clinical features are enabled
        - clinical_dim: int - Number of clinical features
        - clinical: pd.DataFrame or None - Clinical features dataframe
    """
    # Check if clinical features are requested
    use_clinical = False
    clinical_dim = 0
    clinical_df = None
    
    if experiment_config is None:
        return {
            'use_clinical': use_clinical,
            'clinical_dim': clinical_dim,
            'clinical': clinical_df
        }
    
    # Check model_training config for metadata features
    mt_config = experiment_config.get('model_training', {})
    features_config = mt_config.get('features', {})
    
    use_clinical = features_config.get('use_metadata', False)
    
    if use_clinical:
        # Try to load clinical features from metadata
        metadata_path = Path(output_dir) / "metadata" / "AGP_IBD_metadata.txt"
        
        if metadata_path.exists():
            try:
                clinical_df = pd.read_csv(metadata_path, sep='\t', index_col=0)
                
                # Select relevant clinical columns
                clinical_columns = []
                
                # Check which features to include
                include_age = features_config.get('include_age', True)
                include_sex = features_config.get('include_sex', True)
                include_bmi = features_config.get('include_bmi', True)
                include_antibiotics = features_config.get('include_antibiotics', False)
                
                if include_age and 'age_years' in clinical_df.columns:
                    clinical_columns.append('age_years')
                if include_sex and 'sex' in clinical_df.columns:
                    clinical_columns.append('sex')
                if include_bmi and 'bmi' in clinical_df.columns:
                    clinical_columns.append('bmi')
                if include_antibiotics and 'antibiotic_history' in clinical_df.columns:
                    clinical_columns.append('antibiotic_history')
                
                if clinical_columns:
                    clinical_df = clinical_df[clinical_columns]
                    clinical_dim = len(clinical_columns)
                else:
                    use_clinical = False
                    clinical_df = None
                    
            except Exception as e:
                print(f"⚠️ Error loading clinical features: {e}")
                use_clinical = False
                clinical_df = None
        else:
            print(f"⚠️ Metadata file not found: {metadata_path}")
            use_clinical = False
    
    return {
        'use_clinical': use_clinical,
        'clinical_dim': clinical_dim,
        'clinical': clinical_df
    }


