
import pandas as pd
from pathlib import Path


def load_features(experiment_config, output_dir):
    use_clinical = False
    clinical_dim = 0
    clinical_df = None
    
    if experiment_config is None:
        return {
            'use_clinical': use_clinical,
            'clinical_dim': clinical_dim,
            'clinical': clinical_df
        }
    
    de_config = experiment_config.get('data_extraction', {})
    clinical_config = de_config.get('clinical_features', {})
    
    mt_config = experiment_config.get('model_training', {})
    arch_config = mt_config.get('architecture', {})
    
    use_clinical = clinical_config.get('enable', False) or arch_config.get('use_clinical_features', False)
    
    if use_clinical:
        disease = de_config.get('disease', 'IBD').upper()
        metadata_path = Path(output_dir) / "metadata" / f"AGP_{disease}_metadata.txt"
        
        if metadata_path.exists():
            try:
                clinical_df = pd.read_csv(metadata_path, sep='\t', index_col=0)
                
                clinical_columns = []
                
                features_config = clinical_config.get('features', {})
                include_age = features_config.get('use_age', True)
                include_sex = features_config.get('use_sex', True)
                include_bmi = features_config.get('use_bmi', True)
                include_antibiotics = features_config.get('use_antibiotics', False)
                
                if include_age:
                    for age_col in ['age_cat', 'age_years', 'age']:
                        if age_col in clinical_df.columns:
                            clinical_columns.append(age_col)
                            break
                if include_sex and 'sex' in clinical_df.columns:
                    clinical_columns.append('sex')
                if include_bmi:
                    for bmi_col in ['bmi_cat', 'bmi']:
                        if bmi_col in clinical_df.columns:
                            clinical_columns.append(bmi_col)
                            break
                if include_antibiotics:
                    for ab_col in ['antibiotic_history', 'antibiotics_past_year']:
                        if ab_col in clinical_df.columns:
                            clinical_columns.append(ab_col)
                            break
                
                if clinical_columns:
                    clinical_df = clinical_df[clinical_columns].copy()
                    age_mapping = {
                        '20s': 0, '30s': 1, '40s': 2, '50s': 3, '60s': 4, '70+': 5,
                        '20-29': 0, '30-39': 1, '40-49': 2, '50-59': 3, '60-69': 4, '70 or older': 5,
                        'child': -2, 'teen': -1,
                    }
                    for col in clinical_df.columns:
                        if 'age' in col.lower():
                            clinical_df[col] = clinical_df[col].map(age_mapping)
                            median_val = clinical_df[col].median()
                            clinical_df[col] = clinical_df[col].fillna(median_val if pd.notna(median_val) else 0)
                    
                    bmi_mapping = {
                        'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3,
                        'underweight': 0, 'normal': 1, 'overweight': 2, 'obese': 3,
                        'Under': 0, 'Normal': 1, 'Over': 2, 'Obese': 3,
                    }
                    for col in clinical_df.columns:
                        if 'bmi' in col.lower():
                            clinical_df[col] = clinical_df[col].map(bmi_mapping)
                            median_val = clinical_df[col].median()
                            clinical_df[col] = clinical_df[col].fillna(median_val if pd.notna(median_val) else 1)
                    
                    if 'sex' in clinical_df.columns:
                        sex_mapping = {'male': 0, 'female': 1, 'Male': 0, 'Female': 1, 'M': 0, 'F': 1}
                        clinical_df['sex'] = clinical_df['sex'].map(sex_mapping)
                        clinical_df['sex'] = clinical_df['sex'].fillna(0.5)
                    
                    for col in clinical_df.columns:
                        col_min = clinical_df[col].min()
                        col_max = clinical_df[col].max()
                        if col_max > col_min:
                            clinical_df[col] = (clinical_df[col] - col_min) / (col_max - col_min)
                        else:
                            clinical_df[col] = 0.5
                    
                    clinical_dim = len(clinical_columns)
                    print(f"Clinical features loaded: {clinical_columns}")
                else:
                    use_clinical = False
                    clinical_df = None
                    
            except Exception as e:
                raise RuntimeError(
                    f"Clinical features were enabled (use_clinical=True) but "
                    f"failed to load from {metadata_path}: {e}\n"
                    f"Set clinical_features.enable=false in config to skip, "
                    f"or fix the metadata file."
                ) from e
        else:
            raise FileNotFoundError(
                f"Clinical features were enabled (use_clinical=True) but "
                f"metadata file not found: {metadata_path}\n"
                f"Run 01_data_extraction first, or set "
                f"clinical_features.enable=false in config."
            )
    
    return {
        'use_clinical': use_clinical,
        'clinical_dim': clinical_dim,
        'clinical': clinical_df
    }


