#!/usr/bin/env python3
"""
Script to simulate data changes for testing the MLOps workflow.
This adds a small amount of noise to the existing data to trigger the pipeline.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def simulate_data_change():
    """Simulate a data change by adding noise to existing data."""
    
    data_file = "data/diabetes.csv"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    print("🔄 Simulating data change...")
    
    # Load existing data
    df = pd.read_csv(data_file)
    print(f"Original data shape: {df.shape}")
    
    # Add small amount of noise to numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'encounter_id':  # Don't modify ID columns
            noise = np.random.normal(0, 0.01, len(df))
            df[col] = df[col] + noise
    
    # Add a few new rows with slight variations
    new_rows = df.sample(n=min(10, len(df)//10), random_state=42).copy()
    new_rows['encounter_id'] = range(df['encounter_id'].max() + 1, 
                                   df['encounter_id'].max() + 1 + len(new_rows))
    
    # Add the new rows
    df = pd.concat([df, new_rows], ignore_index=True)
    
    # Create backup of original
    backup_file = f"data/diabetes_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(backup_file, index=False)
    print(f"✅ Created backup: {backup_file}")
    
    # Save modified data
    df.to_csv(data_file, index=False)
    print(f"✅ Modified data saved: {df.shape}")
    
    # Create a change log
    change_log = f"""
# Data Change Log

## Change Details
- **Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **Type**: Simulated data change for testing
- **Original rows**: {len(df) - len(new_rows)}
- **New rows**: {len(new_rows)}
- **Total rows**: {len(df)}

## Changes Made
1. Added small noise to numeric columns
2. Added {len(new_rows)} new rows with variations
3. Created backup: {backup_file}

## Expected Workflow Trigger
This change should trigger the MLOps pipeline to:
1. Detect data changes
2. Validate data quality
3. Retrain the model
4. Build new Docker image
5. Deploy to staging

## Reverting Changes
To revert to original data:
```bash
cp {backup_file} {data_file}
```
"""
    
    with open("data/CHANGE_LOG.md", "w") as f:
        f.write(change_log)
    
    print("✅ Change log created: data/CHANGE_LOG.md")
    print("\n🚀 Ready to commit and push changes to trigger the workflow!")
    
    return True

if __name__ == "__main__":
    simulate_data_change()
