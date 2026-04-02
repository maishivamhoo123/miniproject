def enforce_thermodynamics(preds: dict) -> dict:
    """Post-prediction check to ensure physical validity."""
    # Ensure AS < AF and MF < MS
    preds['AF'] = max(preds['AF'], preds['AS'] + 0.1)
    preds['MS'] = max(preds['MS'], preds['MF'] + 0.1)
    
    # Calculate derived properties AFTER prediction to prevent data leakage
    preds['TSPAN'] = preds['AF'] - preds['MF']
    
    # Round everything to 2 decimal places for clean UI presentation
    for key in preds:
        preds[key] = round(preds[key], 2)
        
    return preds