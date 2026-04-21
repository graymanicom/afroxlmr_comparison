import numpy as np
import krippendorff

LABEL_MAP = {"VALID": 1, "INVALID": 0, "UNCERTAIN": 2}

def krippendorff_alpha_from_long(df, item_col="id", rater_col="rater",label_col="label"):
    items = df[item_col].unique().tolist()
    raters = df[rater_col].unique().tolist()
    mat = np.full((len(raters), len(items)), np.nan)
    item_index = {it:i for i,it in enumerate(items)}
    rater_index = {r:i for i,r in enumerate(raters)}
    
    for _, row in df.iterrows():
        i = rater_index[row[rater_col]]
        j = item_index[row[item_col]]
        mat[i, j] = LABEL_MAP[row[label_col]]

    return float(krippendorff.alpha(reliability_data=mat,level_of_measurement="nominal"))