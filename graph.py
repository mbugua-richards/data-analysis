import pandas as pd

# Creating the DataFrame for the Face Recognition Model Results
data = {
    "Parameter": [
        "Intercept", 
        "CFMT", 
        "Spotlight Size", 
        "CFMT_Spotlight", 
        "Group Variance"
    ],
    "Coefficient (Coef.)": [
        0.409, 
        0.004, 
        0.004, 
        -0.000, 
        0.002
    ],
    "Standard Error (Std. Err.)": [
        0.067, 
        0.001, 
        0.001, 
        0.000, 
        0.010
    ],
    "z-value": [
        6.092, 
        4.154, 
        4.107, 
        -1.134, 
        "N/A"
    ],
    "p-value (P>|z|)": [
        0.000, 
        0.000, 
        0.000, 
        0.257, 
        "N/A"
    ],
    "95% Confidence Interval": [
        "[0.278, 0.541]", 
        "[0.002, 0.006]", 
        "[0.002, 0.006]", 
        "[-0.000, 0.000]", 
        "N/A"
    ]
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Adding model fit summary as a separate section
model_fit_summary = {
    "Model": ["Mixed Linear Model (MixedLM)"],
    "Dependent Variable": ["A-prime"],
    "No. Observations": [268],
    "Method": ["REML"],
    "No. Groups": [45],
    "Log-Likelihood": [223.7841],
    "Converged": ["Yes"],
    "Scale": [0.0076],
    "Min. Group Size": [5],
    "Max. Group Size": [6],
    "Mean Group Size": [6.0]
}

# Creating another DataFrame for the model fit summary
df_model_fit = pd.DataFrame(model_fit_summary)

# Saving both DataFrames to an Excel file
with pd.ExcelWriter('/mnt/data/Face_Recognition_Model_Results.xlsx') as writer:
    df.to_excel(writer, sheet_name='Results', index=False)
    df_model_fit.to_excel(writer, sheet_name='Model Fit Summary', index=False)

# Path to the saved file
"/mnt/data/Face_Recognition_Model_Results.xlsx"
