import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, f_oneway
import warnings
warnings.filterwarnings("ignore")

# Load your dataset
df = pd.read_csv(r"C:\Users\Administrator\Desktop\insurance_eda\data\MachineLearningRating_v3.txt", delimiter="|") 

print(df.columns.tolist())
# Define claim occurred column
df['ClaimOccurred'] = df['TotalClaims'] > 0
df['ClaimOccurred'] = df['ClaimOccurred'].astype(int)

# Define key metrics
df['ClaimFrequency'] = df['ClaimOccurred']
df['ClaimSeverity'] = df['TotalClaims'] / df['ClaimOccurred'].replace(0, np.nan)
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

print("Metrics computed successfully.")


# Group data by Province
province_groups = df.groupby('Province')

# Collect lists of values for ANOVA
frequency_data = [group['ClaimFrequency'].dropna() for name, group in province_groups]
severity_data = [group['ClaimSeverity'].dropna() for name, group in province_groups]

# Run ANOVA for Claim Frequency and Claim Severity
from scipy.stats import f_oneway

freq_stat, freq_p = f_oneway(*frequency_data)
sev_stat, sev_p = f_oneway(*severity_data)

print(f"\n[Claim Frequency] ANOVA p-value across provinces: {freq_p:.5f}")
print(f"[Claim Severity]  ANOVA p-value across provinces: {sev_p:.5f}")

if freq_p < 0.05:
    print("✅ Reject H₀: Significant differences in Claim Frequency across provinces.")
else:
    print("❌ Fail to Reject H₀: No significant differences in Claim Frequency.")

if sev_p < 0.05:
    print("✅ Reject H₀: Significant differences in Claim Severity across provinces.")
else:
    print("❌ Fail to Reject H₀: No significant differences in Claim Severity.")



# Group data by PostalCode
zipcode_groups = df.groupby('PostalCode')

# Collect lists of values for ANOVA
zip_freq_data = [group['ClaimFrequency'].dropna() for name, group in zipcode_groups if len(group) > 10]
zip_sev_data = [group['ClaimSeverity'].dropna() for name, group in zipcode_groups if len(group) > 10]

# Run ANOVA
zip_freq_stat, zip_freq_p = f_oneway(*zip_freq_data)
zip_sev_stat, zip_sev_p = f_oneway(*zip_sev_data)

print(f"\n[Claim Frequency] ANOVA p-value across zip codes: {zip_freq_p:.5f}")
print(f"[Claim Severity]  ANOVA p-value across zip codes: {zip_sev_p:.5f}")

if zip_freq_p < 0.05:
    print("✅ Reject H₀: Significant differences in Claim Frequency between zip codes.")
else:
    print("❌ Fail to Reject H₀: No significant differences in Claim Frequency.")

if zip_sev_p < 0.05:
    print("✅ Reject H₀: Significant differences in Claim Severity between zip codes.")
else:
    print("❌ Fail to Reject H₀: No significant differences in Claim Severity.")

# Define metrics
df['ClaimFrequency'] = df['ClaimOccurred']  # Assumes binary: 1 = claim, 0 = no claim
df['ClaimSeverity'] = df['TotalClaims'] / (df['ClaimOccurred'].replace(0, np.nan))
df['Margin'] = df['TotalPremium'] - df['TotalClaims']

print("Data loaded and metrics computed.")



# Group by PostalCode
zip_margin_groups = df.groupby('PostalCode')

# Filter for groups with enough data and collect margin data
margin_data = [group['Margin'].dropna() for name, group in zip_margin_groups if len(group) > 10]

# Run ANOVA
from scipy.stats import f_oneway

margin_stat, margin_p = f_oneway(*margin_data)

print(f"\n[Margin] ANOVA p-value across zip codes: {margin_p:.5f}")

if margin_p < 0.05:
    print("✅ Reject H₀: Significant differences in Margin between zip codes.")
else:
    print("❌ Fail to Reject H₀: No significant differences in Margin.")


from scipy.stats import ttest_ind

# Filter Gender values
df_gender = df[df['Gender'].isin(['Male', 'Female'])]

# Group by Gender
male_data = df_gender[df_gender['Gender'] == 'Male']
female_data = df_gender[df_gender['Gender'] == 'Female']

# Frequency t-test
freq_t_stat, freq_p = ttest_ind(male_data['ClaimFrequency'], female_data['ClaimFrequency'], equal_var=False, nan_policy='omit')
# Severity t-test
sev_t_stat, sev_p = ttest_ind(male_data['ClaimSeverity'], female_data['ClaimSeverity'], equal_var=False, nan_policy='omit')

print(f"\n[Claim Frequency] Gender t-test p-value: {freq_p:.5f}")
print(f"[Claim Severity]  Gender t-test p-value: {sev_p:.5f}")

if freq_p < 0.05:
    print("✅ Reject H₀: Significant differences in Claim Frequency between Women and Men.")
else:
    print("❌ Fail to Reject H₀: No significant differences in Claim Frequency.")

if sev_p < 0.05:
    print("✅ Reject H₀: Significant differences in Claim Severity between Women and Men.")
else:
    print("❌ Fail to Reject H₀: No significant differences in Claim Severity.")
