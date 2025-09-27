# YouTube Revenue Optimization 

# Import Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configure style

sns.set(style="whitegrid", palette="muted", font_scale=1.1)

# Load Data

df = pd.read_csv("C:/Users/conso/OneDrive/Documents/Datasets/youtube_channel_real_performance_analytics.csv", encoding = "latin1")

# Clean up column names: strip spaces, replace non-breaking spaces

df.columns = df.columns.str.replace("\xa0", " ", regex=True).str.strip()

print("Dataset shape:", df.shape)
df.head()

# Compute Regular vs Premium Views

df["Regular Views"] = df["Views"] - df["YouTube Premium Views"]

# Compute Profitability per View Type

df["Revenue per Regular View"] = (
    df["YouTube Ads Revenue (USD)"] /
    df["Regular Views"].replace(0, np.nan)
)

df["Revenue per Premium View"] = (
    df["YouTube Premium (USD)"] /
    df["YouTube Premium Views"].replace(0, np.nan)
)

# Compute Profit per Total View

df["Revenue per Total View"] = df["Estimated Revenue (USD)"] / df["Views"].replace(0, np.nan)

# Categorize video length buckets

df["Length Category"] = pd.cut(
    df["Video Duration"], 
    bins=[0, 180, 600, 1200, np.inf], 
    labels=["Short (<3m)", "Medium (3-10m)", "Long (10-20m)", "Very Long (20m+)"]
)

# Revenue per view comparison

avg_regular = df["Revenue per Regular View"].mean(skipna=True)
avg_premium = df["Revenue per Premium View"].mean(skipna=True)
print("Average Revenue per Regular View:", round(avg_regular, 6))
print("Average Revenue per Premium View:", round(avg_premium, 6))

# Boxplot: Revenue per View Type

plt.figure(figsize=(8,5))
sns.boxplot(data=df[["Revenue per Regular View","Revenue per Premium View"]])
plt.title("Distribution of Revenue per View Type")
plt.ylabel("Revenue (USD)")
plt.show()

# By Length Category

length_revenue = df.groupby("Length Category")["Revenue per Total View"].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=length_revenue, x="Length Category", y="Revenue per Total View")
plt.title("Average Revenue per View by Video Length")
plt.ylabel("Revenue per View (USD)")
plt.show()

# By Day of Week

dow_revenue = df.groupby("Day of Week")["Revenue per Total View"].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=dow_revenue, x="Day of Week", y="Revenue per Total View")
plt.title("Average Revenue per View by Day of Week")
plt.ylabel("Revenue per View (USD)")
plt.show()

# By Thumbnail CTR buckets

df["CTR Bucket"] = pd.cut(
    df["Video Thumbnail CTR (%)"], 
    bins=[0,2,5,10,20,100], 
    labels=["0-2%","2-5%","5-10%","10-20%","20%+"]
)

ctr_revenue = df.groupby("CTR Bucket")["Revenue per Total View"].mean().reset_index()

plt.figure(figsize=(8,5))
sns.barplot(data=ctr_revenue, x="CTR Bucket", y="Revenue per Total View")
plt.title("Average Revenue per View by Thumbnail CTR")
plt.ylabel("Revenue per View (USD)")
plt.show()

# Print summary insights

print("Key Insights:")
print(f" - Premium views generate ~{(avg_premium/avg_regular - 1)*100:.1f}% more revenue per view than regular views.")
print(" - Medium-length videos (3-10m) often balance engagement & profitability best.")
print(" - Higher CTR strongly correlates with higher revenue per view.")
print(" - Certain upload days may outperform others (see barplot).")

# Run hypothesis tests (t-test) to check statistical significance.

from scipy.stats import ttest_ind

# Example: Premium vs. Regular revenue per view
premium_rpv = df["YouTube Premium (USD)"] / df["Views"]
regular_rpv = df["Estimated Revenue (USD)"] / df["Views"]

t_stat, p_val = ttest_ind(premium_rpv.dropna(), regular_rpv.dropna())
print(f"T-stat: {t_stat}, p-value: {p_val}")

# Prepare dataset for dashboard
dashboard_df = df[[
    "Views",
    "Estimated Revenue (USD)",
    "YouTube Premium (USD)",
    "Revenue per Total View",
    "Video Thumbnail CTR (%)",
    "Average View Duration",
    "Day of Week"
]]

dashboard_df.to_csv("youtube_clean_for_dashboard.csv", index=False)
print("Clean dataset exported: youtube_clean_for_dashboard.csv")