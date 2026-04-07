import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("healthcare_dataset.csv")
df.head()
print("Shape:", df.shape)
print("Columns:", df.columns)

df.info()
df.describe()
df.isnull().sum()

# If missing values exist
df = df.dropna()
print("Duplicates:", df.duplicated().sum())

df = df.drop_duplicates()
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Gender'] = df['Gender'].str.strip().str.capitalize()
df['Medical Condition'] = df['Medical Condition'].str.strip()
df['Admission Type'] = df['Admission Type'].str.strip()
df['Length of Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days
plt.figure()
sns.boxplot(x=df['Billing Amount'])
plt.title("Before Outlier Removal")
plt.show()
q1 = df['Billing Amount'].quantile(0.25)
q3 = df['Billing Amount'].quantile(0.75)

iqr = q3 - q1

df = df[(df['Billing Amount'] >= q1 - 1.5*iqr) &
        (df['Billing Amount'] <= q3 + 1.5*iqr)]
plt.figure()
sns.boxplot(x=df['Billing Amount'])
plt.title("After Outlier Removal")
plt.show()
print(df['Medical Condition'].value_counts())
print(df['Gender'].value_counts())
df.groupby('Medical Condition')['Billing Amount'].mean()
df.groupby('Admission Type')['Billing Amount'].mean()
df.corr(numeric_only=True)
plt.figure()
sns.countplot(x='Medical Condition', data=df)
plt.xticks(rotation=45)
plt.title("Medical Condition Distribution")
plt.show()
plt.figure()
sns.histplot(df['Billing Amount'], bins=30)
plt.title("Billing Amount Distribution")
plt.show()
plt.figure()
sns.boxplot(x='Admission Type', y='Billing Amount', data=df)
plt.title("Admission Type vs Billing")
plt.show()
plt.figure()
sns.boxplot(x='Gender', y='Billing Amount', data=df)
plt.title("Gender vs Billing")
plt.show()
plt.figure()
sns.scatterplot(x='Length of Stay', y='Billing Amount', data=df)
plt.title("Length of Stay vs Billing")
plt.show()
plt.figure()
df['Test Results'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title("Test Results Distribution")
plt.ylabel("")
plt.show()
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.title("Correlation Heatmap")
plt.show()
print("Key Insights:")

print("1. Most common condition:")
print(df['Medical Condition'].value_counts().idxmax())

print("2. Highest cost condition:")
print(df.groupby('Medical Condition')['Billing Amount'].mean().idxmax())

print("3. Cost by Admission Type:")
print(df.groupby('Admission Type')['Billing Amount'].mean())

print("4. Avg Length of Stay:", df['Length of Stay'].mean())
df.to_csv("cleaned_healthcare_data.csv", index=False)
