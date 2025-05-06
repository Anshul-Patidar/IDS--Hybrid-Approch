#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd

# Set the folder path where all CSV files are stored
folder_path = "/Users/anshulpatidar/Downloads/MachineLearningCVE"

# Get a list of all CSV files
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Load and merge all CSV files
df = pd.concat([pd.read_csv(os.path.join(folder_path, file)) for file in csv_files], ignore_index=True)

# Display the dataset shape
print("Dataset Loaded. Shape:", df.shape)

# Display first 5 rows
df.head()


# In[7]:


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Path to the folder with all CSV files
folder_path = '/Users/anshulpatidar/Documents/CICdataset'

# Step 1: Load and concatenate all CSV files
dataframes = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"Loading {filename}")
        df = pd.read_csv(file_path, low_memory=False)
        dataframes.append(df)

# Combine all files into one DataFrame
df = pd.concat(dataframes, ignore_index=True)
print("Combined CSVs shape:", df.shape)

# Step 2: Replace infinite values and drop missing values
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

# Step 3: Drop irrelevant columns (adjust if necessary)
columns_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Step 4: Label Encoding for categorical features
categorical_cols = df.select_dtypes(include='object').columns.drop('Label', errors='ignore')

#categorical_cols = df.select_dtypes(include='object').columns.drop('Label')  # Exclude label for now
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 5: Binary label transformation (e.g., BENIGN = 0, all else = 1)
df['Label'] = df['Label'].apply(lambda x: 0 if x.strip().upper() == 'BENIGN' else 1)

# Step 6: Feature Normalization (excluding label)
features = df.drop('Label', axis=1)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Combine scaled features and label
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled['Label'] = df['Label'].values

# Save final cleaned dataset
df_scaled.to_csv('cleaned_cicids_dataset.csv', index=False)
print("Preprocessing complete. Saved to cleaned_cicids_dataset.csv")


# In[9]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Step 1: Load your dataset
# df = pd.read_csv('your_dataset.csv')

# Step 2: Drop unnecessary columns (adjust this list as needed)
columns_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Step 3: Handle missing values
df.dropna(inplace=True)

# Step 4: Identify the label column
possible_label_names = ['Label', 'label', 'Attack', 'Target', 'Class']
label_col = next((col for col in possible_label_names if col in df.columns), None)

# Step 5: Label Encoding for categorical features (excluding the label column)
categorical_cols = df.select_dtypes(include='object').columns
if label_col and label_col in categorical_cols:
    categorical_cols = categorical_cols.drop(label_col)

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 6: Encode the label column separately if it exists
if label_col:
    label_encoder = LabelEncoder()
    df[label_col] = label_encoder.fit_transform(df[label_col])
else:
    print("❗ Warning: Label column not found. Make sure it's present in your dataset.")

# Done
print("✅ Preprocessing complete. Here's a preview:")
print(df.head())


# In[15]:


print(df.columns.tolist())


# In[19]:


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

folder_path = '/Users/anshulpatidar/Documents/CICdataset'

# Step 1: Load and concatenate all CSV files
dataframes = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"📥 Loading: {filename}")
        df = pd.read_csv(file_path, low_memory=False)
        dataframes.append(df)

# Combine all files into one DataFrame
df = pd.concat(dataframes, ignore_index=True)
print("✅ Combined CSVs shape:", df.shape)

# Step 2: Clean column names (remove whitespace)
df.columns = df.columns.str.strip()

# Step 3: Replace infinite values and drop missing values
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

# Step 4: Drop irrelevant columns
columns_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Step 5: Detect label column
possible_label_names = ['Label', 'label', 'Attack', 'Target', 'Class']
label_col = next((col for col in possible_label_names if col in df.columns), None)
if not label_col:
    raise Exception("❌ No label column found in the dataset!")

# Step 6: Label encode other categorical columns
categorical_cols = df.select_dtypes(include='object').columns.drop(label_col, errors='ignore')
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 7: Binary classification conversion for label column
df[label_col] = df[label_col].apply(lambda x: 0 if str(x).strip().upper() == 'BENIGN' else 1)

# Step 8: Normalize features (excluding label)
features = df.drop(label_col, axis=1)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Step 9: Reconstruct final dataset
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled[label_col] = df[label_col].values

# Step 10: Save cleaned data
df_scaled.to_csv(os.path.join(folder_path, 'cleaned_cicids_dataset.csv'), index=False)

#df_scaled.to_csv('cleaned_cicids_dataset.csv', index=False)
print("✅ Preprocessing complete. Saved to cleaned_cicids_dataset.csv")


# In[21]:


import pandas as pd

df = pd.read_csv('cleaned_cicids_dataset.csv')
print(df.head())
print(df['Label'].value_counts())  # See how many benign vs attack



# In[23]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the cleaned dataset
df = pd.read_csv("cleaned_cicids_dataset.csv")

# --- Plot 1: Label Distribution ---
plt.figure(figsize=(6, 4))
sns.countplot(x='Label', data=df)
plt.title('Label Distribution (Benign vs Attack)')
plt.xticks([0, 1], ['Benign (0)', 'Attack (1)'])
plt.xlabel('Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# --- Plot 2: Feature Correlation Heatmap ---
plt.figure(figsize=(12, 10))
correlation = df.corr()
sns.heatmap(correlation, cmap='coolwarm', cbar=True)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# --- PCA Visualization (2 Components) ---
features = df.drop('Label', axis=1)
labels = df['Label']
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=pca_result[:, 0], y=pca_result[:, 1], hue=labels, palette='Set1', alpha=0.5)
plt.title('PCA: 2D Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend(title='Label', labels=['Benign', 'Attack'])
plt.tight_layout()
plt.show()

# --- t-SNE Visualization (2 Components, sampled for performance) ---
df_sample = df.sample(n=10000, random_state=42)
features_sample = df_sample.drop('Label', axis=1)
labels_sample = df_sample['Label']

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
tsne_result = tsne.fit_transform(features_sample)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=tsne_result[:, 0], y=tsne_result[:, 1], hue=labels_sample, palette='Set2', alpha=0.6)
plt.title('t-SNE: 2D Visualization (10K Sample)')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.legend(title='Label', labels=['Benign', 'Attack'])
plt.tight_layout()
plt.show()


# In[12]:


import os
import pandas as pd

folder_path = '/Users/anshulpatidar/Documents/CICdataset'

for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        df = pd.read_csv(file_path, low_memory=False)
        print(f"\n📂 File: {filename}")
        print("🔍 Column names:")
        print(df.columns.tolist())
        break  # Only show one file for now


# In[14]:


import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Define the folder path containing the raw CSV files
folder_path = '/Users/anshulpatidar/Documents/CICdataset'

# Step 1: Load and concatenate all CSV files
all_dataframes = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"📅 Loading: {filename}")
        df = pd.read_csv(file_path, low_memory=False)

        # ✅ Clean up column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Debug: Print columns of each CSV
        print(f"🔍 Columns in {filename}: {df.columns.tolist()}")

        # Normalize label column name if exists
        possible_labels = ['Label', 'label', 'Attack', 'Target', 'Class']
        for col in possible_labels:
            if col in df.columns:
                print(f"✅ Found label column '{col}' in {filename}")
                df.rename(columns={col: 'Label'}, inplace=True)
                break

        all_dataframes.append(df)

# Combine all loaded CSV files
df = pd.concat(all_dataframes, ignore_index=True)
print("✅ Combined CSVs shape:", df.shape)

# Step 2: Replace infinite values and drop missing values
df.replace([float('inf'), -float('inf')], pd.NA, inplace=True)
df.dropna(inplace=True)

# Step 3: Drop irrelevant columns
columns_to_drop = ['Flow ID', 'Source IP', 'Destination IP', 'Timestamp']
df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)

# Step 4: Encode all categorical features except 'Label'
categorical_cols = df.select_dtypes(include='object').columns.drop('Label', errors='ignore')
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Step 5: Multiclass label transformation (preserve all classes)
if 'Label' not in df.columns:
    raise KeyError("❗ No label column found even after column name cleanup.")

df['Label'] = df['Label'].astype(str).str.strip()  # Clean up whitespace

# Step 6: Normalize all features (excluding the label)
features = df.drop('Label', axis=1)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

# Step 7: Combine features and label
df_scaled = pd.DataFrame(features_scaled, columns=features.columns)
df_scaled['Label'] = df['Label'].values

# Step 8: Save preprocessed multiclass dataset in the same folder
output_file = os.path.join(folder_path, 'cleaned_cicids_multiclass.csv')
df_scaled.to_csv(output_file, index=False)
print(f"✅ Multiclass preprocessing complete. Saved to {output_file}")


# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns

# Step 9: Plot label distribution
plt.figure(figsize=(12, 6))
sns.countplot(data=df_scaled, x='Label', order=df_scaled['Label'].value_counts().index)
plt.xticks(rotation=45, ha='right')
plt.title('Label Distribution in Multiclass Dataset')
plt.xlabel('Attack Type')
plt.ylabel('Count')
plt.tight_layout()
plt.show()


# In[28]:


# Step 9: Display label distribution
print("\n📊 Label Distribution (Counts):")
print(df_scaled['Label'].value_counts())

# Optional: Plot label distribution
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 6))
sns.countplot(data=df_scaled, x='Label', order=df_scaled['Label'].value_counts().index)
plt.xticks(rotation=35, ha='right')
plt.title("Label Distribution (Multiclass)", fontsize=9)
plt.tight_layout()
plt.show()


# In[30]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the cleaned, preprocessed dataset
file_path = os.path.join(folder_path, 'cleaned_cicids_multiclass.csv')
df = pd.read_csv(file_path)

# Separate features and labels
X = df.drop('Label', axis=1).values
y = df['Label'].values

# Encode target labels into numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label classes for later use in reports
label_classes = label_encoder.classes_

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


# In[32]:


# Define the Autoencoder architecture
input_dim = X_train.shape[1]
encoding_dim = 32  # You can adjust this depending on desired compression

autoencoder = models.Sequential([
    layers.Input(shape=(input_dim,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(encoding_dim, activation='relu'),  # Encoder output
    layers.Dense(128, activation='relu'),
    layers.Dense(input_dim, activation='linear')  # Decoder output
])

autoencoder.compile(optimizer='adam', loss='mse')

# Train the Autoencoder
autoencoder.fit(X_train, X_train, 
                epochs=50, 
                batch_size=256, 
                shuffle=True, 
                validation_data=(X_test, X_test),
                verbose=1)

# Extract encoder model for feature transformation
encoder = models.Model(inputs=autoencoder.input, outputs=autoencoder.layers[2].output)

# Transform features
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)


# In[34]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Define input layer
input_dim = X_train.shape[1]
input_layer = Input(shape=(input_dim,))

# Encoder
encoded = Dense(128, activation='relu')(input_layer)
encoded = Dense(32, activation='relu')(encoded)  # Latent representation

# Decoder
decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

# Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train autoencoder
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(X_test, X_test),
                verbose=1)

# Extract encoder part
encoder = Model(inputs=input_layer, outputs=encoded)

# Encode train and test data
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)


# In[ ]:





# In[37]:


import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize and train XGBoost
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(label_classes),
    eval_metric='mlogloss',
    random_state=42
)


xgb_model.fit(X_train_encoded, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_encoded)

# Evaluation metrics
print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_classes))

print("🧾 Confusion Matrix:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)


# In[39]:


# Plot confusion matrix heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_classes, yticklabels=label_classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - XGBoost on Encoded Features")
plt.tight_layout()
plt.show()



# In[43]:


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Autoencoder Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[48]:


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Autoencoder Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[50]:


from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Generate classification report as dictionary
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# Filter only actual classes (ignore avg/total rows)
df_filtered = df_report.iloc[:-3][['precision', 'recall', 'f1-score']]

# Plotting
plt.figure(figsize=(12, 6))
df_filtered.plot(kind='bar', figsize=(14, 6), colormap='viridis')
plt.title('Classification Metrics per Class')
plt.xlabel('Attack Class')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()


# In[52]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# You can see which class got which number
label_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(label_mapping)


# In[54]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Compute and plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[56]:


from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
print(f"Overall Accuracy: {acc:.4f}")


# In[58]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Example: Replace with your true labels and predicted probabilities
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]         # Your test set true labels
y_scores = [0.1, 0.9, 0.8, 0.2, 0.95, 0.1, 0.88, 0.85, 0.05, 0.2]  # Predicted probabilities

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()


# In[ ]:




