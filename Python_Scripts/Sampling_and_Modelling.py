#%%
import pandas as pd


#Initial Extracted Dataset
extracted_data = pd.read_csv("/Volumes/Main/untitled folder/Untitled/modified_extracted_Final_data.csv")

#Head for the Data
extracted_data.head()

# %%
#dropping some columns from the year 2013 to 2019 as it has only one class. 

# Drop columns from the extracted_data and assign the result to a Final_Data
Final_Data = extracted_data.drop(['id', 'ppt__count_below_mean__2013', 'ppt__count_below_mean__2014', 
                                  'ppt__count_below_mean__2015', 'ppt__count_below_mean__2016',
                                  'ppt__count_below_mean__2017', 'ppt__count_below_mean__2018',
                                  'ppt__count_below_mean__2019', 'ppt__skewness__2013',
                      'ppt__skewness__2014', 'ppt__skewness__2015', 'ppt__skewness__2016',
                      'ppt__skewness__2017', 'ppt__skewness__2018',
                      'ppt__skewness__2019', 'ppt__abs_energy__2013', 'ppt__abs_energy__2014',
                      'ppt__abs_energy__2015', 'ppt__abs_energy__2016', 'ppt__abs_energy__2017',
                      'ppt__abs_energy__2018', 'ppt__abs_energy__2019', 'ppt__variance__2013', 'ppt__variance__2014',
                      'ppt__variance__2015', 'ppt__variance__2016', 'ppt__variance__2017', 'ppt__variance__2018', 'ppt__variance__2019',
                      'ppt__mean_abs_change__2013', 'ppt__mean_abs_change__2014', 'ppt__mean_abs_change__2015', 'ppt__mean_abs_change__2016',
                      'ppt__mean_abs_change__2017', 'ppt__mean_abs_change__2018', 'ppt__mean_abs_change__2019', 'ppt__minimum__2013',
                      'ppt__minimum__2014', 'ppt__minimum__2015', 'ppt__minimum__2016', 'ppt__minimum__2017', 'ppt__minimum__2018', 'ppt__minimum__2019',
                      'ppt__sum_values__2013', 'ppt__sum_values__2014', 'ppt__sum_values__2015', 'ppt__sum_values__2016', 'ppt__sum_values__2017', 'ppt__sum_values__2018', 'ppt__sum_values__2019',
                      'ppt__median__2013', 'ppt__median__2014', 'ppt__median__2015', 'ppt__median__2016', 'ppt__median__2017', 'ppt__median__2018', 'ppt__median__2019',
                      'ppt__maximum__2013', 'ppt__maximum__2014', 'ppt__maximum__2015', 'ppt__maximum__2016', 'ppt__maximum__2017', 'ppt__maximum__2018', 'ppt__maximum__2019',
                      'ppt__mean__2013', 'ppt__mean__2014', 'ppt__mean__2015', 'ppt__mean__2016', 'ppt__mean__2017', 'ppt__mean__2018', 'ppt__mean__2019',
                      'ppt__standard_deviation__2013', 'ppt__standard_deviation__2014', 'ppt__standard_deviation__2015', 'ppt__standard_deviation__2016', 'ppt__standard_deviation__2017', 'ppt__standard_deviation__2018', 'ppt__standard_deviation__2019',
                      'ppt__count_above_mean__2013', 'ppt__count_above_mean__2014', 'ppt__count_above_mean__2015', 'ppt__count_above_mean__2016',
                      'ppt__count_above_mean__2017', 'ppt__count_above_mean__2018', 'ppt__count_above_mean__2019',
                      'rasterized_2015', 'rasterized_2016', 'rasterized_2017', 'rasterized_2018', 'rasterized_2019', 'rasterized_2024',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__median__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__median__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__median__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__median__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__median__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__median__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__median__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__variance__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__variance__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__variance__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__variance__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__variance__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__variance__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__variance__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__maximum__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__maximum__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__maximum__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__maximum__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__maximum__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__maximum__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__maximum__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__sum_values__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__sum_values__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__sum_values__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__sum_values__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__sum_values__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__sum_values__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__sum_values__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_below_mean__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_below_mean__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_below_mean__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_below_mean__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_below_mean__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_below_mean__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_below_mean__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean_abs_change__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean_abs_change__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean_abs_change__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean_abs_change__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean_abs_change__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean_abs_change__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__mean_abs_change__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__standard_deviation__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__standard_deviation__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__standard_deviation__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__standard_deviation__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__standard_deviation__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__standard_deviation__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__standard_deviation__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__abs_energy__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__abs_energy__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__abs_energy__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__abs_energy__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__abs_energy__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__abs_energy__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__abs_energy__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_above_mean__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_above_mean__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_above_mean__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_above_mean__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_above_mean__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_above_mean__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__count_above_mean__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__skewness__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__skewness__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__skewness__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__skewness__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__skewness__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__skewness__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__skewness__2019.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__minimum__2013.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__minimum__2014.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__minimum__2015.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__minimum__2016.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__minimum__2017.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__minimum__2018.tif',
                      '/home/ubuntu/WildFire_Capstone/Output/tmax/Time_Series_Features/tmax__minimum__2019.tif', ], axis=1)

# Save the new DataFrame to a new CSV file
Final_Data.to_csv('/Volumes/Main/untitled folder/Untitled/Final_Data_File.csv', index=False)

'''
    "abs_energy": [{}],
    "maximum": [{}],
    "mean": [{}],
    "minimum": [{}],
    "mean_abs_change": [{}],
    "standard_deviation": [{}],
    "sum_values": [{}],
    #"linear_trend": [{"attr": "slope"}],  # Optionally included
    "median": [{}],
    "count_above_mean": [{}],   
    "count_below_mean": [{}],
    "variance": [{}],
    "skewness": [{}]

'''
# %%
import pandas as pd
new_Data_frame = pd.read_csv("/Volumes/Main/untitled folder/Untitled/Final_Data_File.csv")

new_Data_frame.head()

#%%
#Dropping Null Values
# Dropping rows with any NaN values
 

# Assuming 'df' is your DataFrame
null_counts_per_column = new_Data_frame.isnull().sum()

print(null_counts_per_column)

total_null_counts = new_Data_frame.isnull().sum().sum()

print(f"Total number of null values in the DataFrame: {total_null_counts}")

#%%
cleaned_data = new_Data_frame.dropna()

#%%
cleaned_data.shape

#%%
#Random Forest for one year:




# %%
#Data Preparation
X = cleaned_data.drop(['rasterized_2020', 'rasterized_2021', 'rasterized_2022', 'rasterized_2023'], axis=1)  # Features
y = cleaned_data[['rasterized_2020', 'rasterized_2021', 'rasterized_2022', 'rasterized_2023']]  # Targets



# %%
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


#%%
# Assuming X and y are defined as before
targets = ['rasterized_2021', 'rasterized_2020', 'rasterized_2022', 'rasterized_2023']

# Dictionary to store each model trained on oversampled data for each target
models = {}

for target in targets:
    print(f"Processing target: {target}")
    
    # Splitting the data for the current target
    X_train, X_test, y_train, y_test = train_test_split(X, y[target], test_size=0.2, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    
    # Train a RandomForestClassifier on the oversampled data
    rf_classifier = RandomForestClassifier(random_state=42)
    rf_classifier.fit(X_train_oversampled, y_train_oversampled)
    
    # Store the trained model
    models[target] = rf_classifier
    
    # Optionally, evaluate the model on the test set
    y_pred = rf_classifier.predict(X_test)
    print(f"Classification report for {target}:\n{classification_report(y_test, y_pred)}\n")

# At this point, 'models' dictionary holds a trained RandomForestClassifier for each target




# %%
#Weighted class Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier


X = cleaned_data.drop(['rasterized_2020', 'rasterized_2021', 'rasterized_2022', 'rasterized_2023'], axis=1)  # Features
y = cleaned_data[['rasterized_2020', 'rasterized_2021', 'rasterized_2022', 'rasterized_2023']]  # Targets


# Train a RandomForestClassifier on the oversampled data with class_weight='balanced'
rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')

models = {}
targets = ['rasterized_2021', 'rasterized_2020', 'rasterized_2022', 'rasterized_2023']
class_weights = {0: 1, 1: 138}  
for target in targets:
    print(f"Processing target: {target}")
    
    # Splitting the data for the current target
    X_train, X_test, y_train, y_test = train_test_split(X, y[target], test_size=0.2, random_state=42)
    
    # Apply SMOTE
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    
    # Define class weights for RandomForestClassifier
    # This example uses 'balanced', but you can specify your own weights if needed
    rf_classifier = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Train the classifier on the oversampled data
    rf_classifier.fit(X_train_oversampled, y_train_oversampled)
    
    # Store the trained model
    models[target] = rf_classifier
    
    # Optionally, evaluate the model on the test set
    y_pred = rf_classifier.predict(X_test)
    print(f"Classification report for {target}:\n{classification_report(y_test, y_pred)}\n")

# ...

# %%
#PCA
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Example dataset
X = cleaned_data.drop(['rasterized_2020', 'rasterized_2021', 'rasterized_2022', 'rasterized_2023'], axis=1)
y = cleaned_data['rasterized_2020']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a pipeline with PCA and Random Forest
pipeline = Pipeline([
    ('pca', PCA(n_components=0.95)),  # Keep 95% of variance
    ('rf', RandomForestClassifier(random_state=42))
])

# Train the pipeline
pipeline.fit(X_train, y_train)

# Now you can use the pipeline for predictions and evaluating model performance
y_pred = pipeline.predict(X_test)

# Evaluate model performance...

# %%
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}\n")

# Generate and print a classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)

# Visualizing the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=['Not Fire', 'Fire'], yticklabels=['Not Fire', 'Fire'])
plt.title("Confusion Matrix")
plt.ylabel("Actual Values")
plt.xlabel("Predicted Values")
plt.show()

# %%
