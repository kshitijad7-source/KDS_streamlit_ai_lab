import streamlit as st
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt

# Load Iris dataset from scikit-learn
iris = datasets.load_iris()

# Create a pandas DataFrame for easier data manipulation
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target

# Title for the Streamlit app
st.title("Iris Dataset Exploratory Data Analysis")

# Display first few rows of the dataset
st.header("First Rows of the Dataset")
st.write(iris_df.head())

# Display summary statistics
st.header("Summary Statistics")
st.write(iris_df.describe())

# Sidebar for user interaction
st.sidebar.header("Filter and Select Columns")

# Map numeric target codes to species names for filtering
species_map = dict(enumerate(iris.target_names))
iris_df['species'] = iris_df['target'].map(species_map)

# Allow the user to filter by species
chosen_species = st.sidebar.multiselect(
    "Filter by species:",
    options=list(species_map.values()),
    default=list(species_map.values())
)

# Filtered dataframe based on species selection
filtered_df = iris_df[iris_df['species'].isin(chosen_species)]

# Allow the user to select numeric columns for plotting
numeric_columns = filtered_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
selected_columns = st.sidebar.multiselect("Choose numeric columns:", numeric_columns, default=numeric_columns[:2])

# Histogram
if selected_columns:
    st.header("Histogram")
    column_for_hist = selected_columns[0]
    fig, ax = plt.subplots()
    ax.hist(iris_df[column_for_hist], bins=20, color='skyblue', edgecolor='black')
    ax.set_title(f"Histogram of {column_for_hist}")
    ax.set_xlabel(column_for_hist)
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

    # Scatter plot if two columns selected
    if len(selected_columns) >= 2:
        st.header("Scatter Plot")
        x_col = selected_columns[0]
        y_col = selected_columns[1]
        fig2, ax2 = plt.subplots()
        ax2.scatter(iris_df[x_col], iris_df[y_col], c=iris_df['target'], cmap='viridis')
        ax2.set_title(f"Scatter Plot of {x_col} vs {y_col}")
        ax2.set_xlabel(x_col)
        ax2.set_ylabel(y_col)
        st.pyplot(fig2)
else:
    st.warning("Please select at least one numeric column for visualization.")
