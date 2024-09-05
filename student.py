import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.title('Lab Activity #1')

# Sidebar for options
st.sidebar.title("Navigation")

# Sidebar options: Upload CSV, Dataset Overview, and Visualizations
option = st.sidebar.radio("Go to", ["Upload CSV", "Dataset Overview", "Visualizations"])

# Phase 1: CSV Upload in the Sidebar
if option == "Upload CSV":
    st.subheader("Upload Your CSV File")
    uploaded_file = st.file_uploader("Upload CSV file here", type="csv")

    if uploaded_file is not None:
        # Once the file is uploaded, save it in session state
        df = pd.read_csv(uploaded_file, delimiter=';')
        st.session_state['df'] = df
        st.session_state['file_uploaded'] = True
        st.success("CSV file uploaded successfully!")
    else:
        st.warning("Please upload a CSV file to proceed.")

# Phase 2: Dataset Overview (only available after CSV upload)
if option == "Dataset Overview":
    if 'file_uploaded' in st.session_state and st.session_state['file_uploaded']:
        st.subheader("Dataset Overview")

        # Task b: Display the first few rows of the dataset
        st.write("First few rows of the dataset:")
        st.write(st.session_state['df'].head())

        # Task c: Show dataset information (e.g., data types, missing values)
        st.write("Dataset information:")
        buffer = io.StringIO()  # Create an in-memory buffer
        st.session_state['df'].info(buf=buffer)  # Send info to buffer
        s = buffer.getvalue()  # Retrieve the buffer content as a string
        st.text(s)  # Display in Streamlit

        # Task d: Generate summary statistics for the dataset
        st.write("Summary statistics for the dataset:")
        st.write(st.session_state['df'].describe())
    else:
        st.warning("Please upload a CSV file in the 'Upload CSV' option to view the dataset.")

# Phase 3: Visualizations (only available after CSV upload)
if option == "Visualizations":
    if 'file_uploaded' in st.session_state and st.session_state['file_uploaded']:
        st.subheader("Visualizations")

        # Task e: Correlation Heatmap with Filterable Options (for numeric features only)
        st.subheader('Correlation Heatmap with Filterable Options')

        # Select only numeric columns from the dataframe
        numeric_columns = st.session_state['df'].select_dtypes(include=[np.number]).columns.tolist()

        # Create checkboxes (radio buttons) for each numeric column
        selected_columns = st.multiselect('Select Numeric Columns to Include in Heatmap:', numeric_columns, default=numeric_columns)

        # Filter the dataframe based on selected columns
        if selected_columns:
            filtered_df = st.session_state['df'][selected_columns]
            corr = filtered_df.corr()

            # Create the heatmap
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, annot_kws={"size": 8})
            st.pyplot(fig)
        else:
            st.info('Please select at least one numeric column to display the heatmap.')

        # Task f: Dynamic Boxplot for a Single Factor
        st.subheader('Boxplot of Selected Numeric Feature')

        # Dropdown to select a numeric column for the boxplot
        selected_feature = st.selectbox('Select a Numeric Feature for Boxplot', numeric_columns)

        # Create the boxplot for the selected feature
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(y=st.session_state['df'][selected_feature], ax=ax)
        ax.set_title(f'Boxplot of {selected_feature}')
        st.pyplot(fig)

        # Load the dataset (replace with your own CSV load logic)
        df = st.session_state['df']

        # Dropdown to select X-axis and Y-axis for bar chart
        st.subheader('Flexible Bar Chart: Select X and Y Axes')

        # For X-axis, we can use categorical columns (e.g., gender, school, study time, etc.)
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

        # For Y-axis, we use numeric columns (e.g., G1, G2, G3)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Dropdown for X-axis (categorical feature)
        x_axis = st.selectbox('Select X-Axis (Categorical)', options=categorical_columns)

        # Dropdown for Y-axis (numeric feature to analyze, like final grades)
        y_axis = st.selectbox('Select Y-Axis (Numeric)', options=numeric_columns)

        # Checkbox to filter by gender (optional)
        gender_filter = st.checkbox('Filter by Gender')
        if gender_filter:
            gender_choice = st.multiselect('Select Gender(s)', options=df['sex'].unique())
            df = df[df['sex'].isin(gender_choice)]

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=x_axis, y=y_axis, data=df, ax=ax, estimator=np.mean)

        # Set chart labels and title
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title(f'Bar Chart: {y_axis} vs {x_axis}')

        # Show the plot
        st.pyplot(fig)

        # Load the dataset (replace with your own CSV load logic)
        df = st.session_state['df']

        # Dropdowns for X-axis and Y-axis (numeric features)
        st.subheader('Scatter Plot: Select X and Y Axes')

        # Get numeric columns for X and Y axis selection
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Dropdown for selecting X-axis
        x_axis = st.selectbox('Select X-Axis', options=numeric_columns)

        # Dropdown for selecting Y-axis
        y_axis = st.selectbox('Select Y-Axis', options=numeric_columns)

        # Create the scatter plot based on selected X and Y axes
        fig = px.scatter(df, x=x_axis, y=y_axis, title=f'Scatter Plot: {x_axis} vs {y_axis}',
                        labels={x_axis: x_axis, y_axis: y_axis})

        # Display the scatter plot
        st.plotly_chart(fig)

    else:
        st.warning("Please upload a CSV file in the 'Upload CSV' option to view the visualizations.")
