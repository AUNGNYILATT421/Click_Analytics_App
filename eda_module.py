import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
from utils import new_line
from progress_bar import progress_bar

def count_values(n, singular, plural):
    return f"{n} value isn't {singular}" if n == 1 else f"{n} values aren't {plural}"

def examples(values, n=3):
    shown = ", ".join(f"`{v}`" for v in pd.unique(values)[:n])
    return f" (e.g. {shown})"

def convert_column(s, new_type):
    # Return (converted Series, None), or (None, reason) when some values can't be converted
    if new_type in ("Integer", "Float"):
        num = pd.to_numeric(s, errors='coerce')
        bad = num.isna() & s.notna()
        if bad.any():
            return None, f"{count_values(int(bad.sum()), 'a number', 'numbers')}{examples(s[bad])}"
        if new_type == "Float":
            return num.astype(float), None
        if num.isna().any():
            return None, "it has missing values, which Integer can't hold. Fill them first, or choose Float"
        if (num % 1 != 0).any():
            return None, f"it has decimal values{examples(num[num % 1 != 0])}, which would be cut off. Choose Float instead"
        return num.astype("int64"), None

    if new_type == "Text":
        # Keep missing values missing instead of turning them into the text "nan"
        return s.astype(object).where(s.isna(), s.astype(str)), None

    if new_type == "Boolean":
        if s.isna().any():
            return None, "it has missing values, which Boolean can't hold. Fill them first"
        mapping = {"true": True, "false": False, "yes": True, "no": False, "1": True, "0": False, "1.0": True, "0.0": False}
        mapped = s.astype(str).str.strip().str.lower().map(mapping)
        bad = mapped.isna()
        if bad.any():
            return None, f"{count_values(int(bad.sum()), 'True/False, Yes/No or 1/0', 'True/False, Yes/No or 1/0')}{examples(s[bad])}"
        return mapped.astype(bool), None

    if new_type == "Datetime":
        if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
            return None, "numbers can't be converted to dates directly"
        dates = pd.to_datetime(s, errors='coerce', format='mixed')
        bad = dates.isna() & s.notna()
        if bad.any():
            return None, f"{count_values(int(bad.sum()), 'a date', 'dates')}{examples(s[bad])}"
        return dates, None

def show_eda(df):
    # Assuming necessary imports are available like numpy, pandas, seaborn, matplotlib, etc.
    st.markdown("### 🕵️‍♂️ Exploratory Data Analysis", unsafe_allow_html=True)
    st.write("")  # To add a new line or space, if needed
    

    with st.expander("Show EDA"):
        new_line()

        # Head
        head = st.checkbox("Show First 5 Rows", value=False)    
        new_line()
        if head:
            st.dataframe(df.head(), use_container_width=True)

        # Tail
        tail = st.checkbox("Show Last 5 Rows", value=False)
        new_line()
        if tail:
            st.dataframe(df.tail(), use_container_width=True)

        # Shape
        shape = st.checkbox("Show Shape", value=False)
        new_line()
        if shape:
            st.write(f"This DataFrame has **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
            new_line()

        # Columns
        columns = st.checkbox("Show Columns", value=False)
        new_line()
        if columns:
            st.write(pd.DataFrame(df.columns, columns=['Columns']).T)
            new_line()

        if st.checkbox("Check Data Types", value=False):
            st.write(df.dtypes)
            new_line()

        new_line()  
        if st.checkbox("Show Skewness and Kurtosis", value=False):
            skew_kurt = pd.DataFrame(data={
                'Skewness': df.skew(numeric_only=True),
                'Kurtosis': df.kurtosis(numeric_only=True)
            })
            st.write(skew_kurt)
            new_line()

        new_line()  
        # Describe Numerical
        describe = st.checkbox("Show Description **(Numerical Features)**", value=False)
        new_line()
        if describe:
            st.dataframe(df.describe(), use_container_width=True)
            new_line()

        if st.checkbox("Unique Value Count", value=False):
            unique_counts = pd.DataFrame(df.nunique()).rename(columns={0: 'Unique Count'})
            st.write(unique_counts)
            new_line()

        new_line()  
        # Describe Categorical
        describe_cat = st.checkbox("Show Description **(Categorical Features)**", value=False)
        new_line()
        if describe_cat:
            if df.select_dtypes(include=[object, 'string']).columns.tolist():
                st.dataframe(df.describe(include=['object']), use_container_width=True)
                new_line()
            else:
                st.info("There is no Categorical Features.")
                new_line()

        # Correlation Matrix using heatmap seabron
        corr = st.checkbox("Show Correlation", value=False)
        new_line()
        if corr:
            numeric_df = df.select_dtypes(include='number')
            if numeric_df.corr().columns.tolist():
                fig, ax = plt.subplots()
                sns.heatmap(numeric_df.corr(), cmap='Blues', annot=True, ax=ax)
                st.pyplot(fig)
                new_line()
            else:
                st.info("There is no Numerical Features.")
            

        # Missing Values
        missing = st.checkbox("Show Missing Values", value=False)
        new_line()
        if missing:
            col1, col2 = st.columns([0.4,1])
            with col1:
                st.markdown("<h6 align='center'> Number of Null Values", unsafe_allow_html=True)
                st.dataframe(df.isnull().sum().sort_values(ascending=False),height=350, use_container_width=True)

            with col2:
                st.markdown("<h6 align='center'> Plot for the Null Values ", unsafe_allow_html=True)
                null_values = df.isnull().sum()
                null_values = null_values[null_values > 0]
                null_values = null_values.sort_values(ascending=False)
                null_values = null_values.to_frame()
                null_values.columns = ['Count']
                null_values.index.names = ['Feature']
                null_values['Feature'] = null_values.index
                fig = px.bar(null_values, x='Feature', y='Count', color='Count', height=350)
                st.plotly_chart(fig, use_container_width=True)

            new_line()
                 

        # Delete Columns
        delete = st.checkbox("Delete Columns", value=False)
        new_line()
        if delete:
            col_to_delete = st.multiselect("Select Columns to Delete", df.columns)
            new_line()
            
            col1, col2, col3 = st.columns([1,0.7,1])
            if col2.button("Delete", use_container_width=True):
                st.session_state.all_the_process += f"""
# Delete Columns
df.drop(columns={col_to_delete}, inplace=True)
\n """
                progress_bar()
                df.drop(columns=col_to_delete, inplace=True)
                st.session_state.df = df
                st.success(f"The Columns **`{col_to_delete}`** are Deleted Successfully!")


        # Remove Duplicate Rows (rows identical in every column)
        remove_dup = st.checkbox("Remove Duplicate Rows", value=False)
        new_line()
        if remove_dup:
            n_dup = int(df.duplicated().sum())
            if n_dup == 0:
                st.info("There are no duplicate rows.")
            else:
                st.write(f"This DataFrame has **{n_dup}** rows that are exact copies of an earlier row.")
                col1, col2, col3 = st.columns([1,0.7,1])
                if col2.button("Remove", use_container_width=True, key="remove_duplicates"):
                    st.session_state.all_the_process += f"""
# Remove Duplicate Rows
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
\n """
                    progress_bar()
                    df.drop_duplicates(inplace=True)
                    df.reset_index(drop=True, inplace=True)
                    st.session_state.df = df
                    st.success(f"**{n_dup}** duplicate rows have been removed.")


        # Change Data Type
        change_type = st.checkbox("Change Data Type", value=False)
        new_line()
        if change_type:
            col1, col2 = st.columns(2)
            with col1:
                cols_to_convert = st.multiselect("Select Columns to Convert", df.columns, key="convert_cols")
            with col2:
                new_type = st.selectbox("Convert To", ["Select", "Integer", "Float", "Text", "Boolean", "Datetime"], key="convert_type")

            if cols_to_convert and new_type != "Select":
                converted, problems = {}, []
                for col in cols_to_convert:
                    result, problem = convert_column(df[col], new_type)
                    if problem:
                        problems.append(f"- **`{col}`**: {problem}.")
                    else:
                        converted[col] = result

                # Don't convert anything until every selected column can be converted
                if problems:
                    st.warning(f"These columns can't be converted to **{new_type}**:\n\n" + "\n".join(problems))

                new_line()
                col1, col2, col3 = st.columns([1,0.7,1])
                if col2.button("Convert", use_container_width=True, key="convert_apply", disabled=bool(problems)):
                    st.session_state.all_the_process += f"""
# Change Data Type to {new_type}
# columns: {cols_to_convert}
\n """
                    progress_bar()
                    for col, result in converted.items():
                        df[col] = result
                    st.session_state.df = df
                    st.success(f"The Columns **`{cols_to_convert}`** have been converted to **{new_type}**.")


        # Show DataFrame Button
        col1, col2, col3 = st.columns([0.15,1,0.15])
        col2.divider()
        col1, col2, col3 = st.columns([1, 0.7, 1])
        if col2.button("Show DataFrame", use_container_width=True):
            st.dataframe(df, use_container_width=True)

        #start point

        # Histograms for Numerical Features
        hist = st.checkbox("Show Histograms", value=False)
        new_line()
        if hist:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.info("There is no Numerical Features.")
            else:
                col_for_hist = st.selectbox("Select Column for Histogram", options=numeric_cols)
                num_bins = st.slider("Select Number of Bins", min_value=10, max_value=100, value=30)
                fig, ax = plt.subplots()
                df[col_for_hist].hist(bins=num_bins, ax=ax, color='skyblue')
                ax.set_title(f'Histogram of {col_for_hist}')
                st.pyplot(fig)
                new_line()

        # Box Plots for Numerical Features
        boxplot = st.checkbox("Show Box Plots", value=False)
        new_line()
        if boxplot:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.info("There is no Numerical Features.")
            else:
                col_for_box = st.selectbox("Select Column for Box Plot", options=numeric_cols)
                fig, ax = plt.subplots()
                df.boxplot(column=[col_for_box], ax=ax)
                ax.set_title(f'Box Plot of {col_for_box}')
                st.pyplot(fig)
                new_line()


        # Scatter Plots for Numerical Features
        scatter = st.checkbox("Show Scatter Plots", value=False)
        new_line()
        if scatter:
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.info("There is no Numerical Features.")
            else:
                x_col = st.selectbox("Select X-axis Column", options=numeric_cols, index=0)
                y_col = st.selectbox("Select Y-axis Column", options=numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
                fig, ax = plt.subplots()
                df.plot(kind='scatter', x=x_col, y=y_col, ax=ax, color='red')
                ax.set_title(f'Scatter Plot between {x_col} and {y_col}')
                st.pyplot(fig)
                new_line()

        # Pair Plots for Numerical Features
        pairplot = st.checkbox("Show Pair Plots", value=False)
        new_line()
        if pairplot:
            numeric_df = df.select_dtypes(include=np.number)
            if numeric_df.empty:
                st.info("There is no Numerical Features.")
            else:
                pair_grid = sns.pairplot(numeric_df)
                st.pyplot(pair_grid.figure)
                plt.close(pair_grid.figure)

        # Count Plots for Categorical Data
        countplot = st.checkbox("Show Count Plots", value=False)
        new_line()
        if countplot:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols:
                st.info("There is no Categorical Features.")
            else:
                col_for_count = st.selectbox("Select Column for Count Plot", options=categorical_cols)
                fig, ax = plt.subplots()
                sns.countplot(x=df[col_for_count], data=df, ax=ax)
                ax.set_title(f'Count Plot of {col_for_count}')
                st.pyplot(fig)
                new_line()

        # Pie Charts for Categorical Data
        pie_chart = st.checkbox("Show Pie Charts", value=False)
        new_line()
        if pie_chart:
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols:
                st.info("There is no Categorical Features.")
            else:
                col_for_pie = st.selectbox("Select Column for Pie Chart", options=categorical_cols)
                pie_data = df[col_for_pie].value_counts()
                fig, ax = plt.subplots()
                ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                ax.set_title(f'Pie Chart of {col_for_pie}')
                st.pyplot(fig)
                new_line()

        new_line()
        if st.checkbox("Identify Outliers", value=False):
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if not numeric_cols:
                st.info("There is no Numerical Features.")
            else:
                col_for_outliers = st.selectbox("Select Column to Check Outliers", options=numeric_cols)
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col_for_outliers], ax=ax)
                ax.set_title(f'Outliers in {col_for_outliers}')
                st.pyplot(fig)
                new_line()

        new_line()
        if st.checkbox("Show Cross-tabulations", value=False):
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if not categorical_cols:
                st.info("There is no Categorical Features.")
            else:
                x_col = st.selectbox("Select X-axis Column for Cross-tab", options=categorical_cols, index=0)
                y_col = st.selectbox("Select Y-axis Column for Cross-tab", options=categorical_cols, index=1 if len(categorical_cols) > 1 else 0)
                cross_tab = pd.crosstab(df[x_col], df[y_col])
                st.write(cross_tab)
                new_line()

        new_line()
        if st.checkbox("Segmented Analysis", value=False):
            segments = st.selectbox("Select Segment", options=df.columns)
            segment_values = df[segments].dropna().unique()
            selected_segment = st.selectbox("Choose Segment Value", options=segment_values)
            segmented_data = df[df[segments] == selected_segment]
            st.write(segmented_data)
            new_line()

        new_line()
        if st.checkbox("Temporal Analysis", value=False):
            date_col_options = df.select_dtypes(include=[np.datetime64]).columns.tolist()
            value_col_options = df.select_dtypes(include=np.number).columns.tolist()
            
            if not date_col_options:
                st.error("No datetime columns found in the DataFrame.")
            elif not value_col_options:
                st.error("No numeric columns found in the DataFrame.")
            else:
                date_col = st.selectbox("Select Date Column", options=date_col_options)
                value_col = st.selectbox("Select Value Column", options=value_col_options)
                
                fig, ax = plt.subplots()
                df.set_index(date_col)[value_col].plot(ax=ax)
                ax.set_title(f'Trend Over Time - {value_col}')
                st.pyplot(fig)

        new_line()
        if st.checkbox("Show Word Cloud", value=False):
            # Get the list of object-type columns for user to choose from
            text_col_options = df.select_dtypes(include=[object, 'string']).columns.tolist()
            
            if text_col_options:
                # Let the user select a text column
                text_col = st.selectbox("Select Text Column for Word Cloud", options=text_col_options)
                
                # Collect text data, dropping NA values and joining them into a single string
                text_data = ' '.join(df[text_col].dropna()).strip()
                
                if text_data:  # Check if there is any text data to use
                    try:
                        wordcloud = WordCloud(width=800, height=400).generate(text_data)
                        fig, ax = plt.subplots()
                        ax.imshow(wordcloud.to_image(), interpolation='bilinear')
                        ax.axis('off')
                        st.pyplot(fig)
                    except ValueError as e:
                        st.error("Failed to generate word cloud: " + str(e))
                else:
                    st.error("No words available to create a word cloud. Please check the selected text data.")
            else:
                st.error("No suitable text columns found for creating a word cloud.")


        new_line()    
        # Interactive Data Tables
        interactive_table = st.checkbox("Show Interactive Data Table", value=False)
        new_line()
        if interactive_table:
            st.dataframe(df)
            new_line()

