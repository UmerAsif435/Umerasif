import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

# Load dataset
df = pd.read_csv('C:\\Users\\DELL\\Desktop\\data analysis\\weather.csv')

# Clean column names (remove leading and trailing spaces)
df.columns = df.columns.str.strip()

# Function to plot a histogram for all numeric columns
def plot_combined_histogram():
    plt.figure(figsize=(10, 6))
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    all_data = df_numeric.values.flatten()
    plt.hist(all_data, bins=50, color='skyblue', edgecolor='black')
    plt.title('Combined Histogram of All Numeric Data')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()

# Function to plot a scatter graph for Min Dewpoint vs. Max Humidity
def plot_scatter():
    plt.figure(figsize=(8, 6))
    if 'Min_Dewpoint_F' in df.columns and 'Max_Humidity' in df.columns:
        df_sorted = df[['Min_Dewpoint_F', 'Max_Humidity']].dropna().sort_values('Max_Humidity')
        plt.scatter(df_sorted['Max_Humidity'], df_sorted['Min_Dewpoint_F'], color='green', marker='o')
        plt.title('Scatter Plot: Min Dewpoint vs. Max Humidity')
        plt.xlabel('Max Humidity')
        plt.ylabel('Min Dewpoint Temperature (°F)')
        plt.grid(True)
        plt.show()
    else:
        print("Required columns 'Min_Dewpoint_F' or 'Max_Humidity' do not exist in the dataset.")

# Function to plot a heatmap showing correlations across numeric columns
def plot_heatmap():
    plt.figure(figsize=(10, 8))
    df_numeric = df.select_dtypes(include=['float64', 'int64']).drop(columns=['Date', 'Events'], errors='ignore')
    correlation_matrix = df_numeric.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Heatmap Showing Correlations')
    plt.show()

# Function to perform K-Means clustering and visualize clusters
def plot_kmeans_clustering():
    try:
        required_features = ['Min_Dewpoint_F', 'Max_Humidity', 'Max_Sea_Level_Pressure_In']
        available_features = [feature for feature in required_features if feature in df.columns]
        features = df[available_features].dropna()

        if len(available_features) >= 2:
            kmeans = KMeans(n_clusters=3, random_state=42)
            kmeans.fit(features)
            features['Cluster'] = kmeans.labels_

            plt.figure(figsize=(8, 6))
            plt.scatter(features['Min_Dewpoint_F'], features['Max_Humidity'], c=features['Cluster'], cmap='viridis', marker='o')
            plt.title('K-Means Clustering')
            plt.xlabel('Min Dewpoint Temperature (°F)')
            plt.ylabel('Max Humidity')
            plt.colorbar(label='Cluster')
            plt.show()

        else:
            print(f"Insufficient data columns for clustering: {available_features}")

    except KeyError as e:
        print(f"KeyError in K-Means clustering: {e}")

# Function to perform Linear Regression and visualize the regression line
def plot_linear_regression():
    df_clean = df.dropna(subset=['Min_Dewpoint_F', 'Max_Humidity', 'Max_Sea_Level_Pressure_In'])
    
    try:
        X = df_clean[['Max_Humidity', 'Max_Sea_Level_Pressure_In']]
        y = df_clean['Min_Dewpoint_F']

        model = LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        plt.figure(figsize=(8, 6))
        plt.scatter(X['Max_Humidity'], y, color='blue', alpha=0.5, label='Actual Dewpoint')
        plt.plot(X['Max_Humidity'], predictions, color='red', linewidth=2, label='Regression Line')
        plt.title('Linear Regression: Min Dewpoint vs. Max Humidity')
        plt.xlabel('Max Humidity')
        plt.ylabel('Min Dewpoint Temperature')
        plt.legend(loc='best')
        plt.show()

    except KeyError as e:
        print(f"KeyError in Linear Regression Fitting: {e}")

# Function to create an Elbow Plot for K-Means Clustering
def plot_elbow():
    inertia_values = []
    df_clean_for_kmeans = df.select_dtypes(include=['float64', 'int64']).dropna()

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_clean_for_kmeans)
        inertia_values.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, 11), inertia_values, marker='o', linestyle='--')
    plt.title('Elbow Plot for K-Means')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.grid(True)
    plt.show()

# Call functions to display each plot
plot_combined_histogram()
plot_scatter()
plot_heatmap()
plot_kmeans_clustering()
plot_linear_regression()
plot_elbow()
