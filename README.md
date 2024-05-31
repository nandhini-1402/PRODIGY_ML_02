## RFM Segmentation and Clustering

### Overview

This repository provides a Python implementation for RFM (Recency, Frequency, Monetary) segmentation and clustering. The goal is to analyze customer behavior based on their transaction history and to segment them into different groups using K-Means clustering. The implementation includes functions for calculating RFM metrics, creating clusters, and visualizing the results.

### Files

- `rfm_segmentation.py`: Contains the main functions for calculating RFM metrics, clustering, and visualization.
- `requirements.txt`: Lists the Python dependencies required to run the scripts.
- `example_notebook.ipynb`: Jupyter notebook demonstrating the usage of the functions with an example dataset.
- `README.md`: Documentation for the repository.

### Installation

To run the code, you need Python 3.x and the following packages:
- pandas
- numpy
- matplotlib
- scikit-learn

You can install the dependencies using the following command:
```sh
pip install -r requirements.txt
```

### Usage

1. **RFM Calculation**:
    The `RFM_dataframe` function calculates the Recency, Frequency, and Monetary (Revenue) values for each customer.
    ```python
    import pandas as pd
    from rfm_segmentation import RFM_dataframe
    
    df = pd.read_csv('your_dataset.csv')
    rfm_df = RFM_dataframe(customer_col='customer_id', purchase_date_col='purchase_date', revenue_col='revenue', df=df)
    ```

2. **Elbow Plot**:
    The `SSE_plot` function generates an elbow plot to determine the optimal number of clusters.
    ```python
    from rfm_segmentation import SSE_plot
    
    SSE_plot(rfm_df, col_to_plot='Frequency')
    SSE_plot(rfm_df, col_to_plot='Recency')
    SSE_plot(rfm_df, col_to_plot='logRevenue')
    ```

3. **Clustering**:
    The `frequency_cluster`, `recency_cluster`, and `revenue_cluster` functions perform K-Means clustering on the respective RFM dimensions.
    ```python
    from rfm_segmentation import frequency_cluster, recency_cluster, revenue_cluster
    
    rfm_df = frequency_cluster(5, 'Frequency', rfm_df)
    rfm_df = recency_cluster(4, 'Recency', rfm_df)
    rfm_df = revenue_cluster(5, 'logRevenue', rfm_df)
    ```

4. **Overall Scoring**:
    Combine the clusters to calculate an overall score for each customer.
    ```python
    rfm_df['OverallScore'] = rfm_df['RecencyCluster'] + rfm_df['FrequencyCluster'] + rfm_df['RevenueCluster']
    rfm_df.groupby('OverallScore')['Recency', 'Frequency', 'logRevenue'].mean()
    ```

5. **3D Visualization**:
    Visualize the clusters in a 3D plot.
    ```python
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    low = rfm_df.query('OverallScore == 0')
    mid = rfm_df.query('OverallScore == 1')
    high = rfm_df.query('OverallScore == 2')
    
    g1= (low['Frequency'].values, low['Recency'].values, low['logRevenue'].values)
    g2 = (mid['Frequency'].values, mid['Recency'].values, mid['logRevenue'].values)
    g3= (high['Frequency'].values, high['Recency'].values, high['logRevenue'].values)
    
    data = [g1, g2, g3]
    colors = ['#440154FF', '#20A387FF', '#FDE725FF']
    groups = ['Low', 'Med', 'High']
    
    for data, color, group in zip(data, colors, groups):
        x, y, z = data
        ax.scatter(x, y, z, alpha=0.5, c=color, label=group)
    
    ax.legend()
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Recency')
    ax.set_zlabel('Revenue')
    ax.set_title('Spatial Representation of Segments', loc='left')
    plt.show()
    ```

### Functions

1. **RFM_dataframe**:
    - **Description**: Calculates Recency, Frequency, and Monetary (Revenue) metrics for each customer.
    - **Parameters**:
        - `customer_col`: Column name for customer IDs.
        - `purchase_date_col`: Column name for purchase dates.
        - `revenue_col`: Column name for revenue.
        - `df`: Input dataframe.
    - **Returns**: Dataframe with Recency, Frequency, and Average Revenue for each customer.

2. **SSE_plot**:
    - **Description**: Generates an elbow plot for determining the optimal number of clusters.
    - **Parameters**:
        - `df`: Input dataframe.
        - `col_to_plot`: Column to be used for K-Means clustering.

3. **order_cluster**:
    - **Description**: Orders clusters from worst to best based on the mean value of the feature used for clustering.
    - **Parameters**:
        - `cluster_col`: Column name for cluster assignments.
        - `feature_col`: Column name for the feature used to assign clusters.
        - `df`: Input dataframe.
        - `ascending`: Boolean indicating the order of sorting.

4. **frequency_cluster**:
    - **Description**: Performs K-Means clustering on the Frequency column.
    - **Parameters**:
        - `cluster_number`: Number of clusters.
        - `frequency_col`: Column name for Frequency.
        - `dataframe`: Input dataframe.
    - **Returns**: Dataframe with frequency clusters.

5. **recency_cluster**:
    - **Description**: Performs K-Means clustering on the Recency column.
    - **Parameters**:
        - `cluster_number`: Number of clusters.
        - `recency_col`: Column name for Recency.
        - `dataframe`: Input dataframe.
    - **Returns**: Dataframe with recency clusters.

6. **revenue_cluster**:
    - **Description**: Performs K-Means clustering on the Revenue column.
    - **Parameters**:
        - `cluster_number`: Number of clusters.
        - `revenue_col`: Column name for Revenue.
        - `dataframe`: Input dataframe.
    - **Returns**: Dataframe with revenue clusters.

### Notes

- Ensure the input dataframe is cleaned and preprocessed before using these functions.
- The column names provided should match the ones in your dataframe.
- The elbow plot helps in determining the optimal number of clusters by identifying the "elbow point" where the SSE starts to level off.

