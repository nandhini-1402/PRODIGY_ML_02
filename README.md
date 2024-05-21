def RFM_dataframe(customer_col, purchase_date_col, revenue_col, df):
   '''
    This function takes in a previously cleaned
    dataframe and the column names for customers, 
    purchase dates, and revenue as strings.
    Returns dataframe with two engineerd columns: 
    recency, and frequency, as well as an average
    revenue total for each unique
    visitor to the Google Merch Store.  
    '''
    # Select only unique customer IDs
    cust_df = df.groupby(customer_col)
    cust_df = pd.DataFrame(df[customer_col].unique(), columns=[customer_col])
    cust_df[customer_col].astype(str)
    
    # Create Dataframe of unique visitors and most recent visit to site
    df_recency = df.groupby(customer_col)[purchase_date_col].max().reset_index()
    df_recency.columns = [customer_col,purchase_date_col]
    df_recency[purchase_date_col] = pd.to_datetime(df_recency[purchase_date_col])
    df_recency['Recency'] = (df_recency[purchase_date_col].max() - df_recency[purchase_date_col]).dt.days
    df_recency = df_recency.drop(purchase_date_col, axis=1)
    df_recency['Recency'].astype('int')
    
    # Get visit counts for each user and create dataframe
    # Frequency is determined by repeat purchases: Order count - 1
    df_frequency = df.groupby(customer_col)[purchase_date_col].count().reset_index()
    
    df_frequency.columns = [customer_col,'Frequency']
    df_frequency['Frequency'] = df_frequency['Frequency'] - 1 
    
    # Get total order revenue for each unique visitor 
    df_revenue = df.groupby(customer_col)[revenue_col].mean().reset_index()
    
    # Merge data
    
    dfs = [df_frequency, df_recency, df_revenue]
    
    for e in dfs:
        cust_df = pd.merge(cust_df, e, on=customer_col)
        
    return cust_df

def SSE_plot(df, col_to_plot=False):
    '''
    This function takes in one column of
    the RFM dataframe as a string
    and will plot a KMeans elbow plot of the 
    sum of squared estimate. Where the elbow
    bends will determine how many clusters are
    optimal for the Kmeans clustering.
    '''
    sse={}
    if col_to_plot:
        km_var = df[[col_to_plot]].copy()
    else:
        km_var = df.copy()
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(km_var)
        km_var["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_ 
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
#     plt.title(str(km_var))
    plt.show();

for col in ['Frequency', 'Recency', 'logRevenue']:
    print(col)
    SSE_plot(rfm_df, col)

def order_cluster(cluster_col, feature_col, df, ascending):
    '''
    This function takes in a column of cluster
    assignments and features used to assign the 
    cluster as strings, dataframe in which clusters
    are featured, and ascending argument. Returns 
    dataframe with clusters ordered from worst to best.
    '''
    
    df_new = df.groupby(cluster_col)[feature_col].mean().reset_index()
    df_new = df_new.sort_values(by=feature_col,ascending=ascending).reset_index(drop=True)
    df_new['index'] = df_new.index
    df_final = pd.merge(df,df_new[[cluster_col,'index']], on=cluster_col)
    df_final = df_final.drop([cluster_col],axis=1)
    df_final = df_final.rename(columns={"index":cluster_col})
    return df_final
  
  def frequency_cluster(cluster_number, frequency_col, dataframe):
    frequency_kmeans = KMeans(n_clusters=cluster_number)
    frequency_kmeans.fit(dataframe[[frequency_col]])
    
    # Assigning cluster prediction to customers
    dataframe['FrequencyCluster'] = frequency_kmeans.predict(dataframe[[frequency_col]])
    
    # Ordering clusters from low to high and identifying statistics
    dataframe = order_cluster('FrequencyCluster', frequency_col, dataframe, True)
    
    return dataframe

rfm_df = frequency_cluster(5, 'Frequency', rfm_df)
rfm_df.groupby('FrequencyCluster')['Frequency'].describe()

def recency_cluster(cluster_number, recency_col, dataframe):
    recency_kmeans = KMeans(n_clusters=cluster_number)
    recency_kmeans.fit(dataframe[[recency_col]])
    
    # Assigning cluster prediction to customers
    dataframe['RecencyCluster'] = recency_kmeans.predict(dataframe[[recency_col]])
    
    # Ordering clusters from low to high and identifying statistics
    dataframe = order_cluster('RecencyCluster', recency_col, dataframe, False)
    
    return dataframe

rfm_df = recency_cluster(4, 'Recency', rfm_df)
rfm_df.groupby('RecencyCluster')['Recency'].describe()

def revenue_cluster(cluster_number, revenue_col, dataframe):    
    revenue_kmeans = KMeans(n_clusters=cluster_number)
    revenue_kmeans.fit(dataframe[[revenue_col]])
    
    # Assigning cluster prediction to customers
    dataframe['RevenueCluster'] = revenue_kmeans.predict(dataframe[[revenue_col]])
    
    # Ordering clusters from low to high and identifying statistics
    dataframe = order_cluster('RevenueCluster', revenue_col,dataframe,True)
    
    return dataframe

rfm_df = revenue_cluster(5, 'logRevenue', rfm_df)
rfm_df.groupby('RevenueCluster')['logRevenue'].describe()

rfm_df[‘OverallScore’] = rfm_df[‘RecencyCluster’] + rfm_df[‘FrequencyCluster’] + rfm_df[‘RevenueCluster’]
rfm_df.groupby(‘OverallScore’)[‘Recency’,’Frequency’,’logRevenue’].mean()

high = rfm_df.query('Segment == 2')
mid = rfm_df.query('Segment == 1')
low = rfm_df.query('Segment == 0')

from mpl_toolkits.mplot3d import Axes3D
%matplotlib notebook

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

g1= (low['Frequency'].values, low['Recency'].values, low['logRevenue'].values)
g2 = (mid['Frequency'].values, mid['Recency'].values, mid['logRevenue'].values)
g3= (high['Frequency'].values, high['Recency'].values, high['logRevenue'].values)

data = [g1, g2, g3]
colors = ['#440154FF', '#20A387FF', '#FDE725FF']
groups = ['Low', 'Med', 'High']

for data, color, group in zip(data, colors, groups):
    x, y, z = data
    ax.scatter(x, y, z, alpha=0.5, c=color, label=group)

# Make legend
    ax.legend()
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Recency')
    ax.set_zlabel('Revenue')
    ax.set_title('Spatial Representation of Segments', loc='left')
    plt.show();

    
