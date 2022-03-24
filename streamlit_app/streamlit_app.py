import streamlit as st
import pandas as pd
import plotly
import plotly.express as px
import numpy as np
from streamlit_lottie import st_lottie
import requests
from pathlib import Path
st.set_page_config(layout="centered")

#=============head=============#
st.write(Path(__file__).parents[0])
st.title('Clustering H&M Customers by Their Shopping Patterns 👕👖')
st.write('A Web App by [Minseok Song](https://github.com/msong-ds)')

st.subheader('Summary')

'''
This visualization is inspired by a kaggle competition hosted by H&M.

The purposes of clustering are two.
First, it is to understand the dataset better and gain insights for further analysis.
Also, cluster information can be used as a feature when building a model.

9 clusters were identified using K-means clustering.



Customer segmentation is important in understanding and discovering new opportunities to improve the business metrics.

The dataset was originally provided by H&M.(put link).
I aggregated the dataset to represent the customers' shopping pattern.
The features of the new dataset are the number of items that each customer bought in each category.
I identified 9 clusters using K-means clustering.
For more details about the process, please refer to the clustering notebook in the repo.

Now let's take a look at the clusters in detail.
'''

#=====================create graphs=====================
#=========== scatter plot
df_pca_samples = np.load("df_pca_samples.npy")
cluster_labels_samples = np.load("../streamlit_app/cluster_labels_samples.npy")
cluster_labels_samples = [str(i) for i in cluster_labels_samples]
all_clusters = np.unique(cluster_labels_samples)

scatter_3D = px.scatter_3d(df_pca_samples, x= df_pca_samples[:,0], y= df_pca_samples[:,1],z = df_pca_samples[:,2],
                  color=cluster_labels_samples, labels={'x':'PC 1', 'y':'PC 2','z':'PC 3'})
scatter_3D.update_traces(marker_size = 3)

#=========== histogram
cluster_labels = np.load("../streamlit_app/cluster_labels.npy")
counts = pd.Series(cluster_labels).value_counts().sort_values(ascending=False)
counts = counts.rename('Number of Customers')
bins = np.array(counts.index)
# bins, counts = np.unique(cluster_labels, return_counts=True)
color = px.colors.qualitative.Plotly[:9]
histogram = px.bar(x=bins, y=counts, labels={'x':'Cluster Label', 'y':'Number of Customers'}, color=color)
histogram.update_layout(xaxis = dict(tickmode = 'array',
                                     showticklabels =  True,
                                     type = 'category'
                                    ),
                        showlegend = False
                       )

#=========== bar chart - average spending
cluster_mean_spending = np.load("../streamlit_app/cluster_mean_spending.npy") /2 # per year
mean_spending = np.array([cluster_mean_spending[n] for n in bins])

# bins, same order of bins as previous histogram will be used for comparison
bar_mean_spending = px.bar(x=bins, y=mean_spending, labels={'x':'Cluster Label', 'y':'Average Amount Spent per Year'}, color = px.colors.qualitative.Plotly[:9])
bar_mean_spending.update_layout(xaxis = dict(tickmode = 'array',
                                showticklabels =  True,
                                type = 'category'
                              ),
                  showlegend=False
                 )

#=========== bar chart - average orders placed
cluster_mean_orders = np.load("../streamlit_app/cluster_mean_orders.npy")
mean_orders = np.array([cluster_mean_orders[n] for n in bins]) / 2 # orders per year
# same order of bins as previous histogram will be used for comparison
bar_mean_orders = px.bar(x=bins, y=mean_orders, labels={'x':'Cluster Label', 'y':'Average Number of Orders Placed per Year'},color = px.colors.qualitative.Plotly[:9])
bar_mean_orders.update_layout(xaxis = dict(tickmode = 'array',
                               showticklabels =  True,
                               type = 'category'
                               ),
                  showlegend=False
                 )

#=========== bar chart - average orders placed
amount_spent_per_order = mean_spending / mean_orders
# same order of bins as previous histogram will be used for comparison
bar_cart_size = px.bar(x=bins, y=amount_spent_per_order, labels={'x':'Cluster Label', 'y':'Average Basket Size'},color = px.colors.qualitative.Plotly[:9])
bar_cart_size.update_layout(xaxis = dict(tickmode = 'array',
                               showticklabels =  True,
                               type = 'category'
                               ),
                  showlegend=False
                 )


#=====================Body=====================
#=============3D scatter plot=============#
st.subheader('3D Scatter Plot with PCA')

'''
Each customer's transactions were aggregated into a dataset with 20 features. Each feature is the percentage of purchase in the feature category among the total purchase.

To visualize the clusters on 3D chart, the dimension was reduced into 3 using PCA.
Main clusters are formed around the vertices and in the middle of the pyramid.
The plot can be rotated and filtered.
'''
### plot
st.plotly_chart(scatter_3D)

#=============Histogram=============#
st.subheader('Number of Customers in Each Cluster')
st.write("More than 43.74% of the customers belong to the cluster 4. This shows that majority of customers' shopping pattern is similar.")
### plot
st.plotly_chart(histogram)

#=============Radar Graph=============#
st.subheader('The Characteristics of Each Cluster')
"""The clusters have very differnent shopping pattern. Most of their purchases are concentrated in one category. Though H&M covers very diverse product categories, many customers buy mainly from one category.
"""
option = st.selectbox(
     'Cluster Number',
     ([str(i) for i in bins]))
# interactive radar plot
centers = pd.read_csv("../streamlit_app/centers.csv")
cluster_order = pd.Series(cluster_labels).value_counts().iteritems()
radar_plot = px.line_polar(centers, r=str(option), theta='r', line_close=True,range_r =(0,0.9),title=f'The Purchase Composition of Category {option}')
radar_plot.update_traces(fill='toself')
st.plotly_chart(radar_plot)

description = {4: """This cluster accounts for 44% of the whole customers. Its purchases are mostly distributed among Divided, Womens Everyday Collection, and Lingeries/Tights.
                The cluster buys items from various categories, while other clusters buy mostly from one categories.""",
               3: """This cluster accounts for 15% of the whole customers. Its purchases are are mostly from Divided Category. Since the Divided category is targeted for the younger female customers,
               We can guess that the customers in this cluster are younger than other clusters""",
               2: """This cluster accounts for 13% of the whole customers. Its purchases are are mostly from Lingerie/Tights Category.""",
               5: """This cluster accounts for 10% of the whole customers. Its purchases are are mostly from Womans Everyday Collection Category.""",
               1: """This cluster accounts for 5% of the whole customers. Its purchases are are mostly from Womans Tailoring Category.""",
               0: """This cluster accounts for 4% of the whole customers. Its purchases are are mostly from Menswear Category. We may assume that most of the customer will be male.
               It is interesting that they also buy from Woman's Categories.""",
               7: """This cluster accounts for 4% of the whole customers. Its purchases are are mostly from Baby/Children Category.""",
               6: """This cluster accounts for 3% of the whole customers. Its purchases are are mostly from Sport Category.""",
               8: """This cluster accounts for 2% of the whole customers. Its purchases are are mostly from Ladies Accessories Category."""
              }
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.write("**Detailed Composition**")
    st.write((centers[[str(option)]].set_index(centers['r']).rename(columns={str(option):'%'}).sort_values(by='%', ascending=False)*100).applymap(lambda x: f"{np.round(x,2)}%"))
with row1_col2:
    st.write("**Description**")
    st.write(description[int(option)])

#=============Interactive Bar Chart=============#
st.subheader('Average Spending, Number of Orders, Basket Size of Each Cluster')
""" The cluster 4 place more orders and, thus, spend much more than other clusters. They are loyal customers who buy various things from H&M and frequently make purchases.
The cluster 1, who mainly buys from Woman Tailoring category, has the highest average basket size.

"""
option = st.selectbox(
     'Select a chart',
     ('Average Amount Spent Per Year by Cluster','Average Number of Orders Placed Per Year by Cluster','Average Basket Size by Cluster')
)
# interactive radar plot 
if option == 'Average Amount Spent Per Year by Cluster':
    st.write("The amount is smaller because the price was adjusted to hide the actual price by the data owner. \
         The cluster 4 spends more than 3 times than the other clusters on average. The rest of the clusters spend similar amount except cluster 7 and 8.")
    st.plotly_chart(bar_mean_spending)
elif option == 'Average Number of Orders Placed Per Year by Cluster':
    st.write("It shows the similar pattern as the previous chart. \
         The cluster 4 places 19 orders on average per year, which is more than 3 times as many orders as other clusters. The rest of the clusters place around 5 orders per year")
    st.plotly_chart(bar_mean_orders)
else:
    st.write("The cluster 1 has the highest average basket size on average, which is 20% higher than the cluster 4.")
    st.plotly_chart(bar_cart_size)

# st.subheader('Conclusion')