import streamlit as st
import pandas as pd
import plotly
import plotly.express as px
import numpy as np
import requests
import io
from PIL import Image


#=====================Setting======================#
st.set_page_config(layout="centered")
def load_npy_web(url):
    """
    only for npy file on the web
    """
    response = requests.get(url)
    response.raise_for_status()
    return np.load(io.BytesIO(response.content))

path = "https://github.com/msong-ds/hnm_recommendation/raw/main/streamlit_app/"
cluster_to_name = {'0': 'Maybe Men',
                      '1': 'Businesswomen',
                      '2': 'Lingery lover',
                      '3': 'Young Souls',
                      '4': 'H&M loyals',
                      '5': 'Classic but chic',
                      '6': 'Sportsy in fashion',
                      '7': 'Maybe Parents',
                      '8': 'Accessory Addicts'
                     }
name_to_cluster = {v:k for k,v in cluster_to_name.items()}
#=============head=============#
#===========H&M logo
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    response_logo = requests.get(path + "hnm_logo.png")
    img = Image.open(io.BytesIO(response_logo.content))
    st.image(img)
with col3:
    st.write(' ')

#===========title
st.title('Clustering H&M Customers by Their Shopping Patterns 👗👖')
st.write('A Web App by [Minseok Song](https://github.com/msong-ds/hnm_recommendation)')

st.subheader('Summary')

'''
This visualization is inspired by a [kaggle competition hosted by H&M](https://www.kaggle.com/c/h-and-m-personalized-fashion-recommendations).

The purposes of clustering are three. First, it is to understand the dataset better and gain insights for further analysis.
Second, cluster information can be used as a feature when building a model. Finally, clustering can be used to identify customer segments. 
Customer segmentation is important to understand and discover new opportunities to improve business metrics.

Nine clusters were identified using K-means clustering.

The dataset was originally provided by H&M, from which new features were generated to represent customers' shopping patterns and the dataset was aggregated by unique customer ID.
The new features are the number of items that each customer bought in each category. For more details about the process, please refer to the clustering notebook in the repository.

Now let's take a look at the clusters in detail.
'''

#=====================create graphs=====================
#=========== scatter plot
df_pca_samples = load_npy_web(path + "df_pca_samples.npy")
cluster_labels_samples = load_npy_web(path + "cluster_labels_samples.npy")
cluster_labels_samples = [str(i) for i in cluster_labels_samples]
# all_clusters = np.unique(cluster_labels_samples)

scatter_3D = px.scatter_3d(df_pca_samples, x= df_pca_samples[:,0], y= df_pca_samples[:,1],z = df_pca_samples[:,2],
                  color=cluster_labels_samples, labels={'x':'PC 1', 'y':'PC 2','z':'PC 3'})
scatter_3D.update_traces(marker_size = 3)
scatter_3D.for_each_trace(lambda point: point.update(name = cluster_to_name[point.name]))

#=========== histogram
cluster_labels = load_npy_web(path + "cluster_labels.npy")
counts = pd.Series(cluster_labels).value_counts(sort=False).sort_index()
counts_orderd = counts.sort_values(ascending=False)
bins = np.array(counts_orderd.index)

## fix color scheme to reuse over the bar graphs
colors_bar_graphs = px.colors.qualitative.Plotly[:9]
color_map = {cluster:color for cluster,color in zip(bins,colors_bar_graphs)}
# bins, counts = np.unique(cluster_labels, return_counts=True)
histogram = px.bar(x=bins, y=counts_orderd, labels={'x':'Cluster Label', 'y':'Number of Customers'}, color = colors_bar_graphs)
histogram.update_layout(xaxis = dict(tickmode = 'array',
                                     showticklabels =  True,
                                     type = 'category',
                                     tickvals = bins,
                                     ticktext = [cluster_to_name[str(i)] for i in bins]
                                    ),
                        showlegend = False
                       )
    
#=========== bar chart - average spending
cluster_mean_spending = load_npy_web(path + "cluster_mean_spending.npy") /2 # per year
mean_spending = pd.Series(cluster_mean_spending).sort_values(ascending=False)
bins_spending = mean_spending.index

bar_mean_spending = px.bar(x=bins_spending, y=mean_spending, labels={'x':'Cluster Label', 'y':'Average Amount Spent per Year'},
                           color = [color_map[i] for i in bins_spending], color_discrete_map="identity")
bar_mean_spending.update_layout(xaxis = dict(tickmode = 'array',
                                showticklabels =  True,
                                type = 'category',
                                tickvals = bins_spending,
                                ticktext = [cluster_to_name[str(i)] for i in bins_spending]
                              ),
                  showlegend=False
                 )
#=========== bar chart - average orders placed
cluster_mean_orders = load_npy_web(path + "cluster_mean_orders.npy") /2
mean_orders = pd.Series(cluster_mean_orders).sort_values(ascending=False)
bins_orders = mean_orders.index

bar_mean_orders = px.bar(x=bins_orders, y=mean_orders, labels={'x':'Cluster Label', 'y':'Average Number of Orders Placed per Year'},
                         color = [color_map[i] for i in bins_orders], color_discrete_map="identity")
bar_mean_orders.update_layout(xaxis = dict(tickmode = 'array',
                               showticklabels =  True,
                               type = 'category',
                                tickvals = bins_orders,
                                ticktext = [cluster_to_name[str(i)] for i in bins_orders]
                               ),
                  showlegend=False
                 )

#=========== bar chart - average orders placed
amount_spent_per_order = mean_spending / mean_orders
amount_spent_ordered = pd.Series(amount_spent_per_order).sort_values(ascending=False)
bins_basket = amount_spent_ordered.index
# same order of bins as previous histogram will be used for comparison
bar_cart_size = px.bar(x=bins_basket, y=amount_spent_ordered, labels={'x':'Cluster Label', 'y':'Average Basket Size'},
                       color = [color_map[i] for i in bins_basket], color_discrete_map="identity")
bar_cart_size.update_layout(xaxis = dict(tickmode = 'array',
                               showticklabels =  True,
                               type = 'category',
                                tickvals = bins_basket,
                                ticktext = [cluster_to_name[str(i)] for i in bins_basket]
                               ),
                  showlegend=False
                 )


#=====================Body=====================
#=============3D scatter plot=============#
st.subheader('3D Scatter Plot with PCA')

'''
Each customer's transactions were aggregated into a dataset with 20 features. Each feature is the percentage of purchase in the respective category among the total purchase. For example, for the feature 'Women's every day collection' the values will represent the percentage of items a customer purchased in this category over the total number of items purchased. 

To visualize the clusters on a 3D chart, the dimensions were reduced from 20 to three using PCA. This graph shows that customer segments are distinctly clustered. In this three dimensions, the main clusters can be observed around the vertices and in the middle of the pyramid. For visualization and experimentation, the plot can be rotated and filtered. 
'''
### plot
st.plotly_chart(scatter_3D)

#=============Radar Graph=============#
st.subheader('The Characteristics of Each Cluster')
"""Each identified cluster has a very differnent shopping pattern. Though H&M covers very diverse product categories, many customers buy mainly from one category. The following dropdown menu allows for exploration of each cluster/customer segment. 
"""
option = st.selectbox(
     'Cluster Name',
     ([cluster_to_name[str(i)] for i in bins]))
option_num = name_to_cluster[option]
# interactive radar plot
centers = pd.read_csv(path + "centers.csv")

radar_plot = px.line_polar(centers, r=option_num, theta='r', line_close=True,range_r =(0,0.9),title=f'The Purchase Composition of Category {option}')
radar_plot.update_traces(fill='toself')
st.plotly_chart(radar_plot)

description = {'4': """This cluster accounts for 44% of the customer population. In this customer segment, purchases are mostly distributed among Divided, Womens Everyday Collection, and Lingeries/Tights.
                This is the only cluster that shows a distinct variety of purchases.""",
               '3': """This cluster accounts for 15% of the customer population. In this customer segment, purchases are mostly from the Divided Category. Since the Divided category is targeted to younger female customers, we can infer that the customers in this cluster are of young age""",
               '2': """This cluster accounts for 13% of the customer population. In this customer segment, purchases are mostly from the Lingerie/Tights Category.""",
               '5': """This cluster accounts for 10% of the customer population. In this customer segment, purchases are mostly from the Women's Everyday Collection Category.""",
               '1': """This cluster accounts for 5% of the customer population. In this customer segment, purchases are mostly from the Women's Tailoring Category.""",
               '0': """This cluster accounts for 4% of the customer population. In this customer segment, purchases are mostly from the Menswear Category. While we could infer that most customers in this segment will be male, it is interesting to note that they also buy from Women's Categories. Since gender is unavailable in the dataset, further information would be needed to confirm any hypothesis.""",
               '7': """This cluster accounts for 4% of the customer population. In this customer segment, purchases are mostly from the Baby/Children Category.""",
               '6': """This cluster accounts for 3% of the customer population. In this customer segment, purchases are mostly from the Sport Category.""",
               '8': """This cluster accounts for 2% of the customer population. In this customer segment, purchases are mostly from the Ladies Accessories Category."""
              }
row1_col1, row1_col2 = st.columns(2)
with row1_col1:
    st.write("**Detailed Composition**")
    st.write((centers[[option_num]].set_index(centers['r']).rename(columns={option_num:'%'}).sort_values(by='%', ascending=False)*100).applymap(lambda x: f"{np.round(x,1)}%"))
with row1_col2:
    st.write("**Description**")
    st.write(description[option_num])
    
#=============Histogram=============#
st.subheader('Number of Customers in Each Cluster')
st.write("More than 43.7% of customers belong to cluster number 4. This shows that the distribution is skewed around consumption in a combination of the Divided, Women's Everday Collection and Lingeries/Tights categories. In fact, if we take the next three clusters, which consume in these categories but in isolation, we can see that around 80% of customers are mostly purchasing in these three categories.")
### plot
st.plotly_chart(histogram)


#=============Interactive Bar Chart=============#
st.subheader('Interactive Bar Chart: Spending, Number of Orders and Basket Size')
"""Cluster number 4 is a loyal customer segment; they place more orders, buy from various categories and spend much more yearly than other segments. However, they do not have the highest average basket size. Customers in cluster 1, who mainly buy from the Woman Tailoring category, have the highest average basket size. Unfortunately, the data set does not include price information in absolute values, but rather adjusted to hide actual values. 

"""
option_bar = st.selectbox(
     'Select a chart',
     ('Average Amount Spent Per Year by Cluster','Average Number of Orders Placed Per Year by Cluster','Average Basket Size by Cluster')
)
# interactive radar plot 
if option_bar == 'Average Amount Spent Per Year by Cluster':
    st.write("Customers in cluster 4 spend per year more than 3 times than other customers on average. The smallest average spend comes from customers in clusters 7 and 8, who buy mostly from the Baby Children and Women's Accesories categories, respectively.")
    st.plotly_chart(bar_mean_spending)
elif option_bar == 'Average Number of Orders Placed Per Year by Cluster':
    st.write("Customers in cluster 4 place 19 orders on average per year, which is more than 3 times as many orders as for other customers, who place around 5 orders per year. It is interesting to note that customers in cluster with a predominant category, still place several orders per year.")
    st.plotly_chart(bar_mean_orders)
elif option_bar == 'Average Basket Size by Cluster':
    st.write("Customers in cluster 1 have the highest average basket size, which is 20% higher than in the next cluster. This segment predominantly buys from the Woman Tailoring category, which explains the higher average size of basket. THe smallest basket size is in cluster 7, which buys predominantly from the Baby Children Category.")
    st.plotly_chart(bar_cart_size)
else:
    st.write("Please Select a Chart")

#=============Summary of the Clusters =============#
st.subheader('Summary')
summary_table = pd.DataFrame(cluster_to_name.values()).join(pd.DataFrame(np.vstack([(counts/np.sum(counts)).apply(lambda x: np.round(x*100,1)}"),
                                                                                    cluster_mean_spending,
                                                                                    cluster_mean_orders,
                                                                                    amount_spent_per_order
                                                                                   ]).T)
                                                            ,lsuffix="_")
summary_table.columns = ['Cluster','% of customers', 'Mean Spending','Mean Order No.','Basket Size']
summary_table = summary_table.sort_values(by='% of customers',ascending=False)

st.write(summary_table.style.highlight_max(subset=['% of customers', 'Mean Spending','Mean Order No.','Basket Size']))
        
#=============Conclusion =============#
st.subheader('Conclusion')
"""
             This study shows a clear segmentation of H&M's customers in nine clusters. Based on the results presented above and summarized in the table below, we make three suggestions:
             - Since most customers mostly purchase from a single category, further cross-selling efforts should be considered to increase customers' share of wallet and make H&M their one stop shop for clothing and accessories. Even the most infrequent H&M's customers purchase five times on average per year. Thus, there is great opportunity to increase the basket size in each of these purchases.
             - Given that customers in the biggest cluster purchase from Divided, Women's Everday Collection and Lingeries/Tights categories, an understanding of their demographics would allow for further expansion into other categories. For instance, is it adult women buying clothes for their teenage daughters? Adult women who also like the Divided category? Or teenage girls who wear Women's Everyday Collection?
             - The results of this study should be used to deepen the understanding of H&M's customers shopping behaviors. First, the clusters can be used as inputs for a model to predict shopping patterns and make recommendations. Second, coupled with interviews and surveys, the clusters can be improved with the definition of user personas and a better understanding of shopping decisions in each segment. 
"""