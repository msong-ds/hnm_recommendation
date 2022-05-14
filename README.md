# Clustering H&M Customers by Their Shopping Patterns

This visualization is inspired by a kaggle competition hosted by H&M.

The purposes of clustering are three. First, it is to understand the dataset better and gain insights for further analysis. Second, cluster information can be used as a feature when building a model. Finally, clustering can be used to identify customer segments. Customer segmentation is important to understand and discover new opportunities to improve business metrics.

Nine clusters were identified using K-means clustering.

The dataset was originally provided by H&M, from which new features were generated to represent customers' shopping patterns and the dataset was aggregated by unique customer ID. The new features are the number of items that each customer bought in each category. 

Check out the Steamlit visualization [here](https://share.streamlit.io/msong-ds/hnm_recommendation/main/streamlit_app/streamlit_app.py).
