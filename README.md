# Neighborhood-Livability-Analysis-Machine-learning-Data-Driven-Home-Search
This project was developed as a data-driven approach to finding the perfect neighborhood when relocating. Facing the challenge of choosing a new home with safety, affordability, and excellent amenities in mind, I collected and analyzed various datasets to provide actionable insights.

Project Highlights </b>

Data Collection:
Gathered data from multiple sources:

Crime Data: Extracted using state APIs and Kaggle datasets.

Rental Prices: Scraped from Zillow, Craigslist, and rental APIs.

Amenities Data: Retrieved using OpenStreetMap’s Overpass API for parks, grocery stores, and transit stops.

Data Cleaning & Integration:
Used Pandas to clean and merge data from various formats into a structured dataset.

Modeling & Analysis:

Safety: Implemented K-means clustering on crime data to classify neighborhoods into low, medium, and high-crime zones.

Affordability: Built a Linear Regression model on rental price data to predict trends.

Livability: Scored neighborhoods based on the availability and accessibility of amenities.

Visualization:
Developed an interactive Folium map to visualize the top 15 neighborhoods based on a weighted scoring model that combines safety, affordability, and amenities.


![Figure_1](https://github.com/user-attachments/assets/6d2a61c6-9bb3-4657-b825-c67e8600f81c)
