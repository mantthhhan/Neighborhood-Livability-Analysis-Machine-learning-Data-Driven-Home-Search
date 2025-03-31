
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import folium
import random


# np.random.seed(42)
# n_neighborhoods = 20
# neighborhoods = [f'Neighborhood_{i+1}' for i in range(n_neighborhoods)]

# data = {
#     'neighborhood': neighborhoods,
#     'latitude': np.random.uniform(37.7, 37.8, n_neighborhoods),
#     'longitude': np.random.uniform(-122.5, -122.4, n_neighborhoods),
#     'crime_rate': np.random.uniform(1, 10, n_neighborhoods),
#     'rental_price': np.random.uniform(2000, 5000, n_neighborhoods),
#     'parks_nearby': np.random.randint(0, 10, n_neighborhoods),
#     'grocery_stores': np.random.randint(0, 8, n_neighborhoods),
#     'transit_stops': np.random.randint(0, 15, n_neighborhoods)
# }

# df = pd.DataFrame(data)

# print("Sample neighborhood data:")
# print(df.head())
# print("\
# Data summary:")
# print(df.describe())

oakland_cities = [
    'Troy, MI', 'Farmington Hills, MI', 'Rochester Hills, MI', 'Royal Oak, MI', 
    'Novi, MI', 'Pontiac, MI', 'Auburn Hills, MI', 'Birmingham, MI',
    'Southfield, MI', 'West Bloomfield Township, MI', 'Bloomfield Township, MI',
    'Waterford Township, MI', 'Commerce Township, MI', 'White Lake Township, MI',
    'Oakland Township, MI'
]

import geocoder
import time
import pandas as pd
import numpy as np

oakland_data = {
    'city': [
        'Troy', 'Farmington Hills', 'Rochester Hills', 'Royal Oak', 
        'Novi', 'Pontiac', 'Auburn Hills', 'Birmingham',
        'Southfield', 'West Bloomfield Township', 'Bloomfield Township',
        'Waterford Township', 'Commerce Township', 'White Lake Township',
        'Oakland Township'
    ],
    'latitude': [
        42.6064, 42.4814, 42.6583, 42.4895,
        42.4806, 42.6389, 42.6875, 42.5467,
        42.4733, 42.5567, 42.5836,
        42.6907, 42.5853, 42.6493,
        42.7492
    ],
    'longitude': [
        -83.1497, -83.3771, -83.1499, -83.1446,
        -83.4755, -83.2907, -83.2341, -83.2154,
        -83.2218, -83.3827, -83.2455,
        -83.4033, -83.4855, -83.4919,
        -83.1044
    ]
}

df = pd.DataFrame(oakland_data)

np.random.seed(42)

df['crime_rate'] = df['city'].map({
    'Birmingham': 2, 'Troy': 2, 'Rochester Hills': 2, 'Novi': 2,
    'Auburn Hills': 3, 'West Bloomfield Township': 2, 'Bloomfield Township': 2,
    'Royal Oak': 3, 'Farmington Hills': 3, 'Oakland Township': 1,
    'Commerce Township': 3, 'White Lake Township': 3, 'Waterford Township': 4,
    'Southfield': 5, 'Pontiac': 7
})

df['rental_price'] = df['city'].map({
    'Birmingham': 2800, 'Troy': 2200, 'Rochester Hills': 2000, 'Novi': 2100,
    'Auburn Hills': 1800, 'West Bloomfield Township': 2300, 'Bloomfield Township': 2500,
    'Royal Oak': 1900, 'Farmington Hills': 1800, 'Oakland Township': 2400,
    'Commerce Township': 1900, 'White Lake Township': 1700, 'Waterford Township': 1600,
    'Southfield': 1500, 'Pontiac': 1200
})

df['parks_nearby'] = np.random.randint(2, 15, size=len(df))
df['grocery_stores'] = np.random.randint(3, 12, size=len(df))
df['transit_stops'] = np.random.randint(2, 20, size=len(df))

df.loc[df['city'].isin(['Royal Oak', 'Birmingham', 'Pontiac', 'Southfield']), 'transit_stops'] += 10

df['safety_score'] = ((10 - df['crime_rate']) / 9 * 100).round(1)  
df['affordability_score'] = ((3000 - df['rental_price']) / 1800 * 100).round(1)  
df['livability_score'] = (
    (df['parks_nearby'] / df['parks_nearby'].max() * 0.4 +
     df['grocery_stores'] / df['grocery_stores'].max() * 0.3 +
     df['transit_stops'] / df['transit_stops'].max() * 0.3) * 100
).round(1)

df['overall_score'] = (
    df['safety_score'] * 0.4 +
    df['affordability_score'] * 0.3 +
    df['livability_score'] * 0.3
).round(1)

df_sorted = df.sort_values('overall_score', ascending=False)

print("Oakland County Cities Analysis:")
print("\
Top Cities by Overall Score:")
print(df_sorted[['city', 'overall_score', 'safety_score', 'affordability_score', 'livability_score']].head())

import folium
from folium import plugins
import branca.colormap as cm

map_center = [42.5922, -83.3362]
m = folium.Map(location=map_center, zoom_start=11)

colormap = cm.LinearColormap(
    colors=['red', 'yellow', 'green'],
    vmin=df['overall_score'].min(),
    vmax=df['overall_score'].max()
)
m.add_child(colormap)

for idx, row in df.iterrows():
    popup_content = f"""
    <b>{row['city']}</b><br>
    Overall Score: {row['overall_score']}<br>
    Safety Score: {row['safety_score']}<br>
    Affordability Score: {row['affordability_score']}<br>
    Livability Score: {row['livability_score']}<br>
    Average Rent: ${row['rental_price']}<br>
    Parks Nearby: {row['parks_nearby']}<br>
    Grocery Stores: {row['grocery_stores']}<br>
    Transit Stops: {row['transit_stops']}
    """
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=10,
        popup=folium.Popup(popup_content, max_width=300),
        color=colormap(row['overall_score']),
        fill=True,
        fill_color=colormap(row['overall_score'])
    ).add_to(m)

m.save('oakland_county_map.html')

print("Interactive map has been created and saved as 'oakland_county_map.html'")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
plt.style.use('seaborn-v0_8')

df_sorted = df.sort_values('overall_score', ascending=True)

bars = plt.barh(df_sorted['city'], df_sorted['overall_score'], 
                color='#766CDB', alpha=0.7)

plt.title('Oakland County Cities - Overall Livability Scores', 
          fontsize=20, pad=15)
plt.xlabel('Overall Score', fontsize=16, labelpad=10)
plt.ylabel('City', fontsize=16, labelpad=10)

for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}', 
             ha='left', va='center', fontsize=10)

plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

print("Analysis complete with map and score visualization")