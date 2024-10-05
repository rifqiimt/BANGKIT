import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from babel.numbers import format_currency
from scipy import stats
import numpy as np
from sklearn.cluster import KMeans

def create_rentals_per_year_df(df):
    rentals_per_year = df.groupby('yr')['cnt'].sum().reset_index()
    return rentals_per_year

def create_users_notholiday_and_holiday_df(df):
    users_notholiday_and_holiday = df.holiday.value_counts()
    return users_notholiday_and_holiday

def create_rentals_per_season_df(df):
    rentals_per_season = df.groupby('season')['cnt'].sum()
    return rentals_per_season

def create_users_type_df(df):
    total_casual_users = df['casual'].sum()
    total_registered_users = df['registered'].sum()

    user_type = {
        'Type of Users': ['Casual Users', 'Registered Users'],
        'Users Total': [total_casual_users, total_registered_users]
    }
    return user_type

# Fungsi RFM Analysis yang dimodifikasi
def create_rfm_analysis(df):
    df['dteday'] = pd.to_datetime(df['dteday'])
    last_date = df['dteday'].max()
    
    rfm = df.groupby('registered').agg({
        'dteday': lambda x: (last_date - x.max()).days,  # Recency
        'instant': 'count',  # Frequency
        'cnt': 'sum'  # Monetary (dalam hal ini, total rentals)
    })
    
    rfm.columns = ['recency', 'frequency', 'monetary']
    
    # Gunakan pd.cut() alih-alih pd.qcut()
    r_labels = range(3, 0, -1)
    f_labels = range(1, 4)
    m_labels = range(1, 4)
    
    r_bins = pd.cut(rfm['recency'], bins=3, labels=r_labels, include_lowest=True)
    f_bins = pd.cut(rfm['frequency'], bins=3, labels=f_labels, include_lowest=True)
    m_bins = pd.cut(rfm['monetary'], bins=3, labels=m_labels, include_lowest=True)
    
    rfm['R'] = r_bins
    rfm['F'] = f_bins
    rfm['M'] = m_bins
    
    rfm['RFM_Score'] = rfm['R'].astype(str) + rfm['F'].astype(str) + rfm['M'].astype(str)
    
    return rfm

# Fungsi untuk clustering sederhana
def create_weather_clusters(df):
    X = df[['temp', 'hum']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['weather_cluster'] = kmeans.fit_predict(X)
    
    return df

# Fungsi untuk analisis tren musiman
def analyze_seasonal_trend(df):
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['month'] = df['dteday'].dt.month
    monthly_rentals = df.groupby('month')['cnt'].mean().reset_index()
    
    # Menghitung tren musiman
    seasonal_trend = stats.zscore(monthly_rentals['cnt'])
    monthly_rentals['seasonal_trend'] = seasonal_trend
    
    return monthly_rentals

# Load data
all_df = pd.read_csv("main_data.csv")

# Sidebar
with st.sidebar:
    all_df["dteday"] = pd.to_datetime(all_df["dteday"])
    min_date = all_df["dteday"].min()
    max_date = all_df["dteday"].max()

    # Mengambil start_date & end_date dari date_input
    start_date, end_date = st.date_input(
        label='Time Range',min_value=min_date,
        max_value=max_date,
        value=[min_date, max_date]
    )

# Create dataframes
rentals_per_year = create_rentals_per_year_df(all_df)
users_notholiday_and_holiday = create_users_notholiday_and_holiday_df(all_df)
rentals_per_season = create_rentals_per_season_df(all_df)
user_type = create_users_type_df(all_df)

# Aplikasikan fungsi-fungsi baru
rfm_result = create_rfm_analysis(all_df)
all_df = create_weather_clusters(all_df)
seasonal_trend = analyze_seasonal_trend(all_df)

# Main content
st.header('Bike Sharing Dashboard')

# Rentals by Year
st.subheader('Number of Rentals per Year')

fig, ax = plt.subplots(figsize=(10, 5))
sns.barplot(
    x="yr",
    y="cnt",
    data=rentals_per_year,
    ax=ax
)

plt.title("Number of Rentals per Year", loc="center", fontsize=15)
plt.xlabel("Year")
plt.ylabel("Number of Rentals")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.tick_params(axis='x', labelsize=12)
st.pyplot(fig)

#Number of Users on Holiday and Not Holiday
st.subheader('Number of Users on Holiday and Not Holiday')

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    y=users_notholiday_and_holiday.values,
    x=users_notholiday_and_holiday.index,
    ax=ax
)

plt.title("Number of Users on Holiday and Not Holiday", loc="center", fontsize=15)
plt.ylabel(None)
plt.xlabel(None)
plt.tick_params(axis='x', labelsize=12)
st.pyplot(fig)

#Number of Rentals per Season
st.subheader('Number of Rentals per Season')

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    y=rentals_per_season.values,
    x=rentals_per_season.index,
    ax=ax
)

plt.title("Number of Rentals per Season", loc="center", fontsize=15)
plt.ylabel("Number of Rentals")
plt.xlabel("Season")
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
plt.tick_params(axis='x', labelsize=12)
st.pyplot(fig)

#User Type
st.subheader('Number of Casual Users and Registered Users')

fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    x='Type of Users',
    y='Users Total',
    data=user_type,
    ax=ax)

plt.title('Number of Casual Users and Registered Users', loc="center", fontsize=15)
plt.ylabel('Users Total')
plt.xlabel('Type of Users')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x))))
st.pyplot(fig)

# RFM Analysis
if not rfm_result.empty and 'RFM_Score' in rfm_result.columns:
    st.subheader('RFM Analysis')
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=rfm_result, x='recency', y='frequency', size='monetary', hue='RFM_Score', ax=ax)
    plt.title('RFM Analysis')
    st.pyplot(fig)
else:
    st.warning("Tidak cukup data untuk melakukan RFM Analysis")

# Weather Clusters
st.subheader('Weather Clusters')
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=all_df, x='temp', y='hum', hue='weather_cluster', ax=ax)
plt.title('Weather Clusters')
st.pyplot(fig)

# Seasonal Trend
st.subheader('Seasonal Trend')
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=seasonal_trend, x='month', y='seasonal_trend', ax=ax)
plt.title('Seasonal Trend in Bike Rentals')
plt.xlabel('Month')
plt.ylabel('Trend (Z-Score)')
st.pyplot(fig)