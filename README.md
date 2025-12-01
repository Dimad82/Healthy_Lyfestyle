
#Healthy Lifestyle data analysing with Pandas, Seaborn and MatPlotLib. Main question is correlation between yearly working hours and life expectancy.

#Importing necessary libraries

#Imports all the necessary libraries
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import random

#Imports Kaggle for reading directly the dataset
import kagglehub

#Downloads the dataset
path = kagglehub.dataset_download("tan5577/healthy-life-style-dataset2025")

print("Path to dataset files:", path)

#Reads csv file and converting to Dataframe
df = pd.read_csv(path + "/Healthy-Lifestyle (2021).csv")

#Checking the dataset

#Check the first 5 rows
df.head()

#Check the dataset
df.info()

#Cleaning the data

#Cleaning the dataset
df.isnull().sum

#Cleaning the dataset
df[df.duplicated()]

#Checking the info about dataset
df.describe()

#Checking the columns
df.columns

#Changing the column names (I have seen this somewhere and thought it's easier to use fürther)
df.columns = ['city','rank','sun_hours','water_bottle_price','obesity','life_expectancy','pollution','yearly_working_hours','happiness_level','outdoor_activities','take_out_places','gym_cost']

#Rechecking the cleaned dataset
df.head()

#Rechecking the cleaned dataset
df.tail()

#Rechecking the cleaned dataset
df.sample(5)

#Rechecking the cleaned dataset
df.info()

#Rechecking the cleaned dataset
df

#Cleaning the dataset
df.isnull().sum()

#Cleaning the dataset
df.dropna(inplace=True)

#Rechecking the dataset
df.head()

#Removing the signs pound and percent

#Remove pound sign and convert to float
df['water_bottle_price'] = (
    df['water_bottle_price']
    .replace('£', '', regex=True)
    .astype(float)
)

#Remove percent sign and convert to float
df['obesity'] = (
    df['obesity']
    .replace('%', '', regex=True)
    .astype(float)
)

#Remove pound sign and convert to float
df['take_out_places'] = (
    df['take_out_places']
    .replace('£', '', regex=True)
    .astype(float)
)

#Rechecking the dataset
df.head()

#Starting the visualizations

#Writes for Sunshine hours the mean, variance and standard deviations
mean = df['sun_hours'].mean()
var = df['sun_hours'].var()
std = df['sun_hours'].std()
print(f"Mean = {mean}\nVariance = {var}\nStandard Deviation = {std}")

#It shows the distribution of sunshine hours but no outliers
print("Sunshine hours")
plt.figure(figsize=(10,2))
plt.boxplot(df['sun_hours'], vert=False, showmeans=True)
plt.grid(color='gray', linestyle='dotted')
#plt.xticks(rotation='vertical')
plt.tight_layout()
plt.show()

#Visualise distribution of Outdoor Activities
plt.figure(figsize=(10,6))
sns.histplot(df['outdoor_activities'], bins=30, kde=True)
plt.title('Distribution of Outdoor Activities')
plt.xlabel('outdoor_activities')
plt.ylabel('obesity')
plt.show()

#The happiness level in cities
print("The happiness level in cities")
sns.catplot(x="happiness_level", y="city", kind="bar", data=df.nlargest(20, 'happiness_level',));

# Scatter and bar graphs of all numerical variable
sns.pairplot(df);

#Correlation between Obesity and life expectancy in cities
print("Obesity vs Life Expectancy")
sns.relplot(
    data=df,
    x="obesity",
    y="life_expectancy",
    hue="city",
)

plt.show()

#Correlation between life expectancy and outdoor activities
print("Life expectancy vs yearly working hours")
sns.relplot(
    data=df,
    x="life_expectancy",
    y="outdoor_activities",
)

plt.show()

#Correlation between Outdoor activities and gym prices
sns.relplot(
    data=df,
    x="outdoor_activities",
    y="gym_cost",
    hue="city",       # color by city
    size="city",      # point size by city
    style="city",     # marker style by city
    palette="Set2",   # optional: nicer color palette
    sizes=(40, 200),  # optional: control min/max point sizes
)

plt.title("Outdoor Activities vs Gym Cost by City")
plt.show()


sns.relplot(
    data=df,
    x="outdoor_activities",
    y="gym_cost",
    hue="yearly_working_hours",     #Color encodes yearly working hours
    size="life_expectancy",   #Cbubble size encodes life expectancy
    style="city",             #Marker style encodes city
    palette="coolwarm",
    sizes=(40, 200),
)
plt.title("Activities vs Gym Cost vs Work vs Life Expectancy")
plt.show()

sns.relplot(
    data=df,
    x="outdoor_activities",
    y="obesity",
    kind="scatter"
)
plt.title("Relationship between Outdoor Activities and Obesity Rate")
plt.xlabel("Average Outdoor Activity Hours")
plt.ylabel("Obesity Rate (%)")
plt.title("Obesity vs Outdoor Activities")
plt.show()

#With this i filtered Berlin
df_berlin = df[df["city"] == "Berlin"]

#It shows Berlin information filtered from cities
df_berlin.head()

#Pie chart of only europian cities
europe_cities = ["Berlin", "Frankfurt", "Dublin", "Brussels", "Geneva", "Zurich", "Istanbul", "Paris", "Milan", "Madrid", "Barcelona", "Vienna", "Amsterdam", "Helsinki", "Copenhagen", "Stockholm", "Moscow"]
df_europe = df[df["city"].isin(europe_cities)]

fig = px.pie(
    df_europe,
    values="obesity",
    names="city",
    hole=0.1,
    title="European Cities VS Obesity"
)

fig.update_traces(textinfo="label+value") #Show both city and obesity values directly on the chart

fig.update_layout(showlegend=False) #Hides the legend on the right side

fig.update_layout(title_x=0.5) #Center the title


fig.show()

#Correlation between life expectancy and gym prices
print("Life expectancy vs gym cost")
sns.relplot(
    data=df,
    x="life_expectancy",
    y="gym_cost",
)

plt.show()

#Correlation between outdoor activities and gym prices
print("Outdoor activities vs Gym cost")
sns.relplot(
    data=df,
    x="outdoor_activities",
    y="gym_cost",
)

plt.show()

#Correlation between life expectancy and yearly working hours
sns.relplot(
    data=df,
    x="life_expectancy",
    y="yearly_working_hours",
)

plt.show()

pivot = df.pivot_table(values="life_expectancy", index="city", columns="yearly_working_hours")
sns.heatmap(pivot, cmap="YlGnBu")
plt.title("Life Expectancy by City and Working Hours")
plt.show()

#Correlation between life expectancy and yearly working hours
df["yearly_working_hours"] = pd.to_numeric(df["yearly_working_hours"], errors="coerce")

df_sorted = df.sort_values(by="yearly_working_hours", ascending=True)

sns.relplot(
    data=df,
     x="yearly_working_hours",
    y="life_expectancy",
    kind="scatter",
    height=5,     # vertical size
    aspect=2      # width/height ratio (2 makes it twice as wide)
).set(title="Correlation between Life Expectancy and Yearly Working Hours")

plt.show()

#I did 3 Correlations that where important in this work and i was surprised with the results. First one was as expected between life expectancy and gym costs and this shows us that as more expensive is, that it is affordable in the cities with high cost of living and this people live longer. It means that they are wealthy and can afford to maintain healthy habbits. Second is very surprising the correlation between gym cost and outdoor activities. As more the gym costs the less they go outdoor. But it could mean that that is happening in wealthy cities but they have hursher winters so they stay in. Third is the correlation between annual working hours and life expectancy and it shows us that the less people work they live longer.  
