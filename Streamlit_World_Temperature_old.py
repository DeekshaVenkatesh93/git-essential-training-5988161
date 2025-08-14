import pandas as pd
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

path1=r'C:\Users\koush\OneDrive\Desktop\ProjectWork\Global-mean monthly, seasonal, and annual means.csv'
path2=r'C:\Users\koush\OneDrive\Desktop\ProjectWork\owid-co2-data.csv'
path3=r'C:\Users\koush\OneDrive\Desktop\ProjectWork\Countries.csv'
path4=r'C:\Users\koush\OneDrive\Desktop\ProjectWork\Northern Hemisphere-mean monthly, seasonal, and annual means.csv'
path5=r'C:\Users\koush\OneDrive\Desktop\ProjectWork\Southern Hemisphere-mean monthly, seasonal, and annual means.csv'

st.image(r'C:\Users\koush\DataScientest\World_tempertaure.png',use_column_width='True')
st.title("World Temperature")
   ###st.sidebar.image(r'C:\Users\koush\DataScientest\tempertaure_image1.jpg')
st.sidebar.title("Table of contents")
pages=["Exploration", "DataVizualization", "Modelling","Prediction"]
page=st.sidebar.radio("Go to", pages)
glob=pd.read_csv(path1,header=1)
nor=pd.read_csv(path4,header=1)
sou=pd.read_csv(path5,header=1)
   ##########-----------Global_Northern_Southern-----------#############
glob=glob[(glob["Year"]!=2024)&(glob["Year"]!=2023)]
#print(glob["Year"].unique()) --- data is wihtout year 2024 --> OK, choose only columns year and J-D which are those of interest
#-----N.B. glob_prep the DF we are going to merge with the others below for the preprop. step---------------------------------
glob_prep=glob[["Year","J-D"]]
glob_prep["J-D"]=glob_prep["J-D"].astype(float)
dictionary={"Year":"year"}
glob_prep=glob_prep.rename(dictionary, axis=1)
####st.dataframe(glob_prep.info())
####st.dataframe(glob_prep.head())
index_glob_prep=glob_prep.index

df=pd.read_csv(path2)
emissions=df[["country","year","cement_co2","co2_including_luc","coal_co2","consumption_co2","gas_co2","methane",
              "nitrous_oxide","oil_co2","other_industry_co2", "co2_including_luc_per_capita", "cumulative_co2", "cumulative_coal_co2",
             "cumulative_gas_co2"]]
#####   1)  dfco2_inc_luc where country = world
dfco2_inc_luc=emissions[["country","year","co2_including_luc"]]
dfco2_inc_luc=dfco2_inc_luc[(dfco2_inc_luc["country"]=="World")&(dfco2_inc_luc["year"]>=1880)]
dfco2_inc_luc=dfco2_inc_luc[["year","co2_including_luc"]]
dfco2_inc_luc=dfco2_inc_luc.set_index(index_glob_prep)
####st.dataframe(dfco2_inc_luc.info())
####st.dataframe(dfco2_inc_luc.head())
#----we prepare the data for concat naming c1
c1=dfco2_inc_luc["co2_including_luc"]
#####   2)  dfcoal_co2 where country = world
dfcoal_co2=emissions[["country","year","coal_co2"]]
dfcoal_co2=dfcoal_co2[(dfcoal_co2["country"]=="World")&(dfcoal_co2["year"]>=1880)]
dfcoal_co2=dfcoal_co2[["year","coal_co2"]]
dfcoal_co2=dfcoal_co2.set_index(index_glob_prep)
####st.dataframe(dfcoal_co2.info())
#----we prepare the data for concat naming c2
c2=dfcoal_co2["coal_co2"]
#####   3)  dfgas_co2 where country = world
dfgas_co2=emissions[["country","year","gas_co2"]]
dfgas_co2=dfgas_co2[(dfgas_co2["country"]=="World")&(dfgas_co2["year"]>=1880)]
dfgas_co2=dfgas_co2[["year","gas_co2"]]
dfgas_co2=dfgas_co2.set_index(index_glob_prep)
####st.dataframe(dfgas_co2.info())
#----we prepare the data for concat naming c3
c3=dfgas_co2["gas_co2"]
#####   4)  dfoil_co2 where country = world
dfoil_co2=emissions[["country","year","oil_co2"]]
dfoil_co2=dfoil_co2[(dfoil_co2["country"]=="World")&(dfoil_co2["year"]>=1880)]
dfoil_co2=dfoil_co2[["year","oil_co2"]]
dfoil_co2=dfoil_co2.set_index(index_glob_prep)
####st.dataframe(dfoil_co2.info())
#----we prepare the data for concat naming c4
c4=dfoil_co2["oil_co2"]

#####   5)  dfcement_co2 where country = world
dfcement_co2=emissions[["country","year","cement_co2"]]
dfcement_co2=dfcement_co2[(dfcement_co2["country"]=="World")&(dfcement_co2["year"]>=1880)]
dfcement_co2=dfcement_co2[["year","cement_co2"]]
dfcement_co2=dfcement_co2.set_index(index_glob_prep)
####st.dataframe(dfcement_co2.info())
#----we prepare the data for concat naming c5
c5=dfcement_co2["cement_co2"]
##### +++++++++++++  OTHER COLUMNS    ++++++++++++++++         ##############
#####   6)  dfother_industry_co2 where country = world
dfother_industry_co2=emissions[["country","year","other_industry_co2"]]
dfother_industry_co2=dfother_industry_co2[(dfother_industry_co2["country"]=="World")&(dfother_industry_co2["year"]>=1880)]
dfother_industry_co2=dfother_industry_co2[["year","other_industry_co2"]]
dfother_industry_co2=dfother_industry_co2.set_index(index_glob_prep)
####st.dataframe(dfother_industry_co2.info())
####st.dataframe(dfother_industry_co2.head())
####st.dataframe(dfother_industry_co2.tail(10))
#----we prepare the data for concat naming c6
c6=dfother_industry_co2["other_industry_co2"]

#####   7)  dfco2_including_luc_per_capita  where country = world
dfco2_including_luc_per_capita =emissions[["country","year","co2_including_luc_per_capita"]]
dfco2_including_luc_per_capita =dfco2_including_luc_per_capita [(dfco2_including_luc_per_capita ["country"]=="World")&(dfco2_including_luc_per_capita ["year"]>=1880)]
dfco2_including_luc_per_capita =dfco2_including_luc_per_capita [["year","co2_including_luc_per_capita"]]
dfco2_including_luc_per_capita =dfco2_including_luc_per_capita .set_index(index_glob_prep)
####st.dataframe(dfco2_including_luc_per_capita .info())
####st.dataframe(dfco2_including_luc_per_capita .head())
#----we prepare the data for concat naming c1
c7=dfco2_including_luc_per_capita ["co2_including_luc_per_capita"]
#####   8)  dfcumulative_co2  where country = world
dfcumulative_co2 =emissions[["country","year","cumulative_co2"]]
dfcumulative_co2 =dfcumulative_co2 [(dfcumulative_co2 ["country"]=="World")&(dfcumulative_co2["year"]>=1880)]
dfcumulative_co2 =dfcumulative_co2[["year","cumulative_co2"]]
dfcumulative_co2 =dfcumulative_co2.set_index(index_glob_prep)
####st.dataframe(dfcumulative_co2.info())
####st.dataframe(dfcumulative_co2.head())
#----we prepare the data for concat naming c1
c8=dfcumulative_co2["cumulative_co2"]
#####   9)  dfcumulative_coal_co2  where country = world
dfcumulative_coal_co2 =emissions[["country","year","cumulative_coal_co2"]]
dfcumulative_coal_co2 =dfcumulative_coal_co2 [(dfcumulative_coal_co2 ["country"]=="World")&(dfcumulative_coal_co2["year"]>=1880)]
dfcumulative_coal_co2 =dfcumulative_coal_co2[["year","cumulative_coal_co2"]]
dfcumulative_coal_co2 =dfcumulative_coal_co2.set_index(index_glob_prep)
####st.dataframe(dfcumulative_coal_co2.info())
####st.dataframe(dfcumulative_coal_co2.head())
#----we prepare the data for concat naming c1
c9=dfcumulative_coal_co2["cumulative_coal_co2"]
#####   10)  dfcumulative_gas_co2  where country = world
dfcumulative_gas_co2 =emissions[["country","year","cumulative_gas_co2"]]
dfcumulative_gas_co2 =dfcumulative_gas_co2 [(dfcumulative_gas_co2 ["country"]=="World")&(dfcumulative_gas_co2["year"]>=1880)]
dfcumulative_gas_co2 =dfcumulative_gas_co2[["year","cumulative_gas_co2"]]
dfcumulative_gas_co2 =dfcumulative_gas_co2.set_index(index_glob_prep)
####st.dataframe(dfcumulative_gas_co2.info())
####st.dataframe(dfcumulative_gas_co2.head())
#----we prepare the data for concat naming c1
c10=dfcumulative_gas_co2["cumulative_gas_co2"]
#####--------- MERGE THE DATAFRAMES ABOVE TO OBTAIN THE DF FOR PREP-------------------
prep=pd.concat([glob_prep,c1,c2,c3,c4,c5,c7,c8,c9,c6,c10] ,axis=1)
prep=prep.set_index(["year"]) 


if page == pages[0] : 
        st.write("### Exploration of data")
        st.dataframe(prep.head(10))
        st.write(prep.shape)
        st.dataframe(prep.describe())
####st.dataframe(prep.tail())
####st.dataframe(prep.info())
        if st.checkbox("Show NA") :
         st.dataframe(prep.isna().sum())


if page == pages[1] : 
    st.write("### DataVizualization")
    glob=glob.loc[glob["Year"]!=2024]
    glob["J-D"]=glob["J-D"].astype(float)
    #glob.info()
    nor=nor.loc[nor["Year"]!=2024]
    nor["J-D"]=nor["J-D"].astype(float)
    #nor.info()
    sou=sou.loc[sou["Year"]!=2024]
    sou["J-D"]=sou["J-D"].astype(float)
    #sou.info()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x = glob["Year"], # column world rank
                         y = glob["J-D"], name="Global_mean_annual"))
    fig.add_trace(go.Scatter(x = nor["Year"], # column world rank
                         y = nor["J-D"], name="Northern_mean_annual"))
    fig.add_trace(go.Scatter(x = sou["Year"], # column world rank
                         y = sou["J-D"], name="Southern_mean_annual"))
    fig.update_layout(autosize = False,
                 width = 1100, # Figure width
                 height = 600)
#fig.update_layout(legend_title = 'Glob, North and South Cha of Temp')    # Legend title

    fig.update_layout(title='Global, Northern and Southern Mean Annual Change of Temperature from 1880 to 2023',  # title
                   xaxis_title='Year',   # x label
                   yaxis_title='Change of Temperature in °C')

    st.write(fig)

#########################-----------CO2 contributes----------------###########################
    df=pd.read_csv(path2)
####st.dataframe(df.head())
    df=df[["country","year","cement_co2","co2_including_luc","coal_co2","consumption_co2","gas_co2","methane","nitrous_oxide","oil_co2","other_industry_co2"]]
####st.dataframe(df.tail())

# line plot with plotly of  co2_cement,coal_co2, consumption_co2,oil_co2 -------- WORLD-----GLOBAL
    dfcement_co2=df[["country","year","cement_co2"]]
    dfcement_co2=dfcement_co2.dropna(axis=0,how="all", subset="cement_co2")
    dfcement_co2=dfcement_co2.loc[(dfcement_co2["country"]=="World")&(dfcement_co2["year"]>=1870)]
####st.dataframe(dfco2.info())

    dfcoal_co2=df[["country","year","coal_co2"]]
    dfcoal_co2=dfcoal_co2.dropna(axis=0,how="all", subset="coal_co2")
    dfcoal_co2=dfcoal_co2.loc[(dfcoal_co2["country"]=="World")&(dfcoal_co2["year"]>=1870)]
    dfoil_co2=df[["country","year","oil_co2"]]
    dfoil_co2=dfoil_co2.dropna(axis=0,how="all", subset="oil_co2")
    dfoil_co2=dfoil_co2.loc[(dfoil_co2["country"]=="World")&(dfoil_co2["year"]>=1870)]

    dfgas_co2=df[["country","year","gas_co2"]]
    dfgas_co2=dfgas_co2.dropna(axis=0,how="all", subset="gas_co2")
    dfgas_co2=dfgas_co2.loc[(dfgas_co2["country"]=="World")&(dfgas_co2["year"]>=1870)]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = dfcement_co2["year"], # column world rank
                         y = dfcement_co2["cement_co2"], name="cement_co2"))
    fig.add_trace(go.Scatter(x = dfcoal_co2["year"], # column world rank
                         y = dfcoal_co2["coal_co2"], name="coal_co2"))
    #fig.add_trace(go.Scatter(x = dfconsumption_co2["year"], # column world rank
                         #y = dfconsumption_co2["consumption_co2"], name="consumption_co2"))
    fig.add_trace(go.Scatter(x = dfoil_co2["year"], # column world rank
                         y = dfoil_co2["oil_co2"], name="oil_co2"))
    fig.add_trace(go.Scatter(x = dfgas_co2["year"], # column world rank
                         y = dfgas_co2["gas_co2"], name="gas_co2"))

    fig.update_layout(autosize = False,
                 width = 1000, # Figure width
                 height = 600)
    fig.update_layout(legend_title = 'CO2 Emissions')    # Legend title

    fig.update_layout(title='Global CO2 Emissions Over Years',  # title
                   xaxis_title='Year',   # x label
                   yaxis_title='CO2 in million of tonnes')

    st.write(fig)
##################################-----------CO2 including LUC----------###############
#####---------read countries correct------------
    dfcountries=pd.read_csv(path3)
####st.dataframe(dfcountries.head())
#dfcountries["country"]=dfcountries["country"].replace(to_replace="United States of America",value="United States")
    countries_correct=dfcountries["country"].unique()
###print the list of the correct countries ---SO COUNTRIES_CORRECT is the list of countries where we are going to filter the top 10
        ##print(countries_correct)
        ##print("number of correct countries ", len(countries_correct))
#########    dfco2_inc_luc
    dfco2_inc_luc=df[["country","year","co2_including_luc"]]
#display(dfco2_inc_luc.head())
    dfco2_inc_luc=dfco2_inc_luc.dropna(axis=0,how="all",subset="co2_including_luc")
####st.dataframe(dfco2_inc_luc.info())
#print(countries)
    dfco2_inc_luc=dfco2_inc_luc[dfco2_inc_luc['country'].isin(countries_correct)]
    dfco2_inc_luc=dfco2_inc_luc[dfco2_inc_luc["year"]>=1870]
    countries=dfco2_inc_luc["country"].unique()
        ##print("number of unique countries", len(countries))
#### -----  dfco2_inc_luc graph by country
# top_10_countries-----------
# Calculate median co2_including_luc for each country
    median_co2_including_luc = dfco2_inc_luc.groupby('country')['co2_including_luc'].median().sort_values(ascending=False)
# Get the top 10 countries
    top_countries = median_co2_including_luc.head(5).index
# Filter the dataframe to include only the top 10 or top 5 countries
    dfco2_inc_luc_top = dfco2_inc_luc[dfco2_inc_luc['country'].isin(top_countries)]
    fig=go.Figure()
    for country in top_countries:
        fig.add_trace(
            go.Scatter(x=dfco2_inc_luc_top[dfco2_inc_luc_top["country"] == country]["year"],
                   y=dfco2_inc_luc_top[dfco2_inc_luc_top["country"] == country]["co2_including_luc"],
                   name=country)
        ) 
    fig.update_layout(
          title='CO2 including land use change over the years for 5 countries with the highest emissions ',
            xaxis_title='Year',
           yaxis_title='CO2 in millions of tonnes'
           #,plot_bgcolor = 'white'
            )
    fig.update_layout(autosize = False,
                 width = 1000, # Figure width
                 height = 600)
    st.write(fig)

#########################-----------Map Visualazation----------------###########################
    countries_correct=dfcountries["country"].unique()
    df_map = pd.read_csv(path2)
# filtering out required data for the visuliazation country,year,iso_code,tempertaure change from global warming
    data_tempchange_by_GlobalWarming = df_map[['country', 'year', 'iso_code','share_of_temperature_change_from_ghg']]
    data_tempchange_by_GlobalWarming=data_tempchange_by_GlobalWarming.dropna(axis=0,how="all",subset="share_of_temperature_change_from_ghg")
        #####data_tempchange_by_GlobalWarming.head()
        #####data_tempchange_by_GlobalWarming.info()
# correcting the countrydata used for the plot
    data_tempchange_by_GlobalWarming=data_tempchange_by_GlobalWarming[data_tempchange_by_GlobalWarming['country'].isin(countries_correct)]
    data_tempchange_by_GlobalWarming=data_tempchange_by_GlobalWarming[data_tempchange_by_GlobalWarming["year"]>=1870]
    countries=data_tempchange_by_GlobalWarming["country"].unique()
#print(countries)
#print("number of unique countries", len(countries))
    latest_year = data_tempchange_by_GlobalWarming['year'].max()
    start_year = latest_year - 9 # Check the decade date start year for the given data
####print(start_year)
    min_ghg = data_tempchange_by_GlobalWarming['share_of_temperature_change_from_ghg'].min()
    max_ghg = data_tempchange_by_GlobalWarming['share_of_temperature_change_from_ghg'].max()
####print(min_ghg, max_ghg) # Check the share_of_temperature_change_from_ghg values before the plot

# filter for the negative values in the share_of_temperature_change_from_ghg
    df_filtered = data_tempchange_by_GlobalWarming.loc[data_tempchange_by_GlobalWarming['share_of_temperature_change_from_ghg'] > 0]
###df_filtered.head()
    df_Group=df_filtered.dropna(subset=['share_of_temperature_change_from_ghg', 'country', 'iso_code','year'])

    fig = px.scatter_geo(df_Group[df_Group['year'] > start_year], locations="iso_code", color="share_of_temperature_change_from_ghg",
                     hover_name="country", size="share_of_temperature_change_from_ghg",
                     animation_frame="year",
                     projection="natural earth")
# Add a slider to control the animation and buttons to play/pause the animation
    fig.update_layout(width = 1000, # Figure width
                 height = 600,
    title='Share of Contribution to Global Warming Over Time and Countries',
    updatemenus=[dict(
        type='buttons',
        buttons=[dict(label='Play',
                        method='animate',
                        args=[None, {'frame': {'duration': 1000, 'redraw': True},
                                  'fromcurrent': True, 'transition': {'duration': 500}}]),
                  dict(label='Pause',
                        method='animate',
                        args=[[None], {'frame': {'duration': 0, 'redraw': False},
                                   'mode': 'immediate',
                                   'transition': {'duration': 0}}])
                ])]
)


    st.plotly_chart(fig)



if page == pages[2] : 
      st.write("### Modelling")#   decided to replace the only 2 MISSING VALUES  with fillna method without using SimpleImputer for gas_co2
      prep["gas_co2"]=prep["gas_co2"].fillna(prep["gas_co2"].min())
      ######display(prep.head())
      ##   decided to replace the  MISSING VALUES  with fillna method without using SimpleImputer for other_industry_co2
      prep["other_industry_co2"]=prep["other_industry_co2"].fillna(prep["other_industry_co2"].min())
      ######display(prep.head())

##   decided to replace the  MISSING VALUES  with fillna method without using SimpleImputer for cumulative_gas_co2
      prep["cumulative_gas_co2"]=prep["cumulative_gas_co2"].fillna(prep["cumulative_gas_co2"].min())
      ######display(prep.head())
#   2) Separate the data into a FEATS DataFrame containing the explanatory variables and a TARGET  containing the J-D variable.
      feats = prep.drop("J-D", axis=1)
# target variable is the anomalies change of temperature yearly "J-D" --> from January to December for each year, where year is categorical var
      target = prep["J-D"]

#   3) Create a training set and a test set corresponding to 75% and 25% of the data respectively.
      X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.25, random_state = 42)

#   4) I don't think we have to encode the categorical variable "year" which is already numerical

#   5) Apply standardization using StandardScaler on the variables : "co2_including_luc", "coal_co2	", "gas_co2", "oil_co2", "cement_co2"
# and added "year" to standardize. Let's see. Maybe we should put it as index of the prep DF ??????
      sc = StandardScaler()
      num = ["co2_including_luc", "coal_co2", "gas_co2", "oil_co2", "cement_co2", "other_industry_co2", "co2_including_luc_per_capita",
       "cumulative_co2", "cumulative_coal_co2", "cumulative_gas_co2"]
      X_train[num] = sc.fit_transform(X_train[num])
      X_test[num] = sc.transform(X_test[num])
######------Train the Model-----++++--------Metrics, Model Evaluation
      st.header("Model Selection for results")
      choice = st.selectbox('Choose a Regression Model', ('Linear Regression', 'Random Forest'))

      # Define and fit the chosen model
      if choice == 'Linear Regression': 
          model = LinearRegression()
      elif choice == 'Random Forest':
          model = RandomForestRegressor()
      else:
        st.warning("Please select a valid model")
      model.fit(X_train, y_train)

    # Make predictions
      y_pred = model.predict(X_test)

    # Evaluate model performance
      st.subheader("Model Evaluation")
      st.write(f"R-squared: {model.score(X_test, y_test):.3f}")
    # Store model choice for prediction page
      st.session_state["model_choice"] = choice 

    # Visualize predicted vs. actual temperatures
      fig = plt.figure(figsize = (10,10))
      plt.scatter(y_pred, y_test, c='blue')
      plt.plot((y_test.min(), y_test.max()), (y_test.min(), y_test.max()), color = 'red')
      plt.xlabel("Predicted values")
      plt.ylabel("True values")
      plt.title(f"{choice} - Predicted vs. Actual Temperatures")
      st.pyplot(fig)




 
