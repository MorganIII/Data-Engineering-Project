#load libraries
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
import seaborn as sns
import numpy as np
from scipy import stats
from IPython.display import display
import pyarrow.parquet as pq
from sklearn.preprocessing import LabelEncoder



#create method to plot the missing values in the dataset 
def plot_missing_values(df):
    fig, ax = plt.subplots()
    # plot the missing values in the dataset
    df.isnull().sum().plot(kind='bar', figsize=(15, 10), ax=ax)
    ax.set_title('Missing Values in Dataset')
    ax.set_xlabel('Columns')
    ax.set_ylabel('Missing Values')
    return fig
 

    

############################################################################################################


## Handling Missing data
#### general functions
#to show the distribution for any column
def one_distribution(df, column_value,plot_value):
    exec('sns.'+ plot_value + '(df[column_value])')
    plt.title('Dist. of ' + column_value)
    plt.show()

def two_distribution(df1,df2,column_value):
    
    fig, ax = plt.subplots()
    # plot the curve before imputation
    sns.kdeplot(df1[column_value], color='r', fill=True, ax=ax)
    # plot the curve after imputation
    sns.kdeplot(df2[column_value], color='b', fill=True, ax=ax)
    # plot the legend
    ax.legend(['Before Imputation', 'After Imputation'])
    return fig
    
       
def end_of_tail_imputation(df , target):
    temp = df.copy()
    dev_target = df[target].mean() + 3 * df[target].std()
    temp[target] = temp[target].fillna(dev_target)
    return temp

def mapped_most_frequent_imputation(df , target , mapped):
    temp = df.copy()
    df_1 = temp.groupby(temp[mapped])[target].apply(lambda x: x.value_counts().index[0]).reset_index()
    map_dict = dict(zip(df_1[mapped], df_1[target]))
    temp[target] = temp[target].fillna(temp[mapped].map(map_dict))
    return temp
    
def missing_indicator_imputation(df , target ):
    temp = df.copy()
    Missing_indicator = temp[target].isnull().astype(int)    
    column_index = temp.columns.get_loc(target)+1
    temp.insert(loc= column_index , column= 'Missing_Indicator' ,value=Missing_indicator)
    temp[target] = temp[target].fillna(temp[target].mean())
    return temp
 
def arbitrary_value_imputation(df , target , value):
    temp = df.copy()
    temp[target] = temp[target].fillna(value)
    return temp   
    
    
def replace_values_in_column(df , target , value , new_value):
    temp = df.copy()
    temp[target] = temp[target].replace(value, new_value)
    return temp

def handling_missing_values(df_accident):
    df = df_accident.copy()
    df_accident_imputed1 = end_of_tail_imputation(df,'location_easting_osgr')
    df_accident_imputed2 = end_of_tail_imputation(df_accident_imputed1,'location_northing_osgr')
    df_accident_imputed3 = end_of_tail_imputation(df_accident_imputed2,'longitude')
    df_accident_imputed4 = end_of_tail_imputation(df_accident_imputed3,'latitude')
    df_accident_imputed4 = replace_values_in_column(df_accident_imputed4,'first_road_number','first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ',0.0)
    df_accident_imputed4["first_road_number"] = df_accident_imputed4["first_road_number"].astype(float)
    df_accident_imputed_5 = df_accident_imputed4.dropna(axis='index', subset=['first_road_number'])
    df_accident_imputed_6 = mapped_most_frequent_imputation(df_accident_imputed_5,'road_type','first_road_class')
    df_accident_imputed_6 = replace_values_in_column(df_accident_imputed_6,'second_road_number','first_road_class is C or Unclassified. These roads do not have official numbers so recorded as zero ',0.0)
    df_accident_imputed_6["second_road_number"] = df_accident_imputed_6["second_road_number"].astype(float)
    df_accident_imputed_7 = missing_indicator_imputation(df_accident_imputed_6,'second_road_number')
    df_accident_imputed_8 = arbitrary_value_imputation(df_accident_imputed_7,'weather_conditions','Missing')
    df_accident_imputed_9 = df_accident_imputed_8.drop('special_conditions_at_site', axis=1)
    df_accident_imputed_10 = df_accident_imputed_9.drop('carriageway_hazards', axis=1)
    return df_accident_imputed_10


############################################################################################################

def create_graph(df_accident, target):
    df_imputed = handling_missing_values(df_accident)
    graph = two_distribution(df_accident,df_imputed,target)
    return graph


## Handling Outliers
#### general functions

#visualize the boxplot and the density functions for a column
def visualize(dataframe , target ):
    plt.boxplot(dataframe[target])
    plt.title("distribution of " + target)
    plt.show()
    sns.kdeplot(dataframe[target])
    plt.show()
    
#Detecting and remove outliers using z-score
def detecting_outliers_zscore(dataframe , target):
    temp = dataframe.copy()
    z_score = np.abs(stats.zscore(temp[target]))
    filtered_entries = z_score < 3
    df_zscore_filter = temp[filtered_entries]
    return df_zscore_filter

#Detecting and remove ouliers using IQR
def detecting_outliers_IQR(dataframe , target):
    temp = dataframe.copy()
    Q1 = temp[target].quantile(0.25)
    Q3 = temp[target].quantile(0.75)
    IQR = Q3 - Q1
    cut_off = IQR * 1.5
    lower = Q1 - cut_off
    upper =  Q3 + cut_off
    df_IQR = temp[(temp[target] < upper) & (temp[target] > lower)]
    return df_IQR

#Detecting outliers multivariant using scatter plot
def detecting_outliers_multivariant_scatter(dataframe , target_1 , target_2):
    fig, ax = plt.subplots(figsize=(16,8))
    ax.scatter(dataframe[target_1], dataframe[target_2])
    ax.set_xlabel(target_1)
    ax.set_ylabel(target_2)
    plt.show()

#Detecting outliers multivariant using Local Outlier Factor plot
def detecting_outliers_multivariant_LOF(dataframe , target_1 , target_2):
    clf = LocalOutlierFactor()
    X = dataframe[[target_1,target_2]].values
    y_pred = clf.fit_predict(X)
    plt.figure(figsize=(12,12))
    in_mask = [True if l == 1 else False for l in y_pred]
    out_mask = [True if l == -1 else False for l in y_pred]
    plt.title("Local Outlier Factor (LOF)")
    a = plt.scatter(X[in_mask, 0], X[in_mask, 1], c = 'blue',
                edgecolor = 'k', s = 30)
    # outliers
    b = plt.scatter(X[out_mask, 0], X[out_mask, 1], c = 'red',
                edgecolor = 'k', s = 30)
    plt.axis('tight')
    plt.xlabel(target_1);
    plt.ylabel(target_2);
    plt.show()

#replace outliers values with Quantile-based flooring and capping
def replacing_outliers_quantile_based(dataframe , target):
    temp = dataframe.copy()
    floor = temp[target].quantile(0.10)
    cap = temp[target].quantile(0.90)
    temp[target] = np.where(temp[target] <floor, floor,temp[target])
    temp[target] = np.where(temp[target] >cap,cap,temp[target])
    return temp

#replace outliers values with Mean\Median values 
def replacing_outliers_with_mean(dataframe , target):
    temp = dataframe.copy()
    median = dataframe[target].median()
    cutoff = dataframe[target].mean() + temp[target].std() * 3
    temp[target] = np.where(temp[target]>cutoff, median,temp[target])
    return temp

#Get the number of outliers in a dataframe    
def get_number_of_outliers(dataframe , target):
    temp = dataframe.copy()
    Q1 = temp[target].quantile(0.25)
    Q3 = temp[target].quantile(0.75)
    IQR = Q3 - Q1
    cut_off = IQR * 1.5
    lower = Q1 - cut_off
    upper =  Q3 + cut_off
    df1 = temp[temp[target]> upper]
    df2 = temp[temp[target] < lower]
    no_ouliers =  df1.shape[0]+ df2.shape[0]
    return no_ouliers
    
## Handling outliers for the accident dataset
def handling_outliers(df_accident):
    df_accident_outliers = df_accident.copy()
    df_accident_outliers_1 = detecting_outliers_IQR(df_accident_outliers , 'location_easting_osgr')
    df_accident_outliers_2 = detecting_outliers_IQR(df_accident_outliers_1 , 'longitude' )
    df_accident_outliers_3 = detecting_outliers_IQR(df_accident_outliers_2 , 'location_easting_osgr') 
    df_accident_outliers_4 = replacing_outliers_quantile_based(df_accident_outliers_3 , 'location_northing_osgr')
    df_accident_outliers_5 = replacing_outliers_quantile_based(df_accident_outliers_4 , 'latitude')
    df_accident_outliers_6 = replacing_outliers_quantile_based(df_accident_outliers_5 , 'number_of_vehicles')
    df_accident_outliers_7 = replacing_outliers_quantile_based(df_accident_outliers_6 , 'number_of_casualties')
    df_accident_outliers_8 = replacing_outliers_quantile_based(df_accident_outliers_7 , 'first_road_number')
    df_accident_outliers_9 = replacing_outliers_quantile_based(df_accident_outliers_8 , 'speed_limit')
    df_accident_outliers_10 = replacing_outliers_quantile_based(df_accident_outliers_9 , 'second_road_number')
    return df_accident_outliers_10


################################################################################################


#### Handling UncleanData
###### General Functions

def drop_records_with_negative_one(df , column_name):
    temp_df = df.copy()
    temp_df = temp_df[temp_df[column_name] != '-1']
    return temp_df

def add_week_number(df):
    temp_df = df.copy()
    temp_df['date'] = pd.to_datetime(temp_df['date'] ,format= '%Y-%m-%d')
    temp_df['week_number'] = temp_df['date'].dt.isocalendar().week
    return temp_df

#method that convert the time column to datetime format
def convert_time_to_datetime(df):
    temp_df = df.copy()
    temp_df['time'] = pd.to_datetime(temp_df['time'] ,format= '%H:%M')
    return temp_df

#method that disceretize the date column into years , months ,days and drop the date column
def discretize_date(df):
    temp_df = df.copy()
    temp_df['year'] = temp_df['date'].dt.year
    temp_df['month'] = temp_df['date'].dt.month
    temp_df['day'] = temp_df['date'].dt.day
    temp_df = temp_df.drop('date' , axis = 1)
    return temp_df

#method that disceretize the time column into hours , minutes and drop the time column
def discretize_time(df):
    temp_df = df.copy()
    temp_df['hour'] = temp_df['time'].dt.hour
    temp_df['minute'] = temp_df['time'].dt.minute
    temp_df = temp_df.drop('time' , axis = 1)
    return temp_df

#method that Add Column indicating whether the accident was on a weekend or not
def add_weekend_column(df):
    temp_df = df.copy()
    temp_df['weekend'] = temp_df['day_of_week'].apply(lambda x : 1 if x in ['Saturday' , 'Sunday'] else 0)
    return temp_df


## Handling UncleanData for the accident dataset

def handling_unclean_data(df_accident):
    df_accident_unclean = df_accident.copy()
    df_accident_unclean_data = df_accident_unclean.drop("accident_year" ,axis= 1)
    df_accident_unclean_data_1 = df_accident_unclean_data.drop("accident_reference",axis =1)
    df_accident_unclean_data_2 = df_accident_unclean_data_1.copy()
    df_accident_unclean_data_2['second_road_class'] = df_accident_unclean_data_1['second_road_class'].replace('-1' , 'Did not happen on intersections')
    df_accident_unclean_data_3 = drop_records_with_negative_one(df_accident_unclean_data_2 , 'local_authority_highway')
    df_accident_unclean_data_4 = drop_records_with_negative_one(df_accident_unclean_data_3 , 'lsoa_of_accident_location')
    df_accident_unclean_data_5 = add_week_number(df_accident_unclean_data_4)
    df_accident_unclean_data_6 = convert_time_to_datetime(df_accident_unclean_data_5)
    df_accident_unclean_data_7 = discretize_date(df_accident_unclean_data_6)
    df_accident_unclean_data_8 = discretize_time(df_accident_unclean_data_7)
    df_accident_unclean_data_9 = add_weekend_column(df_accident_unclean_data_8)
    df_accident_unclean_data_9['morning'] = df_accident_unclean_data_9['hour'].apply(lambda x: 1 if x < 12 else 0)
    return df_accident_unclean_data_9




################################################################################################

## Categorical Feature encoding 
#### General Functions



#apply one hot encoding to the categorical features
def one_hot_encoding(df , col_name):
    temp_df = df.copy()
    temp_df = pd.get_dummies(temp_df , columns = [col_name])
    return temp_df
#method that takes the dataset and column and apply label encoding on it and 
def label_encoding(df , col_name):
    temp_df = df.copy()
    if(col_name == 'accident_severity'):
        temp_df[col_name] = temp_df[col_name].apply(lambda x : 0 if x == 'Slight' else 1 if x == 'Serious' else 2)
    le = LabelEncoder()
    temp_df[col_name] = le.fit_transform(temp_df[col_name])
    return temp_df

#method that takes the dataset and column and apply frequency encoding on it
def frequency_encoding(df , col_name):
    temp_df = df.copy()
    temp_df[col_name] = temp_df[col_name].map(temp_df[col_name].value_counts(normalize=True))
    return temp_df

#method to plot rare labels in categorical features
def plot_rare_labels(df, variable, rare_perc):
    temp = df.groupby([variable])[variable].count()/len(df)
    temp_df = pd.DataFrame(temp) 
    plt.figure(figsize=(16, 6))
    temp_df[variable].sort_values(ascending=False).plot.bar()
    plt.axhline(y=rare_perc, color='r', linestyle='-')
    plt.title(variable)
    plt.show()
    
#method to replace rare labels in categorical features
def rare_label_encoding(df, variable, rare_perc):
    temp = df.groupby([variable])[variable].count()/len(df)
    rare_labels = temp[temp<rare_perc].index 
    df[variable] = np.where(df[variable].isin(rare_labels),'Rare', df[variable])
    return df

def encode_lsoa_of_accident_location(df):
    temp_df = df.copy()
    temp_df_1 = df.copy()
    temp_df_2 = df.copy()
    temp_df_1['lsoa_of_accident_location'] = temp_df_1['lsoa_of_accident_location'].astype(str)
    temp_df_1['lsoa_of_accident_location'] = temp_df_1['lsoa_of_accident_location'].apply(lambda x : x[0])
    temp_df['lsoa_of_accident_location_E_W'] = temp_df_1['lsoa_of_accident_location'].apply(lambda x : 1 if x == 'E' else 0)
    temp_df_2['lsoa_of_accident_location'] = temp_df_2['lsoa_of_accident_location'].astype(str)
    temp_df_2['lsoa_of_accident_location'] = temp_df_2['lsoa_of_accident_location'].apply(lambda x : x[1:])
    temp_df['lsoa_of_accident_location_number'] = temp_df_2['lsoa_of_accident_location'].astype(int)
    temp_df = temp_df.drop(['lsoa_of_accident_location'], axis = 1)
    return temp_df


#method that return dataframe with the two columns that are encoded and their mapped values 
def merge_two_columns(temp_df , df1 , df2 , col_name):
    temp_df1 = df1.copy()
    temp_df2 = df2.copy()
    temp_df[col_name] = temp_df1[col_name]
    temp_df[col_name + '_mapped'] = temp_df2[col_name]
    return temp_df


## Handling Categorical Features encoding for the accident dataset

def categorical_encoding(df_accident):
    df = df_accident.copy()
    temp_df = pd.DataFrame()
    df_accident_encoding = rare_label_encoding(df, 'police_force', 0.05)
    df_accident_encoding_1 = one_hot_encoding(df_accident_encoding , 'police_force')
    df_accident_encoding_2 = label_encoding(df_accident_encoding_1 , 'accident_severity')
    temp_df = merge_two_columns(temp_df ,df_accident_encoding_1 , df_accident_encoding_2 , 'accident_severity')
    df_accident_encoding_3 = frequency_encoding(df_accident_encoding_2 , 'day_of_week')
    temp_df_1 = merge_two_columns(temp_df ,df_accident_encoding_2 , df_accident_encoding_3 , 'day_of_week')
    df_accident_encoding_4 = label_encoding(df_accident_encoding_3 , 'local_authority_district')
    df_accident_encoding_5 = label_encoding(df_accident_encoding_4 , 'local_authority_ons_district')
    df_accident_encoding_6 = label_encoding(df_accident_encoding_5 , 'local_authority_highway')
    temp_df_2 = merge_two_columns(temp_df_1 ,df_accident_encoding_3 , df_accident_encoding_4 ,'local_authority_district')
    temp_df_3 = merge_two_columns(temp_df_2 ,df_accident_encoding_4 , df_accident_encoding_5 ,'local_authority_ons_district')
    temp_df_4 = merge_two_columns(temp_df_3 ,df_accident_encoding_5 , df_accident_encoding_6 , 'local_authority_highway')
    df_accident_encoding_7 = one_hot_encoding(df_accident_encoding_6 , 'first_road_class')
    df_accident_encoding_8 = one_hot_encoding(df_accident_encoding_7 , 'road_type')
    df_accident_encoding_9 = one_hot_encoding(df_accident_encoding_8 , 'junction_detail')
    df_accident_encoding_10 = one_hot_encoding(df_accident_encoding_9 , 'junction_control')
    df_accident_encoding_11 = one_hot_encoding(df_accident_encoding_10 , 'second_road_class')
    df_accident_encoding_12 = one_hot_encoding(df_accident_encoding_11 , 'pedestrian_crossing_human_control')
    df_accident_encoding_13 = one_hot_encoding(df_accident_encoding_12 , 'pedestrian_crossing_physical_facilities')
    df_accident_encoding_14 = one_hot_encoding(df_accident_encoding_13 , 'light_conditions')
    df_accident_encoding_15 = one_hot_encoding(df_accident_encoding_14 , 'weather_conditions')
    df_accident_encoding_16 = one_hot_encoding(df_accident_encoding_15 , 'road_surface_conditions')
    df_accident_encoding_17 = one_hot_encoding(df_accident_encoding_16 , 'urban_or_rural_area')
    df_accident_encoding_18 = one_hot_encoding(df_accident_encoding_17 , 'did_police_officer_attend_scene_of_accident')
    df_accident_encoding_19 = one_hot_encoding(df_accident_encoding_18 , 'trunk_road_flag')
    df_accident_encoding_20 = encode_lsoa_of_accident_location(df_accident_encoding_19)
    df_accident_encoding_temp = df_accident_encoding_19.copy()
    df_accident_encoding_temp['lsoa_of_accident_location_E_W'] = df_accident_encoding_19['lsoa_of_accident_location'].astype(str)
    df_accident_encoding_temp['lsoa_of_accident_location_E_W'] = df_accident_encoding_19['lsoa_of_accident_location'].apply(lambda x : x[0])
    temp_df_5 = merge_two_columns(temp_df_4 ,df_accident_encoding_temp , df_accident_encoding_20 , 'lsoa_of_accident_location_E_W')
    temp_df_5.to_csv('/opt/airflow/data/lookup_table.csv')
    return df_accident_encoding_20




############################################################################################################


## Normalization and Standardization 

#### General Functions

#create method that takes dataframe and column and apply min max scaling on the column and return the dataframe
def min_max_scaling(df , column):
    temp_df = df.copy()
    temp_df[column] = (temp_df[column] - temp_df[column].min()) / (temp_df[column].max() - temp_df[column].min())
    return temp_df

#create method that takes dataframe and column and apply standard scaling on the column and return the dataframe
def standard_scaling(df , column):
    temp_df = df.copy()
    temp_df[column] = (temp_df[column] - temp_df[column].mean()) / (temp_df[column].std())
    return temp_df

#create method that takes two dataframes and column and plot column distribution before and after min max scaling 
def plot_min_max_scaling(df1 , df2 , column):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.distplot(df1[column])
    plt.title('Before Min Max Scaling')
    plt.subplot(1, 2, 2)
    sns.distplot(df2[column])
    plt.title('After Min Max Scaling')
    plt.show()

#create method that takes two dataframes and column and plot column distribution before and after standard scaling
def plot_standard_scaling(df1 , df2 , column):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.distplot(df1[column])
    plt.title('Before Standard Scaling')
    plt.subplot(1, 2, 2)
    sns.distplot(df2[column])
    plt.title('After Standard Scaling')
    plt.show()


### Normalization and Standardization for the accident dataset

def normalization_standardization(df_accident):
    df = df_accident.copy()
    df_accident_scaling = min_max_scaling(df , 'location_easting_osgr')
    df_accident_scaling_1 = min_max_scaling(df_accident_scaling , 'location_northing_osgr')
    df_accident_scaling_2 = min_max_scaling(df_accident_scaling_1 , 'longitude')
    df_accident_scaling_3 = min_max_scaling(df_accident_scaling_2 , 'latitude')
    df_accident_scaling_4 = standard_scaling(df_accident_scaling_3 , 'local_authority_district')
    df_accident_scaling_5 = standard_scaling(df_accident_scaling_4 , 'local_authority_ons_district')
    df_accident_scaling_6 = standard_scaling(df_accident_scaling_5 , 'local_authority_highway')
    df_accident_scaling_7 = standard_scaling(df_accident_scaling_6 , 'first_road_number')
    df_accident_scaling_8 = standard_scaling(df_accident_scaling_7 , 'second_road_number')
    df_accident_scaling_9 = standard_scaling(df_accident_scaling_8 , 'speed_limit')
    df_accident_scaling_10 = standard_scaling(df_accident_scaling_9 , 'week_number')
    df_accident_scaling_11 = standard_scaling(df_accident_scaling_10 , 'month')
    df_accident_scaling_12 = standard_scaling(df_accident_scaling_11 , 'day')
    df_accident_scaling_13 = standard_scaling(df_accident_scaling_12 , 'hour')
    df_accident_scaling_14 = standard_scaling(df_accident_scaling_13 , 'minute')
    df_accident_scaling_15 = min_max_scaling(df_accident_scaling_14 , 'lsoa_of_accident_location_number')
    return df_accident_scaling_15


############################################################################################################


## Main Function for Cleaning and Transforming the Dataset Milestone 1
def clean_and_transform(filename):
    # read csv file as parquet file 
    na_vals = ['NA', 'Missing','None']
    df = pd.read_parquet(filename ,engine='pyarrow')
    # convert the parquet file to pandas dataframe
    df_accident = pd.DataFrame(df)
    # replace the values that are in the na_vals list with NaN
    df_accident.replace(na_vals, np.nan, inplace=True)
    # make the accident_index as the index of the dataframe
    df_accident.set_index('accident_index', inplace=True)
    process_1 = handling_missing_values(df_accident)
    process_2 = handling_outliers(process_1)
    process_3 = handling_unclean_data(process_2)
    process_3.to_csv('/opt/airflow/data/accident_cleaned_stage0.csv')
    process_4 = categorical_encoding(process_3)
    process_5 = normalization_standardization(process_4)
    process_5.to_csv('/opt/airflow/data/accident_cleaned_stage1.csv')

    


