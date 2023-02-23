# import libraries
import pandas as pd


#apply lable encoding to nearest hospital column
def label_encoding(df_accident_cleaned,feature):
    df_accident_cleaned[feature] = df_accident_cleaned[feature].astype('category')
    df_accident_cleaned[feature] = df_accident_cleaned[feature].cat.codes
    return df_accident_cleaned


# This is the main function for milestone 2
def add_feature(computed_dataset,cleaned_dataset):
    loaded_data = pd.read_csv(computed_dataset)
    df_accident_cleaned = pd.read_csv(cleaned_dataset)
    df_accident_cleaned = pd.concat([df_accident_cleaned,loaded_data['nearest_hospital']],axis=1)
    df = label_encoding(df_accident_cleaned,'nearest_hospital')
    df.to_csv('/opt/airflow/data/accident_cleaned_stage2.csv',index=False)