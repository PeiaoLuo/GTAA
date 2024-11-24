import pandas as pd
import numpy as np

def load_data() -> list:
    china_df = pd.read_csv("data/data_by_country/China_new.csv",index_col=0)
    china_df.index = pd.to_datetime(china_df.index)

    us_df = pd.read_csv("data/data_by_country/US_new.csv",index_col=0)
    us_df.index = pd.to_datetime(us_df.index)

    uk_df = pd.read_csv("data/data_by_country/UK_new.csv",index_col=0)
    uk_df.index = pd.to_datetime(uk_df.index)

    german_df = pd.read_csv("data/data_by_country/German_new.csv",index_col=0)
    german_df.iloc[:,-2].fillna(0,inplace=True)
    german_df.index = pd.to_datetime(german_df.index)

    japan_df = pd.read_csv("data/data_by_country/Japan_new.csv",index_col=0)
    japan_df.iloc[:,-2].fillna(0,inplace=True)
    japan_df.index = pd.to_datetime(japan_df.index)

    others_df = pd.read_csv("data/data_by_country/Others_new.csv",index_col=0)
    others_df.index = pd.to_datetime(others_df.index)

    df_ls = [china_df, us_df, uk_df, german_df, japan_df, others_df]
    # here's to make the dfs into form required
    for df in df_ls:
        df.index.name = None
    
    return df_ls

def rename_cols(df_ls, asset_dict, country_ls, country_dict) -> None:
#change the name of columns
    for i, df in enumerate(df_ls):
        if i != 5:
            country_name = country_ls[i]
            for j, col in enumerate(df.columns):
                for key, value in asset_dict.items():
                    if key in col:
                        df.rename(columns={col: country_name + '_' + value}, inplace=True)
                        break
        else:
            for j, col in enumerate(df.columns):
                asset_name = None
                country_name = None
                for key, value in asset_dict.items():
                    if key in col:
                        asset_name = value
                        break
                for key, value in country_dict.items():
                    if key in col:
                        country_name = value
                        break
                if asset_name is not None and country_name is not None:
                    df.rename(columns={col: country_name + '_' + asset_name}, inplace=True)

def set_range_and_type(df_ls) -> None:
    for df in df_ls:
        #drop values before 2005-1-1
        df.drop(index=df.index[0:60],inplace=True)
        #ffillna for PPI Exchange_rate and interest_rate
        df.iloc[:, -3:] = df.iloc[:, -3:].ffill()

    #trun into float type
    for df in df_ls:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].str.replace(',', '')
            df[col] = df[col].astype(float)
            
def seperate_df(df_ls, country_ls) -> dict:
    sep_ls = country_ls + ['Others']
    asset_info_dict = {}
    for i,area in enumerate(sep_ls):
        temp_ls = []
        if area != 'Others':
            # seperate into col + PPI + Exchange + Interest
            for j in range(len(df_ls[i].columns)-3):
                temp_df = df_ls[i].iloc[:,[j,-3,-2,-1]].copy()
                temp_df.dropna(how='any',inplace=True)
                temp_df.replace([np.inf, -np.inf], 0.0, inplace=True)
                temp_ls.append(temp_df)
        else:
            # seperate into single col
            for j in range(len(df_ls[i].columns)):
                temp_df = df_ls[i].iloc[:,j].copy()
                temp_df.dropna(how='any',inplace=True)
                temp_df.replace([np.inf, -np.inf], 0.0, inplace=True)
                temp_ls.append(temp_df)
        asset_info_dict[area] = temp_ls
    return asset_info_dict

def order_by_sep_dict(asset_info_dict) -> pd.DataFrame:
    ordered_df = asset_info_dict['China'][0].iloc[:,0]

    for asset_df_name,value in asset_info_dict.items():

        for i,testdt in enumerate(value):
            if asset_df_name == 'China' and i==0 :
                continue
            
            if asset_df_name == 'Others':
                ordered_df = pd.concat([ordered_df,testdt],axis=1)
            else:
                ordered_df = pd.concat([ordered_df,testdt.iloc[:,0]],axis=1)
    return ordered_df
