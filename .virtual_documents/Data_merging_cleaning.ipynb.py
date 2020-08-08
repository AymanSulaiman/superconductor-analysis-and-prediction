import pandas as pd
import os


train_df = pd.read_csv(os.path.join('data','train.csv')) # Left
unique_m_df = pd.read_csv(os.path.join('data','unique_m.csv')) # Right


train_df.to_csv('train.csv', index=True)


unique_m_df.to_csv('unique_m.csv', index=True)


train_indexed_df = pd.read_csv(os.path.join('train.csv')) # Left
unique_indexed_m_df = pd.read_csv(os.path.join('unique_m.csv')) # Right


train_indexed_df


merged_df = pd.merge(train_indexed_df, unique_indexed_m_df, how='left', on=['Unnamed: 0'])
merged_df


merged_df.drop_duplicates()
merged_df_1 = merged_df.drop(['critical_temp_y'], axis=1)
merged_df_1


merged_df_1.rename(columns={'critical_temp_x':'critical_temp'}, inplace=True)
merged_df_1


merged_df_1 = merged_df_1.drop(['Unnamed: 0'], axis=1)
merged_df_1





merged_df_1.to_csv('merged.csv', index=False)



