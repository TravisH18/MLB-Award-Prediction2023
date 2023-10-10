import pandas as pd

bad_columns = ['Rk', 'Age', 'Lg', 'Tm', 'Pos Summary','Name']
keep = ['PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'BA', 'OBP','SLG','OPS','OPS+','TB','GDP','HBP','SH','SF','IBB', 'Name-additional']
col_types = {'PA': int, 'AB': int, 'R': int, 'H': int, '2B': int, '3B': int, 'HR': int, 'RBI': int, 'SB': int, 'CS': int, 'BB': int, 'SO': int, 'BA': float, 'OBP': float,'SLG': float,'OPS': float,'OPS+': float,'TB': float,'GDP': int,'HBP': int,'SH': int,'SF': int,'IBB': int, 'Name-additional': str}
df2018 = pd.read_csv('2018 MLB Player Stats - Batting.csv', usecols=keep, dtype=col_types, index_col=False)
print(df2018.head)
df2018['MVP'] = 0
mvp_names2018 = ['yelicch01', 'bettsmo01']
df2018.loc[df2018['Name-additional'].isin(mvp_names2018), 'MVP'] = 1

#before saving get rid of Rk, Name, Age, Lg, Tm, Pos Summary,Name-additional
#df2018.drop(bad_columns, axis=1)
df2018.to_csv('2018_batting.csv', index=False)

df2019 = pd.read_csv('2019 MLB Player Stats - Batting.csv', usecols=keep, dtype=col_types, index_col=False)
df2019['MVP'] = 0
mvp_names2019 = ['troutmi01', 'bellico01']
df2019.loc[df2019['Name-additional'].isin(mvp_names2019), 'MVP'] = 1
#df2019.drop(bad_columns, axis=1)
df2019.to_csv('2019_batting.csv', index=False)

df2020 = pd.read_csv('2020 MLB Player Stats - Batting.csv', usecols=keep, dtype=col_types, index_col=False)
df2020['MVP'] = 0
mvp_names2020 = ['abreujo02', 'freemfr01']
df2020.loc[df2020['Name-additional'].isin(mvp_names2020), 'MVP'] = 1
#df2020.drop(bad_columns, axis=1)
df2020.to_csv('2020_batting.csv', index=False)

df2021 = pd.read_csv('2021 MLB Player Stats - Batting.csv', usecols=keep, dtype=col_types, index_col=False)
df2021['MVP'] = 0
mvp_names2021 = ['ohtansh01', 'harpebr03']
df2021.loc[df2021['Name-additional'].isin(mvp_names2021), 'MVP'] = 1
#df2021.drop(bad_columns, axis=1)
df2021.to_csv('2021_batting.csv', index=False)

df2022 = pd.read_csv('2022 MLB Player Stats - Batting.csv', usecols=keep, dtype=col_types, index_col=False)
df2022['MVP'] = 0
mvp_names2022 = ['judgeaa01', 'goldspa01']
df2022.loc[df2022['Name-additional'].isin(mvp_names2022), 'MVP'] = 1
#df2022.drop(bad_columns, axis=1)
df2022.to_csv('2022_batting.csv', index=False)