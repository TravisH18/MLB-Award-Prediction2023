import pandas as pd

bad_columns = ['Rk', 'Name', 'Age', 'Lg', 'Tm', 'Pos Summary','Name-additional']
df2018 = pd.read_csv('2018 MLB Player Stats - Batting.csv')
df2018['MVP'] = 0
mvp_names2018 = ['Christian Yelich*', 'Mookie Betts']
df2018.loc[df2018['Name'].isin(mvp_names2018), 'MVP'] = 1

#before saving get rid of Rk, Name, Age, Lg, Tm, Pos Summary,Name-additional
df2018.drop(bad_columns, axis=1)
df2018.to_csv('2018_batting.csv', index=False)

df2019 = pd.read_csv('2019 MLB Player Stats - Batting.csv')
df2019['MVP'] = 0
mvp_names2019 = ['Mike Trout', 'Cody Bellinger*']
df2019.loc[df2019['Name'].isin(mvp_names2019), 'MVP'] = 1
df2019.drop(bad_columns, axis=1)
df2019.to_csv('2019_batting.csv', index=False)

df2020 = pd.read_csv('2020 MLB Player Stats - Batting.csv')
df2020['MVP'] = 0
mvp_names2020 = ['José Abreu', 'Freddie Freeman*']
df2020.loc[df2020['Name'].isin(mvp_names2020), 'MVP'] = 1
df2020.drop(bad_columns, axis=1)
df2020.to_csv('2020_batting.csv', index=False)

df2021 = pd.read_csv('2021 MLB Player Stats - Batting.csv')
df2021['MVP'] = 0
mvp_names2021 = ['Shohei Ohtani*', 'Bryce Harper*']
df2021.loc[df2021['Name'].isin(mvp_names2021), 'MVP'] = 1
df2021.drop(bad_columns, axis=1)
df2021.to_csv('2021_batting.csv', index=False)

df2022 = pd.read_csv('2022 MLB Player Stats - Batting.csv')
df2022['MVP'] = 0
mvp_names2022 = ['Aaron Judge', 'Paul Goldschmidt']
df2022.loc[df2022['Name'].isin(mvp_names2022), 'MVP'] = 1
df2022.drop(bad_columns, axis=1)
df2022.to_csv('2022_batting.csv', index=False)