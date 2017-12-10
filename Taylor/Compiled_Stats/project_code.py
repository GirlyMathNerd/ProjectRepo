from setup import *
import glob
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from pandas.tools.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn import preprocessing
from random import randint

ls = glob.glob('*.csv')
df_tmp = pd.read_csv('_LeagueofLegends.csv')
df_temp = df_tmp[['blueTopChamp','blueJungleChamp','blueMiddleChamp','blueADCChamp',
            'blueSupportChamp','redTopChamp','redJungleChamp','redMiddleChamp','redADCChamp',
            'redSupportChamp']]
list_champions = df_temp.stack().unique()
len(list_champions)
df_tmp['Game'] = df_tmp['MatchHistory'].str.replace(r'http://.*/.*/*/#match-details/.*/','')
df_tmp['Game'] = df_tmp['Game'].str.replace(r'\?.*','')
df_tmp['Game'] = df_tmp['Game'].astype(int)

df = pd.read_csv('_LeagueofLegends.csv')
df['Game'] = df['MatchHistory'].str.replace(r'http://.*/.*/*/#match-details/.*/','')
df['Game'] = df['Game'].str.replace(r'\?.*','')
df['Game'] = df['Game'].astype(int)
df = df.drop_duplicates('Game',keep='first')
blue_df = df[['blueTopChamp','blueJungleChamp','blueMiddleChamp','blueADCChamp','blueSupportChamp']]
blue_df.columns = ['Top','Jungle','Mid','ADC','Support']
red_df = df[['redTopChamp','redJungleChamp','redMiddleChamp','redADCChamp','redSupportChamp']]
red_df.columns = ['Top','Jungle','Mid','ADC','Support']
red_blue_df = pd.concat([red_df,blue_df],axis=0).reset_index(drop=True)
red_blue_df = red_blue_df.apply(lambda g: g.str.lower())

top_count = red_blue_df.groupby('Top').apply(lambda g: g.count())['Top'].to_frame()
jun_count = red_blue_df.groupby('Jungle').apply(lambda g: g.count())['Jungle'].to_frame()
mid_count = red_blue_df.groupby('Mid').apply(lambda g: g.count())['Mid'].to_frame()
adc_count = red_blue_df.groupby('ADC').apply(lambda g: g.count())['ADC'].to_frame()
sup_count = red_blue_df.groupby('Support').apply(lambda g: g.count())['Support'].to_frame()
top_count.index.names=['Champion']
jun_count.index.names=['Champion']
mid_count.index.names=['Champion']
adc_count.index.names=['Champion']
sup_count.index.names=['Champion']
counts = pd.merge(top_count,jun_count,left_index=True,right_index=True,how='outer')
counts = pd.merge(counts,mid_count,left_index=True,right_index=True,how='outer')
counts = pd.merge(counts,adc_count,left_index=True,right_index=True,how='outer')
counts = pd.merge(counts,sup_count,left_index=True,right_index=True,how='outer')
counts.fillna(0).astype('int').to_html('hero_counts.html')

df_assists = pd.read_csv(ls[0])
df_assists.Game = df_assists['Game'].astype(int)
df_assists = df_assists.sort_values('Game')
df_assists = df_assists.drop_duplicates('Game',keep='first')
merged = pd.merge(df_tmp, df_assists, on='Game',how='right')
merged = merged[['Game','blueTopChamp','blueJungleChamp','blueMiddleChamp','blueADCChamp',
            'blueSupportChamp','redTopChamp','redJungleChamp','redMiddleChamp','redADCChamp',
            'redSupportChamp','Champion_1','Champion_2','Champion_3','Champion_4',
                'Champion_5','Champion_6','Champion_7','Champion_8','Champion_9',
                'Champion_10','gamelength']].sort_values('Game').reset_index()
merged.drop('index',axis=1,inplace=True)
merged = merged.drop_duplicates('Game',keep='first')
merged_champs = merged[['blueTopChamp','blueJungleChamp','blueMiddleChamp','blueADCChamp',
            'blueSupportChamp','redTopChamp','redJungleChamp','redMiddleChamp','redADCChamp',
            'redSupportChamp']]
merged_champs = merged_champs.apply(lambda x: x.astype(str).str.lower())
df_merged = pd.DataFrame(merged_champs.stack())
df_time = merged['gamelength']
df_time_length = pd.concat([df_time]*10,axis=1).stack().reset_index()
df_time_length_col = df_time_length[0]

df_assists.drop('Game',axis=1,inplace=True)
df_assists_stacked = df_assists.stack()

df_dmg_healed = pd.read_csv(ls[2])
df_dmg_healed = df_dmg_healed.sort_values('Game')
df_dmg_healed = df_dmg_healed.drop_duplicates('Game',keep='first')
df_dmg_healed.drop('Game',axis=1,inplace=True)
df_dmg_healed_stacked = df_dmg_healed.stack()
df_dmg_taken = pd.read_csv(ls[3])
df_dmg_taken = df_dmg_taken.sort_values('Game')
df_dmg_taken = df_dmg_taken.drop_duplicates('Game',keep='first')
df_dmg_taken.drop('Game',axis=1,inplace=True)
df_dmg_taken_stacked = df_dmg_taken.stack()
df_kills = pd.read_csv(ls[8])
df_kills = df_kills.sort_values('Game')
df_kills = df_kills.drop_duplicates('Game',keep='first')
df_kills.drop('Game',axis=1,inplace=True)
df_kills_stacked = df_kills.stack()
df_magic_dmg_overall = pd.read_csv(ls[12])
df_magic_dmg_overall = df_magic_dmg_overall.sort_values('Game')
df_magic_dmg_overall = df_magic_dmg_overall.drop_duplicates('Game',keep='first')
df_magic_dmg_overall.drop('Game',axis=1,inplace=True)
df_magic_dmg_overall_stacked = df_magic_dmg_overall.stack()
df_magic_dmg_champs = pd.read_csv(ls[14])
df_magic_dmg_champs = df_magic_dmg_champs.sort_values('Game')
df_magic_dmg_champs = df_magic_dmg_champs.drop_duplicates('Game',keep='first')
df_magic_dmg_champs.drop('Game',axis=1,inplace=True)
df_magic_dmg_champs_stacked = df_magic_dmg_champs.stack()
df_minions_killed = pd.read_csv(ls[15])
df_minions_killed = df_minions_killed.sort_values('Game')
df_minions_killed = df_minions_killed.drop_duplicates('Game',keep='first')
df_minions_killed.drop('Game',axis=1,inplace=True)
df_minions_killed_stacked = df_minions_killed.stack()
df_neutral_minions = pd.read_csv(ls[16])
df_neutral_minions = df_neutral_minions.sort_values('Game')
df_neutral_minions = df_neutral_minions.drop_duplicates('Game',keep='first')
df_neutral_minions.drop('Game',axis=1,inplace=True)
df_neutral_minions_stacked = df_neutral_minions.stack()
df_phys_dmg_overall = pd.read_csv(ls[19])
df_phys_dmg_overall = df_phys_dmg_overall.sort_values('Game')
df_phys_dmg_overall = df_phys_dmg_overall.drop_duplicates('Game',keep='first')
df_phys_dmg_overall.drop('Game',axis=1,inplace=True)
df_phys_dmg_overall_stacked = df_phys_dmg_overall.stack()
df_phys_dmg_champs = pd.read_csv(ls[21])
df_phys_dmg_champs = df_phys_dmg_champs.sort_values('Game')
df_phys_dmg_champs = df_phys_dmg_champs.drop_duplicates('Game',keep='first')
df_phys_dmg_champs.drop('Game',axis=1,inplace=True)
df_phys_dmg_champs_stacked = df_phys_dmg_champs.stack()
df_tot_dmg_overall = pd.read_csv(ls[23])
df_tot_dmg_overall = df_tot_dmg_overall.sort_values('Game')
df_tot_dmg_overall = df_tot_dmg_overall.drop_duplicates('Game',keep='first')
df_tot_dmg_overall.drop('Game',axis=1,inplace=True)
df_tot_dmg_overall_stacked = df_tot_dmg_overall.stack()
df_tot_dmg_champs = pd.read_csv(ls[24])
df_tot_dmg_champs = df_tot_dmg_champs.sort_values('Game')
df_tot_dmg_champs = df_tot_dmg_champs.drop_duplicates('Game',keep='first')
df_tot_dmg_champs.drop('Game',axis=1,inplace=True)
df_tot_dmg_champs_stacked = df_tot_dmg_champs.stack()
df_wards_placed = pd.read_csv(ls[31])
df_wards_placed = df_wards_placed.sort_values('Game')
df_wards_placed = df_wards_placed.drop_duplicates('Game',keep='first')
df_wards_placed.drop('Game',axis=1,inplace=True)
df_wards_placed_stacked = df_wards_placed.stack()

df_full = pd.DataFrame({'Assists':df_assists_stacked,'Damage_Healed':df_dmg_healed_stacked,
                       'Damage_Taken':df_dmg_taken_stacked,'Kills':df_kills_stacked,
                       'Magic_Dmg_Overall':df_magic_dmg_overall_stacked,'Magic_Dmg_Champs':df_magic_dmg_champs_stacked,
                       'Minions_Killed':df_minions_killed_stacked,'Neutral_Minions':df_neutral_minions_stacked,
                       'Phys_Dmg_Overall':df_phys_dmg_overall_stacked,'Phys_Dmg_Champs':df_phys_dmg_champs_stacked,
                       'Total_Damage':df_tot_dmg_overall_stacked,'Total_Damage_Champs':df_tot_dmg_champs_stacked,
                       'Wards_Placed':df_wards_placed_stacked})
df_full.replace('-','0',inplace=True)
df_full.Damage_Healed = df_full.Damage_Healed.map(lambda x: x.rstrip('kmil '))
df_full.Damage_Taken = df_full.Damage_Taken.map(lambda x: x.rstrip('kmil '))
df_full.Magic_Dmg_Champs = df_full.Magic_Dmg_Champs.map(lambda x: x.rstrip('kmil '))
df_full.Magic_Dmg_Overall = df_full.Magic_Dmg_Overall.map(lambda x: x.rstrip('kmil '))
df_full.Phys_Dmg_Champs = df_full.Phys_Dmg_Champs.map(lambda x: x.rstrip('kmil '))
df_full.Phys_Dmg_Overall = df_full.Phys_Dmg_Overall.map(lambda x: x.rstrip('kmil '))
df_full.Total_Damage = df_full.Total_Damage.map(lambda x: x.rstrip('kmil '))
df_full.Total_Damage_Champs = df_full.Total_Damage_Champs.map(lambda x: x.rstrip('kmil '))
df_full.Assists = df_full.Assists.astype(float)
df_full.Kills = df_full.Kills.astype(float)
df_full.Minions_Killed = df_full.Minions_Killed.astype(float)
df_full.Neutral_Minions = df_full.Neutral_Minions.astype(float)
df_full.Wards_Placed = df_full.Wards_Placed.astype(float)
df_full.Damage_Healed = df_full.Damage_Healed.astype(float)
df_full.Damage_Taken = df_full.Damage_Taken.astype(float)
df_full.Magic_Dmg_Champs = df_full.Magic_Dmg_Champs.astype(float)
df_full.Magic_Dmg_Overall = df_full.Magic_Dmg_Overall.astype(float)
df_full.Phys_Dmg_Champs = df_full.Phys_Dmg_Champs.astype(float)
df_full.Phys_Dmg_Overall = df_full.Phys_Dmg_Overall.astype(float)
df_full.Total_Damage = df_full.Total_Damage.astype(float)
df_full.Total_Damage_Champs = df_full.Total_Damage_Champs.astype(float)
df_full.Damage_Healed = df_full.Damage_Healed * 1000
df_full.Damage_Taken = df_full.Damage_Taken * 1000
df_full.Magic_Dmg_Champs = df_full.Magic_Dmg_Champs * 1000
df_full.Magic_Dmg_Overall = df_full.Magic_Dmg_Overall * 1000
df_full.Phys_Dmg_Champs = df_full.Phys_Dmg_Champs * 1000
df_full.Phys_Dmg_Overall = df_full.Phys_Dmg_Overall * 1000
df_full.Total_Damage = df_full.Total_Damage * 1000
df_full.Total_Damage_Champs = df_full.Total_Damage_Champs * 1000

min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(df_full)
df_normalized = pd.DataFrame(np_scaled)

pca_orig = PCA().fit(df_normalized)

pca = PCA(8)
projected_normalized = pca.fit_transform(df_normalized)
df_normalized_projected = pd.DataFrame(projected_normalized)

gmm = GaussianMixture(n_components=12,random_state=42).fit(df_normalized_projected)
labels = gmm.predict(df_normalized_projected)

df_labels_normalized_gmm = pd.DataFrame(labels)

df_merged_new = df_merged.reset_index()
df_merged_new['Label'] = df_labels_normalized_gmm
df_merged_new.columns = ['Match_Number','Champion_Role','Champion','Label']
df_merged_final_count = df_merged_new.groupby(['Champion','Label']).count()
df_final = df_merged_final_count['Match_Number'].unstack().fillna(value=0)
df_final['Sum'] = df_final.sum(axis=1)
df_final_tmp = df_final['Sum']
df_final = df_final.apply(lambda g: (g/df_final['Sum'])*100)
df_final['Total_Count'] = df_final_tmp
df_final.to_html('frame.html')

# This cell is just for cleaning up the frame for the report
df_final_new = df_final
df_final_new.columns=['Glass_Cannon_Magic', 'Sustained_Magic_Dmg','Magic_Dmg_Jungle','Bruisers',
                     'Magic_Dmg_Support', 'Physical_Dmg_Jungle_Close', 'Tanky_Support', 'Magic_Dmg_Close',
                     'Hybrid_Dmg', 'Hybrid_Jungle', 'Iffy', 'ADC','Sum','Total_Count']
df_final_new.to_html('Final_Frame.html')

lowest_bic = np.infty
bic = []
n_components_range = range(11, 17)
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        # Fit a Gaussian mixture with EM
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type)
        gmm.fit(df_normalized_projected)
        bic.append(gmm.bic(df_normalized_projected))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
labels_best = best_gmm.predict(df_normalized_projected)

df_labels_normalized_gmm_bic = pd.DataFrame(labels_best)

df_merged_new = df_merged.reset_index()
df_merged_new['Label'] = df_labels_normalized_gmm
df_merged_new.columns = ['Match_Number','Champion_Role','Champion','Label']
df_merged_final_count = df_merged_new.groupby(['Champion','Label']).count()
df_final = df_merged_final_count['Match_Number'].unstack().fillna(value=0)
df_final['Sum'] = df_final.sum(axis=1)
df_final_tmp = df_final['Sum']
df_final = df_final.apply(lambda g: (g/df_final['Sum'])*100)
df_final['Total_Count'] = df_final_tmp
df_final.to_html('opt_frame.html')

df_time_normalized = pd.DataFrame({'Assists':df_full['Assists'],'Damage_Healed':df_full['Damage_Healed'],
                       'Damage_Taken':df_full['Damage_Taken'],'Kills':df_full['Kills'],
                       'Magic_Dmg_Overall':df_full['Magic_Dmg_Overall'],'Magic_Dmg_Champs':df_full['Magic_Dmg_Champs'],
                       'Minions_Killed':df_full['Minions_Killed'],'Neutral_Minions':df_full['Neutral_Minions'],
                       'Phys_Dmg_Overall':df_full['Phys_Dmg_Overall'],'Phys_Dmg_Champs':df_full['Phys_Dmg_Champs'],
                       'Total_Damage':df_full['Total_Damage'],'Total_Damage_Champs':df_full['Total_Damage_Champs'],
                       'Wards_Placed':df_full['Wards_Placed']})
df_time_length = df_time_length.set_index(df_time_normalized.index)
df_time_normalized['Game_Length'] = df_time_length[0]
df_time_normalized = df_time_normalized.apply(lambda g: g/df_time_normalized['Game_Length'])
df_time_normalized_final = df_time_normalized.drop('Game_Length',axis=1)

np_scaled_time = min_max_scaler.fit_transform(df_time_normalized_final)
df_normalized_time = pd.DataFrame(np_scaled_time)

projected_normalized_time = pca.fit_transform(df_normalized_time)
df_normalized_projected_time = pd.DataFrame(projected_normalized_time)

gmm_time = GaussianMixture(n_components=12,random_state=42).fit(df_normalized_projected_time)
labels_time = gmm_time.predict(df_normalized_projected_time)

df_labels_normalized_gmm_time = pd.DataFrame(labels_time)

df_merged_new_time = df_merged.reset_index()
df_merged_new_time['Label'] = df_labels_normalized_gmm_time
df_merged_new_time.columns = ['Match_Number','Champion_Role','Champion','Label']
df_merged_new_time
df_merged_final_count_time = df_merged_new_time.groupby(['Champion','Label']).count()
df_merged_final_count_time
df_final_time = df_merged_final_count_time['Match_Number'].unstack().fillna(value=0)
df_final_time['Sum'] = df_final_time.sum(axis=1)
df_final_tmp_time = df_final_time['Sum']
df_final_time = df_final_time.apply(lambda g: (g/df_final_time['Sum'])*100)
df_final_time['Total_Count'] = df_final_tmp_time
df_final_time.to_html('frame_time.html')

from sklearn.cluster import KMeans
k_means = KMeans(n_clusters=12,random_state=42).fit(df_normalized_projected)
k_labels = k_means.predict(df_normalized_projected)

df_merged_new = df_merged.reset_index()
df_merged_new['Label'] = k_labels
df_merged_new.columns = ['Match_Number','Champion_Role','Champion','Label']
df_merged_final_count = df_merged_new.groupby(['Champion','Label']).count()
df_final = df_merged_final_count['Match_Number'].unstack().fillna(value=0)
df_final['Max'] = df_final.max(axis=1)
df_final_tmp = df_final['Max']
df_final = df_final.apply(lambda g: (g/df_final['Max']))
df_final['Total_Count'] = df_final_tmp
df_final.to_html('K-frame.html')