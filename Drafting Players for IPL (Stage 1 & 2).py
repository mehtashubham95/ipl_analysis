# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 23:42:20 2022

@author: mehta
"""

import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import model_selection
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit,cross_val_score
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from scipy.stats.mstats import gmean
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score, recall_score, precision_score, average_precision_score, f1_score, classification_report, accuracy_score, plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import brier_score_loss
from sklearn.metrics import precision_score
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from sklearn.multioutput import MultiOutputRegressor

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

def rmspe(y_true, y_pred):
    '''
    Compute Root Mean Square Percentage Error between two arrays.
    '''
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / y_true)), axis=0))

    return loss

""" Reading Data file for Ball by Ball data (Data1), Player Characteristics Data, Salary Data(Data2), and Player name lookup data to match with Salary Data """

main_data = pd.read_csv('ipl_08_22_bob_data.csv')
pl_char = pd.read_csv('IPL_Player_Char.csv')
sal_data = pd.read_csv('ipl_player_salary.csv')
player_lu = pd.read_csv('Dissertation\sal_player_lu.csv')
sf_flag = pd.read_csv('IPL Data\semi_final_flag.csv')

md_copy = main_data.copy(deep=True)

''' Aggregating Data for Stage 1 of Analysis - Player Evaluation '''

player_stats_bat = md_copy[md_copy["wide_runs"] == 0].groupby(["Strike Batsman Lookup Name","Season","Batsman Type","Batsman Role","Batting team"]).agg(
        runs = ('batsman_runs','sum'),
        balls = ('batsman','count'),
        innings = ('match_id','nunique')).reset_index()

player_stats_bat_out = md_copy[md_copy["player_dismissed"].notnull()].groupby(["Strike Batsman Lookup Name","Season","Batsman Type","Batsman Role","Batting team"]).agg(
        out = ('player_dismissed','count')).reset_index()

player_stats_bat_dots = md_copy[md_copy["batsman_runs"] == 0].groupby(["Strike Batsman Lookup Name","Season","Batsman Type","Batsman Role","Batting team"]).agg(
        bat_dots = ('batsman','count')).reset_index()

player_stats_bat_avg = player_stats_bat[~player_stats_bat['Batsman Role'].str.contains(pat = 'Bowler', na = False)].groupby(['Season']).agg(
        runs = ("runs","sum"),
        inns = ("innings","sum")).reset_index()

player_stats_bat_avg['avg'] = player_stats_bat_avg['runs']/player_stats_bat_avg['inns']

player_stats_bat = pd.merge(player_stats_bat,player_stats_bat_out,how = 'left',left_on = ["Strike Batsman Lookup Name","Season","Batsman Type","Batsman Role","Batting team"],right_on = ["Strike Batsman Lookup Name","Season","Batsman Type","Batsman Role","Batting team"])
player_stats_bat = pd.merge(player_stats_bat,player_stats_bat_dots,how = 'left',left_on = ["Strike Batsman Lookup Name","Season","Batsman Type","Batsman Role","Batting team"],right_on = ["Strike Batsman Lookup Name","Season","Batsman Type","Batsman Role","Batting team"])
player_stats_bat = player_stats_bat.rename(columns = {"Strike Batsman Lookup Name":"batsman"})

player_stats_bowl = md_copy[md_copy["wide_runs"] == 0].groupby(["Bowler Lookup Name","Season","Bowler Type","Bowler Role","Bowling Team"]).agg(
        runs = ('total_runs','sum'),
        balls = ('batsman','count'),
        innings = ('match_id','nunique')).reset_index()

player_stats_bowl2 = md_copy[md_copy["wide_runs"] == 0].groupby(["Bowler Lookup Name","Season","Bowler Type","Bowler Role","Bowling Team","match_id"]).agg(
        runs = ('total_runs','sum'),
        balls = ('batsman','count')).reset_index()

player_stats_bowl2['runs_p_ball_std'] = player_stats_bowl2['runs']/player_stats_bowl2['balls']
player_stats_bowl3 = player_stats_bowl2.groupby(["Bowler Lookup Name","Season","Bowler Type","Bowler Role","Bowling Team"]).agg(
        runs_p_ball_std = ('runs_p_ball_std','std')).reset_index()
player_stats_bowl = pd.merge(player_stats_bowl,player_stats_bowl3, how = 'left', on = ["Bowler Lookup Name","Season","Bowler Type","Bowler Role","Bowling Team"])
player_stats_bowl_wickets = md_copy[md_copy["player_dismissed"].notnull()].groupby(["Bowler Lookup Name","Season","Bowler Type","Bowler Role","Bowling Team"]).agg(
        wickets = ('player_dismissed','count')).reset_index()
player_stats_bowl_dots = md_copy[md_copy["total_runs"] == 0].groupby(["Bowler Lookup Name","Season","Bowler Type","Bowler Role","Bowling Team"]).agg(
        dot_balls = ('batsman','count')).reset_index()

player_stats_bowl = pd.merge(player_stats_bowl,player_stats_bowl_wickets,how = 'left',on = ["Bowler Lookup Name","Season","Bowler Type","Bowler Role","Bowling Team"])
player_stats_bowl = pd.merge(player_stats_bowl,player_stats_bowl_dots,how = 'left',on = ["Bowler Lookup Name","Season","Bowler Type","Bowler Role","Bowling Team"])
player_stats_bowl = player_stats_bowl.rename(columns = {"Bowler Lookup Name":"bowler"})

player_stats_bowl_avg = player_stats_bowl[~player_stats_bowl['Bowler Role'].str.contains(pat = 'Batsman', na = False)].groupby(['Season']).agg(
        balls = ("balls","sum"),
        inns = ("innings","sum")).reset_index()

player_stats_bowl_avg['bl_avg'] = player_stats_bowl_avg['balls']/player_stats_bowl_avg['inns']

player_stats_overall = pd.merge(player_stats_bat,player_stats_bowl,how ='outer',left_on = ["batsman","Season"],right_on = ["bowler","Season"])
player_stats_overall = player_stats_overall.fillna(0)

player_stats_overall['bat_avg'] = player_stats_overall['runs_x']/player_stats_overall['innings_x']
player_stats_overall = pd.merge(player_stats_overall,player_stats_bat_avg[['Season','avg']], on = 'Season')
player_stats_overall['bat_raa'] = player_stats_overall['bat_avg']/player_stats_overall['avg']
player_stats_overall['bat_raa'] = player_stats_overall['bat_raa'].fillna(0)

player_stats_overall['bowl_avg'] = player_stats_overall['balls_y']/player_stats_overall['innings_y']
player_stats_overall = pd.merge(player_stats_overall,player_stats_bowl_avg[['Season','bl_avg']], on = 'Season')
player_stats_overall['bowl_raa'] = player_stats_overall['bowl_avg']/player_stats_overall['bl_avg']
player_stats_overall['bowl_avg'] = player_stats_overall['bowl_avg'].fillna(0)


''' Calculating Win-Loss Percentage Metric '''

wl_d1 = md_copy[md_copy['Season'] == 2020].groupby(["Season","match_id","Batting team"]).agg(
        Score = ('Score','max')).reset_index()

wl_d2 = wl_d1.groupby(["Season","match_id"]).agg(
        Score = ('Score','max')).reset_index()

wl_d1 = pd.merge(wl_d2,wl_d1, on = ['Season','match_id','Score'], how = 'left')

wl_d3 = md_copy[md_copy['Season'] == 2022].groupby(["Season","match_id","Batting team"]).agg(
        Score = ('Score','max')).reset_index()

wl_d4 = wl_d3.groupby(["Season","match_id"]).agg(
        Score = ('Score','max')).reset_index()

wl_d3 = pd.merge(wl_d4,wl_d3, on = ['Season','match_id','Score'], how = 'left')

fr = [wl_d1,wl_d3]
wl_d1 = pd.concat(fr)

wl_d1 = wl_d1.rename(columns = {"Batting team":"team"})
wl_d5 = pd.read_csv('D:\Dissertation\ipl_team_wins.csv') #Data with team that won each match 
wl_d1 = wl_d1[['match_id','Season','team']]
fr2 = [wl_d5,wl_d1]
wl_d5 = pd.concat(fr2)

wlp_1 = md_copy[["match_id","Season","Strike Batsman Lookup Name","Batting team"]]
wlp_2 = md_copy[["match_id","Season","Runner Lookup Name","Batting team"]]
wlp_3 = md_copy[["match_id","Season","Bowler Lookup Name","Bowling Team"]]

wlp_1 = wlp_1.rename(columns = {"Batting team":"team"})
wlp_2 = wlp_2.rename(columns = {"Batting team":"team"})
wlp_3 = wlp_3.rename(columns = {"Bowling Team":"team"})
wlp_1 = wlp_1.rename(columns = {"Strike Batsman Lookup Name":"Player2"})
wlp_2 = wlp_2.rename(columns = {"Runner Lookup Name":"Player2"})
wlp_3 = wlp_3.rename(columns = {"Bowler Lookup Name":"Player2"})

fr3 = [wlp_1,wlp_2,wlp_3]
wlp = pd.concat(fr3)
wlp=wlp.drop_duplicates()

wlp_final = pd.merge(wlp,wl_d5,on = ['match_id','Season'],how = 'left')

wlp_final['win_flag'] = 0
wlp_final['lose_flag'] = 0

wlp_final.loc[wlp_final['team_x'] == wlp_final['team_y'],'win_flag'] = 1
wlp_final.loc[wlp_final['team_x'] != wlp_final['team_y'],'lose_flag'] = 1

wlp_final2 = wlp_final.groupby(["Season","Player2"]).agg(
        wins = ('win_flag','sum'),
        loss = ('lose_flag','sum')).reset_index()

wlp_final2['wl_per'] = wlp_final2['wins']/(wlp_final2['loss'] + wlp_final2['wins'])

for i in range (0,len(player_stats_overall)):
   if (player_stats_overall['batsman'].iloc[i] == 0):
       player_stats_overall['batsman'].iloc[i] = player_stats_overall['bowler'].iloc[i]
       player_stats_overall['Batting team'].iloc[i] = player_stats_overall['Bowling Team'].iloc[i]

player_stats_overall = pd.merge(player_stats_overall,wlp_final2[["Season","Player2","wl_per"]],left_on = ['Season','batsman'], right_on = ['Season','Player2'], how = 'left')

p_2023 = player_stats_overall[player_stats_overall['Season'] == 2022]
p_2023['Season'] = 2023
frames = [player_stats_overall,p_2023]
player_stats_overall = pd.concat(frames).reset_index()


'''Calculating the cumulative stats across seasons '''

player_stats_overall=player_stats_overall.sort_values(["batsman","Season"]).reset_index()
player_stats_overall['cum_runs']=0
player_stats_overall['cum_balls']=0
player_stats_overall['cum_out']=0
player_stats_overall['cum_innings_bat']=0
player_stats_overall['cum_bat_dots']=0
player_stats_overall['cum_runs_given']=0
player_stats_overall['cum_balls_bowled']=0
player_stats_overall['cum_wickets']=0
player_stats_overall['cum_innings_bowl']=0
player_stats_overall['cum_dot_balls']=0
player_stats_overall['cum_bat_raa']=0
player_stats_overall['cum_bowl_raa']=0
player_stats_overall['cum_wl_per']=0

for i in range (1,len(player_stats_overall)):
   if(player_stats_overall['batsman'].iloc[i] == player_stats_overall['batsman'].iloc[i-1]):
       player_stats_overall['cum_runs'].iloc[i] = player_stats_overall['cum_runs'].iloc[i-1] + player_stats_overall['runs_x'][i-1]
       player_stats_overall['cum_balls'].iloc[i] = player_stats_overall['cum_balls'].iloc[i-1] + player_stats_overall['balls_x'][i-1]
       player_stats_overall['cum_out'].iloc[i] = player_stats_overall['cum_out'].iloc[i-1] + player_stats_overall['out'][i-1]
       player_stats_overall['cum_innings_bat'].iloc[i] = player_stats_overall['cum_innings_bat'].iloc[i-1] + player_stats_overall['innings_x'][i-1]
       player_stats_overall['cum_bat_dots'].iloc[i] = player_stats_overall['cum_bat_dots'].iloc[i-1] + player_stats_overall['bat_dots'][i-1]     
       player_stats_overall['cum_runs_given'].iloc[i] = player_stats_overall['cum_runs_given'].iloc[i-1] + player_stats_overall['runs_y'][i-1]
       player_stats_overall['cum_balls_bowled'].iloc[i] = player_stats_overall['cum_balls_bowled'].iloc[i-1] + player_stats_overall['balls_y'][i-1]
       player_stats_overall['cum_wickets'].iloc[i] = player_stats_overall['cum_wickets'].iloc[i-1] + player_stats_overall['wickets'][i-1]
       player_stats_overall['cum_innings_bowl'].iloc[i] = player_stats_overall['cum_innings_bowl'].iloc[i-1] + player_stats_overall['innings_y'][i-1]
       player_stats_overall['cum_dot_balls'].iloc[i] = player_stats_overall['cum_dot_balls'].iloc[i-1] + player_stats_overall['dot_balls'][i-1]      
       player_stats_overall['cum_bat_raa'].iloc[i] = (player_stats_overall['cum_innings_bat'].iloc[i-1]*player_stats_overall['cum_bat_raa'].iloc[i-1] + player_stats_overall['innings_x'][i-1]*player_stats_overall['bat_raa'][i-1])/(player_stats_overall['innings_x'][i-1]+player_stats_overall['cum_innings_bat'].iloc[i-1])  
       player_stats_overall['cum_bowl_raa'].iloc[i] = (player_stats_overall['cum_innings_bowl'].iloc[i-1]*player_stats_overall['cum_bowl_raa'].iloc[i-1] + player_stats_overall['innings_y'][i-1]*player_stats_overall['bowl_raa'][i-1])/(player_stats_overall['innings_y'][i-1]+player_stats_overall['cum_innings_bowl'].iloc[i-1])
   else:
       player_stats_overall['cum_runs'].iloc[i] = 0
       player_stats_overall['cum_balls'].iloc[i] = 0
       player_stats_overall['cum_out'].iloc[i] = 0
       player_stats_overall['cum_innings_bat'].iloc[i] = 0
       player_stats_overall['cum_bat_dots'].iloc[i] = 0    
       player_stats_overall['cum_runs_given'].iloc[i] = 0
       player_stats_overall['cum_balls_bowled'].iloc[i] = 0
       player_stats_overall['cum_wickets'].iloc[i] = 0
       player_stats_overall['cum_innings_bowl'].iloc[i] = 0
       player_stats_overall['cum_dot_balls'].iloc[i] = 0
       player_stats_overall['cum_bat_raa'].iloc[i] = 0
       player_stats_overall['cum_bowl_raa'].iloc[i] = 0
       
player_stats_overall['matches'] = player_stats_overall['cum_innings_bowl']
player_stats_overall.loc[player_stats_overall['cum_innings_bat'] >player_stats_overall['cum_innings_bowl'],'matches'] = player_stats_overall['cum_innings_bat']
player_stats_overall['matches_s'] = player_stats_overall['innings_x']
player_stats_overall.loc[player_stats_overall['innings_y'] >player_stats_overall['innings_x'],'matches_s'] = player_stats_overall['innings_y']

for i in range (1,len(player_stats_overall)):
   if(player_stats_overall['batsman'].iloc[i] == player_stats_overall['batsman'].iloc[i-1]):
       player_stats_overall['cum_wl_per'].iloc[i] = (player_stats_overall['matches'].iloc[i-1]*player_stats_overall['cum_wl_per'].iloc[i-1] + player_stats_overall['matches_s'][i-1]*player_stats_overall['wl_per'][i-1])/(player_stats_overall['matches_s'][i-1]+player_stats_overall['matches'].iloc[i-1])
   else:
       player_stats_overall['cum_wl_per'].iloc[i] = 0
   
player_stats_overall = player_stats_overall.rename(columns = {'batsman':'Player'})
player_stats_overall = player_stats_overall.rename(columns = {'Batting team':'team'})
player_stats_overall = player_stats_overall[player_stats_overall['Season'] > 2010]

player_stats_overall['cum_bat_raa'] = player_stats_overall['cum_bat_raa'].fillna(0)
player_stats_overall['cum_bowl_raa'] = player_stats_overall['cum_bowl_raa'].fillna(0)
player_stats_overall['cum_wl_per'] = player_stats_overall['cum_wl_per'].fillna(0)

player_stats_overall['bat_sr'] = player_stats_overall['cum_runs']/player_stats_overall['cum_balls']
player_stats_overall['runs_per_match'] = player_stats_overall['cum_runs']/player_stats_overall['cum_innings_bat']
player_stats_overall['wickets_per_match'] = player_stats_overall['cum_wickets']/player_stats_overall['cum_balls_bowled']
player_stats_overall['runs_conceeded_per_ball'] = player_stats_overall['cum_runs_given']/player_stats_overall['cum_balls_bowled']
player_stats_overall['bat_dots_pct'] = player_stats_overall['cum_bat_dots']/player_stats_overall['cum_balls']
player_stats_overall['bowl_dots_pct'] = player_stats_overall['cum_dot_balls']/player_stats_overall['cum_balls_bowled']
player_stats_overall = player_stats_overall.fillna(0)
player_stats_overall['cum_bowl_raa'] = player_stats_overall['cum_bowl_raa']*1.5
player_stats_overall['raa'] = 0
player_stats_overall.loc[player_stats_overall['Batsman Role'].str.contains(pat = 'Batsman', na = False),'raa'] = player_stats_overall[player_stats_overall['Batsman Role'].str.contains(pat = 'Batsman', na = False)]['cum_bat_raa']
player_stats_overall.loc[player_stats_overall['Bowler Role'].str.contains(pat = 'Bowler', na = False),'raa'] = player_stats_overall[player_stats_overall['Bowler Role'].str.contains(pat = 'Bowler', na = False)]['cum_bowl_raa']
player_stats_overall.loc[player_stats_overall['Batsman Role'].str.contains(pat = 'Batting Allrounder', na = False),'raa'] = player_stats_overall[player_stats_overall['Batsman Role'].str.contains(pat = 'Batting Allrounder', na = False)]['cum_bowl_raa']*0.4 + player_stats_overall[player_stats_overall['Batsman Role'].str.contains(pat = 'Batting Allrounder', na = False)]['cum_bat_raa']*0.6
player_stats_overall.loc[player_stats_overall['Bowler Role'].str.contains(pat = 'Bowling Allrounder', na = False),'raa'] = player_stats_overall[player_stats_overall['Bowler Role'].str.contains(pat = 'Bowling Allrounder', na = False)]['cum_bowl_raa']*0.6 + player_stats_overall[player_stats_overall['Bowler Role'].str.contains(pat = 'Bowling Allrounder', na = False)]['cum_bat_raa']*0.4
 
sf_flag = sf_flag.rename(columns = {'Team Flag':'team'})


'''Building Logistic Regression Model for Stage 1'''

team_total_p = player_stats_overall[player_stats_overall['Season'] > 2010][player_stats_overall['Season'] < 2023].groupby("team").agg(
        runs_scored = ('cum_runs','sum'),
        bat_inn = ('cum_innings_bat','sum'),
        balls_faced = ('cum_balls','sum'),
        wickets_lost = ('cum_out','sum'),
        balls_bowled = ('cum_balls_bowled','sum'),
        runs_conceeded = ('cum_runs_given','sum'),
        wickets_taken = ('cum_wickets','sum'),
        bowl_inn = ('cum_innings_bowl','sum'),
        bat_sr_std = ('bat_sr','std'),
        runs_pm_std = ('runs_per_match','std'),
        wick_pm_std = ('wickets_per_match','std'),
        runs_p_ball_std = ('runs_conceeded_per_ball','std'),
        runs_conceeded_per_ball = ('runs_conceeded_per_ball','mean'),
        raa = ('raa','mean'),
        wl_ratio = ('cum_wl_per',gmean),
        players = ('Player','nunique'),
        matches = ('matches','mean')).reset_index()

team_total_p = pd.merge(team_total_p,sf_flag,on='team')
team_total_p['bat_sr'] = team_total_p['runs_scored']/team_total_p['balls_faced']
team_total_p['runs_per_innings'] = team_total_p['runs_scored']/team_total_p['bat_inn']
team_total_p['wickets_per_innings'] = team_total_p['wickets_taken']/team_total_p['bowl_inn']

random_seed = 889
df3 = team_total_p[["sf_flag","bat_sr","runs_per_innings","wickets_per_innings","runs_conceeded_per_ball","bat_sr_std","runs_pm_std","wick_pm_std","runs_p_ball_std","raa","wl_ratio","matches"]]
df3_train, df3_test = train_test_split(df3, test_size=0.2, random_state=random_seed, stratify=df3['sf_flag'])
x3_train = df3_train[["bat_sr","runs_per_innings","wickets_per_innings","runs_conceeded_per_ball","raa","matches",'runs_p_ball_std',"wl_ratio"]]
x3_test = df3_test[["bat_sr","runs_per_innings","wickets_per_innings","runs_conceeded_per_ball","raa","matches",'runs_p_ball_std',"wl_ratio"]]
y3_train = df3_train[['sf_flag']]
y3_test = df3_test[['sf_flag']]
numeric_cols3 = ["bat_sr","runs_per_innings","wickets_per_innings","runs_conceeded_per_ball","raa","matches","wl_ratio"]

clf3 = LogisticRegression(penalty='none') # logistic regression with no penalty term in the cost function.
clf3.fit(x3_train,y3_train)
plot_roc_curve(clf3, x3_test, y3_test)
test3_prob = clf3.predict_proba(x3_test)[:, 1]
test3_pred = clf3.predict(x3_test)
plot_confusion_matrix(clf3, x3_test, y3_test)

''' Salary Prediction Model '''
ps_m4_bat = player_stats_overall[player_stats_overall['Batsman Role'].str.contains(pat = 'Batsman', na = False)][player_stats_overall['Season'] < 2023]
ps_m4_bowl = player_stats_overall[player_stats_overall['Bowler Role'].str.contains(pat = 'Bowler', na = False)][player_stats_overall['Season'] < 2023]
ps_m4_bat_ar = player_stats_overall[player_stats_overall['Batsman Role'].str.contains(pat = 'Batting Allrounder', na = False)][player_stats_overall['Season'] < 2023]
ps_m4_bowl_ar = player_stats_overall[player_stats_overall['Batsman Role'].str.contains(pat = 'Bowling Allrounder', na = False)][player_stats_overall['Season'] < 2023]
player_stats_overall2 = pd.concat([ps_m4_bat,ps_m4_bowl,ps_m4_bat_ar,ps_m4_bowl_ar], axis = 0)

pl_sal_stats = pd.merge(player_stats_overall2,player_lu, on = ['Player'], how = 'left')
pl_sal_stats = pd.merge(pl_sal_stats,sal_data, on = ['Name','Season'],how = 'left')
mlr_d1 = pl_sal_stats[pl_sal_stats['Tag'] != 'Replace'][pl_sal_stats['Tag'] != 'Injured'][pl_sal_stats['Tag'] != 'Pull']
mlr_d1.loc[mlr_d1['Batsman Role'] == 0,'Batsman Role'] = mlr_d1.loc[mlr_d1['Batsman Role'] == 0]['Bowler Role']

mlr_d1 = mlr_d1[mlr_d1['Salary'].notnull()]
mlr_d1['Salary'] = mlr_d1['Salary'].str.replace(",","")
mlr_d1['Salary'] = mlr_d1['Salary'].astype(float)

m1r_d1_sal_lu = mlr_d1[['player2','Salary','Season']]
m1r_d1_sal_lu['Season'] = m1r_d1_sal_lu['Season'].astype(int)
m1r_d1_sal_lu['Season2'] = m1r_d1_sal_lu['Season'] - 1
mlr_d1['Season2'] = mlr_d1['Season'].astype(int)
mlr_d1 = mlr_d1.merge(m1r_d1_sal_lu, on = ['Season2','player2'], how = 'left')
mlr_d1['Salary_y'] = mlr_d1['Salary_y'].fillna(0)

mlr_d1['ind'] = 0
mlr_d1.loc[mlr_d1['nationality'] == 'India','ind'] = 1
mlr_d1 = mlr_d1[mlr_d1['Salary_y'] > 0]

mlr_d1['s_raa']=0
mlr_d1.loc[mlr_d1['Batsman Role'].str.contains(pat = 'Batsman', na = False),'s_raa'] = mlr_d1[mlr_d1['Batsman Role'].str.contains(pat = 'Batsman', na = False)]['bat_raa']
mlr_d1.loc[mlr_d1['Bowler Role'].str.contains(pat = 'Bowler', na = False),'s_raa'] = mlr_d1[mlr_d1['Bowler Role'].str.contains(pat = 'Bowler', na = False)]['bowl_raa']
mlr_d1.loc[mlr_d1['Batsman Role'].str.contains(pat = 'Batting Allrounder', na = False),'s_raa'] = mlr_d1[mlr_d1['Batsman Role'].str.contains(pat = 'Batting Allrounder', na = False)]['bowl_raa']*0.4 + mlr_d1[mlr_d1['Batsman Role'].str.contains(pat = 'Batting Allrounder', na = False)]['bat_raa']*0.6
mlr_d1.loc[mlr_d1['Bowler Role'].str.contains(pat = 'Bowling Allrounder', na = False),'s_raa'] = mlr_d1[mlr_d1['Bowler Role'].str.contains(pat = 'Bowling Allrounder', na = False)]['bowl_raa']*0.6 + mlr_d1[mlr_d1['Bowler Role'].str.contains(pat = 'Bowling Allrounder', na = False)]['bat_raa']*0.4

mlr_d1['s_matches'] = mlr_d1['innings_x']
mlr_d1.loc[mlr_d1['s_matches'] < mlr_d1['innings_y'],'s_matches'] = mlr_d1['innings_y']

#mlr_d1 = mlr_d1[mlr_d1['Tag'] != 'Retain']
x = mlr_d1[['ind','bat_dots_pct','bowl_dots_pct',"bat_sr","runs_per_match","wickets_per_match","runs_conceeded_per_ball","raa","matches","cum_wl_per",'Salary_y','nationality','Batsman Role','Batsman Type','bowl_type','bowl_type2','Tag','Team','s_raa','wl_per','s_matches']]
y = mlr_d1['Salary_x']
x2 = x[['Salary_y','ind']]

x2train, x2test, y2train, y2test = train_test_split(x2, y, test_size = 0.3,random_state=42)
lm_final = LinearRegression()
lm_final.fit(x2train, y2train)
print (lm_final.intercept_)
print (lm_final.coef_)

lm_final.score(x2train, y2train)
x2predict = lm_final.predict(x2test)
MAPE(x2predict,y2test)
rmspe(x2predict,y2test)

xgb_model = xgb.XGBRegressor(colsample_bytree = 0.8, learning_rate= 0.1, max_depth= 10, min_child_weight= 15, n_estimators= 50, nthread= 4, objective='reg:squarederror', silent= 1, subsample= 0.5, reg_alpha = 1.3, reg_lambda = 1.3)
xgb_model.fit(x2train, y2train)
xgb_model.score(x2train, y2train)
y2_pred = xgb_model.predict(x2test)
MAPE(y2_pred,y2test)
rmspe(y2_pred,y2test)

''' XGBoost Model Parameter Tuning using Grid Search Method '''

xgb1 = XGBRegressor()
parameters = {'nthread':[4,10], 
              'objective':['reg:squarederror'],
              'eval_metric':['rmse','mape'],
              'learning_rate': [.07,0.1], #so called `eta` value
              'max_depth': [10,15],
              'min_child_weight': [15],
              'subsample': [0.5],
              'reg_alpha': [1.3],
              'reg_lambda': [1.1],
              'colsample_bytree': [0.5,0.8, 0.1],
              'n_estimators': [50]}

xgb_grid = GridSearchCV(xgb1,
                        parameters,
                        cv = 3,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(x2train, y2train)
print(xgb_grid.best_params_)
print(xgb_grid.best_score_)
