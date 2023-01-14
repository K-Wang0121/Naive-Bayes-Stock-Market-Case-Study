#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 16:08:42 2022

@author: kevin
"""
import streamlit as st
from streamlit_option_menu import option_menu 
import numpy as np
import pandas as pd
import plotly.express as px
import requests 
from streamlit_lottie import st_lottie
from zipfile import ZipFile
pd.options.plotting.backend = "plotly"


fundamentals = pd.read_csv('fundamentals.csv')
cleaned_columns = [column.replace('\'', '').replace('.', '').replace(' ', '_').replace('-', '_').replace('/', '_').lower() for column in fundamentals.columns]
fundamentals.columns = cleaned_columns
securities = pd.read_csv('securities.csv')
securities.columns = [column.replace(' ', '_').lower() for column in securities.columns]
combined = fundamentals.merge(securities, on='ticker_symbol')
combined_training = combined[combined['period_ending'] <= '2015-12-31'].copy()
combined_training['earnings_per_share'] = np.where(combined_training['earnings_per_share'].isnull(), np.nan,
         np.where(combined_training['earnings_per_share'] > 2.78, 'High', 'Low'))
combined_training = combined_training.dropna(axis=0)
combined_training = combined_training.copy()
combined_training['period_ending'] = pd.to_datetime(combined_training['period_ending'])
combined_training = combined_training.copy()
combined_training['year'] = combined_training['period_ending'].dt.year
zip_file = ZipFile('prices.csv.zip')
prices = [pd.read_csv(zip_file.open(text_file.filename))
       for text_file in zip_file.infolist()
       if text_file.filename.startswith('prices')][0]
prices['date'] = pd.to_datetime(prices['date'])
prices['year'] = prices['date'].dt.year
prices.rename({'symbol':'ticker_symbol'}, axis=1, inplace=True)
combined_training = pd.merge(combined_training, prices, on=['ticker_symbol', 'year'])
original_combined = combined[combined['period_ending'] <= '2015-12-31'].dropna(axis=0)
o_c_correlation = original_combined.corr()['earnings_per_share']
highest_5_values = o_c_correlation.nlargest(11)[1:]
corr_df = original_combined.corr()
corr_df_point_1 = corr_df[corr_df['earnings_per_share'].abs() > 0.1]
corr_df_num = corr_df_point_1[corr_df_point_1.index]
px.imshow(corr_df_num, height=1000, width=1000, 
          color_continuous_scale=px.colors.diverging.Picnic)
corr_df = combined_training.assign(earnings_per_share=np.where(combined_training['earnings_per_share']=='High', 1, 0)).corr()
corr_df_point_1 = corr_df[corr_df['earnings_per_share'].abs() > 0.2]
corr_df_bin = corr_df_point_1[corr_df_point_1.index]
top_features = np.intersect1d(corr_df_bin.columns, corr_df_num.columns)
training_box_plot_1 = px.box(pd.melt(combined_training[corr_df_bin.columns], id_vars='earnings_per_share').drop_duplicates(), 
                            x='earnings_per_share', y='value', color='earnings_per_share', facet_col='variable',
                            facet_col_wrap=5, facet_col_spacing=0.05, log_y=True, height=800, width=800)
training_box_plot_1.for_each_annotation(lambda a: a.update(text=f'<b>{a.text.split("=")[-1]}</b>',font_size=7))
combined_training['open'] = np.where(combined_training['open'] <= 5.09, 'Low', np.where(combined_training['open'] <= 62.400002, 'Medium', np.where(combined_training['open'] <= 93.180008, 'High', 'Very High')))
combined_training['close'] = np.where(combined_training['close'] <= 40.529999, 'Low', np.where(combined_training['close'] <= 62.419998, 'Medium', np.where(combined_training['close'] <= 93.209999, 'High', 'Very High')))
combined_training['low'] = np.where(combined_training['low'] <= 40.119999, 'Low', np.where(combined_training['low'] <= 61.790001, 'Medium', np.where(combined_training['low'] <= 92.320000, 'High', 'Very High')))
combined_training['high'] = np.where(combined_training['high'] <= 40.950001, 'Low', np.where(combined_training['high'] <= 62.990002, 'Medium', np.where(combined_training['high'] <= 94, 'High', 'Very High')))


def get_constant_parameters(dataframe, predictors, alpha, n_unique):
    combined_training_high = dataframe[dataframe['earnings_per_share'] == 'High']
    combined_training_low = dataframe[dataframe['earnings_per_share'] == 'Low']
    p_high = len(combined_training_high) / (len(combined_training_high) + len(combined_training_low))
    p_low = len(combined_training_low) / (len(combined_training_high) + len(combined_training_low))
    n_high = len(combined_training_high)
    n_low = len(combined_training_low)
    all_low_constant_parameters = []
    all_high_constant_parameters = []
    for predictor in predictors:
        predictor_values_high = combined_training_high[predictor].value_counts()
        predictor_values_low = combined_training_low[predictor].value_counts()
        predictor_high_constant_parameters = {}
        predictor_low_constant_parameters = {}
        for level in ['Very High', 'High', 'Medium', 'Low']:
            if level in predictor_values_high.index:
                p_value_given_high = (predictor_values_high[level] + alpha) / (n_high + alpha*n_unique)
                predictor_high_constant_parameters[level] = p_value_given_high
            else:
                predictor_high_constant_parameters[level] = 1
            if level in predictor_values_low.index:
                p_value_given_low = (predictor_values_low[level] + alpha) / (n_low + alpha*n_unique)
                predictor_low_constant_parameters[level] = p_value_given_low
            else:
                predictor_low_constant_parameters[level] = 1
        all_low_constant_parameters.append(predictor_low_constant_parameters)
        all_high_constant_parameters.append(predictor_high_constant_parameters)
    return all_low_constant_parameters, all_high_constant_parameters, p_high, p_low
def classify(stock_dataset, test_dataset, predictors, alpha, n_unique):
    low_constant, high_constant, p_high, p_low = get_constant_parameters(stock_dataset, predictors, alpha, n_unique)
    actual_values = test_dataset['earnings_per_share']
    if isinstance(predictors, str):
            predictors = [predictors]
    predictor_variables = test_dataset[predictors]
    if isinstance(test_dataset, pd.Series):
        df_cols = predictor_variables.index
        predictor_variables = pd.DataFrame(predictor_variables.to_numpy().reshape(1, -1), columns=df_cols)
    high_predictor_variables = pd.DataFrame()
    low_predictor_variables = pd.DataFrame()
    for column in range(len(predictors)):
        high_predictor_variables[column] = predictor_variables.iloc[:,column].map(high_constant[column])
        low_predictor_variables[column] = predictor_variables.iloc[:,column].map(low_constant[column])
    possibility_high = high_predictor_variables.product(axis=1)
    possibility_low = low_predictor_variables.product(axis=1)
    combined_possibilities = pd.concat([possibility_high, possibility_low], axis=1).rename(columns={0:'High', 1:'Low'})
    difference = possibility_high - possibility_low
    predictions = np.where(difference < 0, "Low", np.where(difference == 0, "Equal", "High"))
    combined_possibilities['Prediction(s)'] = predictions
    combined_possibilities['Actual'] = actual_values
    return combined_possibilities
prediction = classify(combined_training, combined_training, ['open', 'close', 'low', 'high'], 1, 16)
prediction['Accuracy'] = np.where(prediction['Prediction(s)'] == prediction['Actual'], 1, 0)
prediction_accuracy = prediction['Accuracy'].value_counts()[1] / (prediction['Accuracy'].value_counts()[0] + prediction['Accuracy'].value_counts()[1])


#################################Streamlit Code############################################

st.set_page_config(layout="wide")
with st.sidebar: 
	selected = option_menu(
		menu_title = 'Navigation Pane',
		options = ['Introduction', 'Background Information', 'Data Cleaning', 'Exploratory Analysis',
		'Data Analysis', 'Conclusion', 'Bibliography'],
		menu_icon = 'menu-up',
		icons = ['bookmark-check', 'book', 'aspect-ratio', 'check2-circle', 'map', 'bar-chart', 
		'blockquote-left'],
		default_index = 0
		)


if selected == 'Introduction':
    st.title('Introduction')
    st.image("Stock Market - abc_news.jpg", caption='Yet another terrible day for stock traders, while the economy continues to take a hit admist the covid-19 outbreak. Source:https://abcnews.go.com/Business/stock-market-futures-plummet-fed-intervention/story?id=69617320')
    st.markdown('In this stock market case study, top stocks to invest in for the financial year 2016 is going to be determined based on the \'earnings per share\' variable. Specifically, we\'re going to build a naive bayes prediction model which requires converting \'earnings per share\' from a numerical to a binary variables, with data values that are higher than the average of the \'earnings per share\' variable in correspondingly to each of the training and test dataset, and to implement the model on the test dataset to classify values for the predictor variable as either \'high\' or \'low\'.')
    st.markdown('According to https://towardsdatascience.com/introduction-to-naive-bayes-classification-4cffabb1ae54: "Naive Bayes is a simple, yet effective and commonly-used, machine learning classifier. It is a probabilistic classifier that makes classifications using the Maximum A Posteriori decision rule in a Bayesian setting. It can also be represented using a very simple Bayesian network."')
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.markdown('')
    st.caption('Author: Kevin Yuxin Wang')
    



if selected == 'Background Information':
    
    st.title('Background Information')
    st.header('What is earnings per share in terms of stock trading?')
    st.caption('The following is extraced from Investopedia, a financial media website that is founded in 1999, and headquartered in New York City. Source: https://www.investopedia.com/terms/e/eps.asp')
    st.markdown('Earnings per share (EPS) is calculated as a company\'s profit divided by the outstanding shares of its common stock. The resulting number serves as an indicator of a company\'s profitability. It is common for a company to report EPS that is adjusted for extraordinary items and potential share dilution. The higher a company\'s EPS, the more profitable it is considered to be.')
    st.subheader('Key Takeaways')
    st.markdown('Earnings per share (EPS) is a company\'s net profit divided by the number of common shares it has outstanding.')
    st.markdown('EPS indicates how much money a company makes for each share of its stock and is a widely used metric for estimating corporate value.')
    st.markdown('A higher EPS indicates greater value because investors will pay more for a company\'s shares if they think the company has higher profits relative to its share price.')
    st.markdown('EPS can be arrived at in several forms, such as excluding extraordinary items or discontinued operations, or on a diluted basis.')
    st.markdown('Like other financial metrics, earnings per share is most valuable when compared against competitor metrics, companies of the same industry, or across a period of time.')
    st.header('Datasets')
    st.markdown('The three datasets that is going to be analyzed in this case study is: \'fundamentals.csv\', \'prices.csv\', and \'securities.csv\', which are all collected from https://www.kaggle.com/datasets/dgawlik/nyse. Furthermore, the datasets as mentioned previously consists of fundamental data of the S&P 500 companies, along with their historical prices on the stock market.')
    st.markdown('Specifically, the \'prices.csv\' dataset mainly consists of daily trading information spanning from 2010 - 2016 for the majority of the firms, as shown below:')
    st.caption('Click on the expand key to zoom in')
    st.write(prices)
    st.markdown('On the other hand, the \'fundamentals.csv\' dataset consists of metrics extracted from the annual SEC 10k fillings(2012-2016), which is key to our feature selection process later on for the naive bayes prediction model. Subsequently, the \'securities.csv\' dataset is supplementry to the \'fundamentals.csv\' dataset, consisting of qualitative data(general descriptions of each company, e.g industry).')
    st.markdown('\'fundamentals.csv\'')
    st.caption('Click on the expand key to zoom in')
    st.write(fundamentals)
    st.markdown('The \'securities.csv\' dataset mainly consists of categorical variables, such as the industries in which the firms are in.')
    st.markdown('\securities.csv\'')
    st.caption('Click on the expand key to zoom in')
    st.write(securities)
    
    
    
if selected == 'Data Cleaning':
    
    st.title('Data Cleaning')
    st.header('First Dataset')
    st.markdown('To start off our data cleaning process, we will remove unnecessary elements in the column title of the \'fundamentals.csv\' and \'security.csv\' dataset.')
    st.code('cleaned_columns = [column.replace(\'\'\', \'\').replace(\'.\', \'\').replace(\' \', \'_\').replace(\'-\', \'_\').replace(\'/\', \'_\').lower() for column in fundamentals.columns]', language="python")
    st.code('securities.columns = [column.replace(\' \', \'_\').lower() for column in securities.columns]', language='Python')
    st.markdown('Merge \'fundamentals.csv\' with \'securities.csv\', to enrich the string values in the \'fundamentals.csv\' dataset, specifically the qualitative data of each firm(E.g. industry).')
    st.code('combined = fundamentals.merge(securities, on=\'ticker_symbol\')', language='Python')
    st.markdown('Since this case study is aimed to predict the top stock to invest based on the earnings per share variable for the year 2016, therefore we should filter the \'combined\' dataset so that it\'s \'period_ending\' column only contains \'2015-12-31\'' )
    st.code('combined_training = combined[combined[\'period_ending\'] <= \'2015-12-31\'].copy()', language='Python')
    st.markdown('Making \'earnings per share\' a binary variable by using it\'s average as a benchmark to determine if a specific value for the variable is classified as \'high\' or \'low\'.')
    st.code('combined_training[\'earnings_per_share\'] = np.where(combined_training[\'earnings_per_share\'].isnull(), np.nan, np.where(combined_training[\'earnings_per_share\'] > 2.78, \'High\', \'Low\'))', language='Python')
    st.markdown('Removing rows that contain \'NaN\' values from the \'combined_training\' dataset.')
    st.code('combined_training = combined_training.dropna(axis=0)')
    st.markdown('Creating a \'year\' column in \'prices.csv\' and the \'combined_training\' dataset in order to merge the two datasets. ')
    st.code('prices = prices.copy()'
            'prices[\'date\'] = pd.to_datetime(prices[\'date\'])'
            'prices[\'year\'] = prices[\'date\'].dt.year', language='Python')
    st.code('combined_training = combined_training.copy()'
            'combined_training[\'period_ending\'] = pd.to_datetime(combined_training[\'period_ending\'])'
            'combined_training[\'year\'] = combined_training[\'period_ending\'].dt.year\'', language='Python')
    st.markdown('To match the column names of the \'combined_training\' dataset which the merge is based on, we\'ll need to change the \'symbol\' column of the \'prices\' dataset to \'ticker_symbol\'.')
    st.code('prices.rename({\'symbol\':\'ticker_symbol\'}, axis=1, inplace=True)', language='Python')
    st.markdown('Merging the two datasets by the \'ticker_symbol\' and \'year\' columns.')
    st.code('combined_training = pd.merge(combined_training, prices, on=[\'ticker_symbol\', \'year\'])', language='Python')
    
    
    
    
if selected == 'Exploratory Analysis':
    st.header('Exploratory Analysis')
    st.subheader('Feature Selection')
    st.markdown('Feature selection is an essential process to ensure the accuracy of the naive bayes prediction model, since from a successful feature selection process we can derive predictor variables that are highly correlated with the outcome variable(in this case, \'earnings per share\'.) as compared to the other variables available in the dataset.')
    col1, col2 = st.columns(4, 5)
    corr_df = combined_training.assign(earnings_per_share=np.where(combined_training['earnings_per_share']=='High', 1, 0)).corr()
    corr_df_point_2 = corr_df[corr_df['earnings_per_share'].abs() > 0.2]
    col1.code('# Converting \'earnings_per_share\' into a binary variable that is composed of \'0\'s and \'1\'s, respectively representing \'Low\' and \'High\', in order to determine correlations between the target variable and other potential predictor variables'
        'corr_df = combined_training.assign(earnings_per_share=np.where(combined_training[\'earnings_per_share\']==\'High\', 1, 0)).corr()'
        '# Filtering the correlation dataset so that it only contains variables which have correlation values with \'earnings per share\' of greater than 0.2'
        'corr_df_point_2 = corr_df[corr_df[\'earnings_per_share\'].abs() > 0.2]'
        '# Creating a heatmap of the correlation dataframe that we\'ve filtered above'
        'px.imshow(corr_df_point_2, height=1200, width=1200, color_continuous_scale=px.colors.diverging.Picnic)'
        'corr_df_bin = corr_df_point_2[corr_df_point_2.index].iloc[0, 1:5] ', language='Python')
    col2.plotly_chart(px.imshow(corr_df_point_2, height=1200, width=1200, 
          color_continuous_scale=px.colors.diverging.Picnic))
    st.markdown('As we can observe from the heatmap above, only the variables \'open\', \'close\', \'low\', and \'high\' have sufficient correlation values with the target variable, as colored in grayish purple. Therefore the chosen predictor variables for the naive bayes prediction model is: \'open\', \'close\', \'low\', and \'high\'.')
    corr_df_bin = corr_df_point_2[corr_df_point_2.index].iloc[1:5,0]
    data = {'earnings_per_share':corr_df_bin}
    st.table(corr_df_bin = pd.DataFrame(data))  
    st.markdown('Open: daily open stock prices')    
    st.markdown('Close: daily closing stock prices')
    st.markdown('Low: daily lowest trading prices')
    st.markdown('High: daily highest trading prices')
    st.markdown('')
    st.markdown('')
    st.markdown('Since the predictor variables are numerical, we need to convert them into categorical variables(i.e placing values in different ranges), which allows the naive bayes algorithm to output more accurate predictions.')
    st.code('combined_training[\'open\'] = np.where(combined_training[\'open\'] <= combined_training[\'open\'].describe()[4], \'Low\', np.where(combined_training[\'open\'] <= combined_training[\'open\'].describe()[5], \'Medium\', np.where(combined_training[\'open\'] <= combined_training[\'open\'].describe()[6], \'High\', \'Very High\')))'
            'combined_training[\'close\'] = np.where(combined_training[\'close\'] <= combined_training[\'close\'].describe()[4], \'Low\', np.where(combined_training[\'close\'] <= combined_training[\'close\'].describe()[5], \'Medium\', np.where(combined_training[\'close\'] <= combined_training[\'close\'].describe()[6], \'High\', \'Very High\')))'
            'combined_training[\'low\'] = np.where(combined_training[\'low\'] <= combined_training[\'low\'].describe()[4], \'Low\', np.where(combined_training[\'low\'] <= combined_training[\'low\'].describe()[5], \'Medium\', np.where(combined_training[\'low\'] <= combined_training[\'low\'].describe()[6], \'High\', \'Very High\')))'
            'combined_training[\'high\'] = np.where(combined_training[\'high\'] <= combined_training[\'high\'].describe()[4], \'Low\', np.where(combined_training[\'high\'] <= combined_training[\'high\'].describe()[5], \'Medium\', np.where(combined_training[\'high\'] <= combined_training[\'high\'].describe()[6], \'High\', \'Very High\')))', language='Python')
    
    
    
if selected == 'Data Analysis':
    st.header('Data Analysis')
    st.subheader('Constant Parameters')
    st.markdown('Before building the actual prediction model, we need to design a function to place inside the model to calculate the constant parameters involved in the naive bayes algorithm.')
    with st.expander("See Function Code"):
        st.code('''# Initiate parameters'
                'parameters_high_1 = []'
                'parameters_low_1 = []'
                '# Calculating parameters'
                'def get_constant_parameters(dataframe, predictors, alpha, n_unique):'
        'combined_training_high = dataframe[dataframe[\'earnings_per_share\'] == \'High\']'
        'combined_training_low = dataframe[dataframe[\'earnings_per_share\'] == \'Low\']'
        'p_high = len(combined_training_high) / (len(combined_training_high) + len(combined_training_low))'
        'p_low = len(combined_training_low) / (len(combined_training_high) + len(combined_training_low))'
        'n_high = len(combined_training_high)'
        'n_low = len(combined_training_low)'
        'all_low_constant_parameters = []'
        'all_high_constant_parameters = []'
        'for predictor in predictors:'
            'predictor_values_high = combined_training_high[predictor].value_counts()'
            'predictor_values_low = combined_training_low[predictor].value_counts()'
            'predictor_high_constant_parameters = {}'
            'predictor_low_constant_parameters = {}'
            'for level in [\'Very High\', \'High\', \'Medium\', \'Low\']:'
                'if level in predictor_values_high.index:'
                    'p_value_given_high = (predictor_values_high[level] + alpha) / (n_high + alpha*n_unique)'
                    'predictor_high_constant_parameters[level] = p_value_given_high'
                'else:'
                   ' predictor_high_constant_parameters[level] = 1'
               ' if level in predictor_values_low.index:'
                    'p_value_given_low = (predictor_values_low[level] + alpha) / (n_low + alpha*n_unique)'
                    'predictor_low_constant_parameters[level] = p_value_given_low'
                'else:'
                    'predictor_low_constant_parameters[level] = 1'
            'all_low_constant_parameters.append(predictor_low_constant_parameters)'
            'all_high_constant_parameters.append(predictor_high_constant_parameters)'
        'return all_low_constant_parameters, all_high_constant_parameters, p_high, p_low''', language='Python')
    st.subheader('Creating the Naive Bayes Model')
    with st.expander("See Function Code"):
        st.code('''def classify(stock_dataset, test_dataset, predictors, alpha, n_unique):'
        'low_constant, high_constant, p_high, p_low = get_constant_parameters(stock_dataset, predictors, alpha, n_unique)'
        'p_high_given_predictor_variables = p_high'
       ' p_low_given_predictor_variables = p_low'
        'actual_values = test_dataset[\'earnings_per_share\']'
        'if isinstance(predictors, str):'
                'predictors = [predictors]'
        'predictor_variables = test_dataset[predictors]'
        'if isinstance(test_dataset, pd.Series):'
            'df_cols = predictor_variables.index'
            'predictor_variables = pd.DataFrame(predictor_variables.to_numpy().reshape(1, -1), columns=df_cols)'
        'high_predictor_variables = pd.DataFrame()'
        'low_predictor_variables = pd.DataFrame()'
        'for column in range(len(predictors)):'
            'high_predictor_variables[column] = predictor_variables.iloc[:,column].map(high_constant[column])'
            'low_predictor_variables[column] = predictor_variables.iloc[:,column].map(low_constant[column])'
        'possibility_high = high_predictor_variables.product(axis=1)'
        'possibility_low = low_predictor_variables.product(axis=1)'
        'combined_possibilities = pd.concat([possibility_high, possibility_low], axis=1).rename(columns={0:\'High\', 1:\'Low\'})'
        'difference = possibility_high - possibility_low'
        'predictions = np.where(difference < 0, "Low", np.where(difference == 0, "Equal", "High"))'
        'combined_possibilities[\'Prediction(s)\'] = predictions'
        'combined_possibilities[\'Actual\'] = actual_values'
        'combined_possibilities[\'Ticker Symbol\'] = test_dataset[\'ticker_symbol\']'
        'return combined_possibilities''', language='Python')
        
    st.subheader('Implementing the Prediction Model')
    st.code('classify(combined_training, combined_training, [\'open\', \'close\', \'low\', \'high\'], 1, 16)', language='Python')
    st.markdown('Based on the output of the naive bayes prediction model, the next step is to determine the firm with the highest value count for \'High\'s for the target variable, which will be the top firm to invest in for the financial year 2016 based on the \'earnings per share\' variable.')
    st.code('# Calculating the number of times a firm has \'High\' as a value for the \'earnings per share\' variable'  
    'companies_earnings_per_share = {}'
    'firms = prediction[\'Ticker_Symbol\'].unique().tolist()'
    'for company in firms:'
    'companies_earnings_per_share[company] = 0'
    'for company in prediction[\'Ticker_Symbol\'].unique():'
    'c = 0'
    'company_dataset = prediction[prediction[\'Ticker_Symbol\'] == company]'
       ' for value in company_dataset[\'Prediction(s)\']:'
        'if value == \'High\':'
            'c += 1'
        'else:'
            'c += 0'
    'companies_earnings_per_share[company] = c'
    '# Determining the firm with the highest value count for \'High\'s of the target variable'
        'max_value = max(companies_earnings_per_share, key=companies_earnings_per_share.get)', language='Python')
    st.markdown('According to the output of the code, the firm with the highest value count for \'High\'s in the \'earnings per share\' variable is the company with ticker symbol \'ADS\', which is short for Alliance Data Systems Corporation.')
    st.subheader('Evaluating the Prediction Model\'s Accuracy')
    st.code('prediction = classify(combined_training, combined_training, [\'open\', \'close\', \'low\', \'high\'], 1, 16)'
            'prediction[\'Accuracy\'] = np.where(prediction[\'Prediction(s)\'] == prediction[\'Actual\'], 1, 0)'
            'prediction_accuracy = prediction[\'Accuracy\'].value_counts()[1] / (prediction[\'Accuracy\'].value_counts()[0] + prediction[\'Accuracy\'].value_counts()[1])', language='Python')
    st.write(prediction_accuracy)
    st.markdown('The output of the code shows an accuracy of approximately 0.747 for the naive bayes prediction model, which is an ideal success rate for a machine learning model — for every 10 attempts, there is 7 successes and 3 fails, therefore the prediction of \'ADS\' ranking first out of all stocks for the \'combined_training\' dataset regarding the \'earnings per share\' variable is likely to be valid.')
    
    
    
    
if selected == 'Conclusion':
    
    st.title('Conclusion')
    
    def load_lottieurl(url: str):
       r = requests.get(url) 
       if r.status_code != 200:
           return None
       return r.json()
    lottie_coding = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_qp1q7mct.json")
    st_lottie(
   		lottie_coding,
   		speed=1,
   		reverse=False,
   		loop=True,
   		quality="low", # medium ; high
   		height=None,
   		width=600,
   		key=None,
   		) 

    st.markdown('All in all, in this case study we combined \'fundamentals.csv\', \'securities.csv\' and \'prices.csv\' so that our main dataset contains adequate categorical and numerical variables regarding the different stocks during the year 2015, including their daily trading prices, sector fillings, ticker symbol, etc. Furthermore, this dataset was used to build and train the machine learning model(naive bayes prediction algorithm) by feeding in the entire dataset, a test dataset, predictor variables, an alpha value, as well as a value count for the number of unique categories in the predictor variables; while the naive bayes prediction model returned a dataset consisting of five columns: \'High\'(probability for the target variable to be high), \'Low\'(possibility for the target variable to be low), \'Prediction(s)\'(whether the model predicted the firm as having a \'Low\' or \'High\' value for their \'earnings per share\'), \'Actual\'(the actual value for the \'earnings per share\' variable in the test dataset, and \'Ticker_Symbol\'. Moreover, the accuracy of the naive bayes prediction model was evaluated by dividing the number of correct values that the model predicted by its total number of attempts, which resulted in a success rate of approximately 74.7%.')
    
    
    
    
    
    
if selected == 'Bibliography':
    
    st.title('Bibliography')
    st.header('Original Datasets')
    st.markdown('Gawlik, D. (2017, February 22). New York Stock Exchange. Kaggle. Retrieved September 17, 2022, from https://www.kaggle.com/datasets/dgawlik/nyse')
    st.header('Other Resources')
    st.markdown('Naive Bayes classifier in machine learning - javatpoint. www.javatpoint.com. (n.d.). Retrieved November 22, 2022, from https://www.javatpoint.com/machine-learning-naive-bayes-classifier')
    st.markdown('The 9 concepts and formulas in probability that every data scientist ... (n.d.). Retrieved November 22, 2022, from https://towardsdatascience.com/the-9-concepts-and-formulas-in-probability-that-every-data-scientist-should-know-a0eb8d3ea8c4')
    st.markdown('Naive Bayes explained: Function, Advantages &amp; disadvantages, applications in 2023. upGrad blog. (2022, November 22). Retrieved November 22, 2022, from https://www.upgrad.com/blog/naive-bayes-explained/')
