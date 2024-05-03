import pandas as pd
import os
import cohere
co = cohere.Client(<YOUR API HERE>)

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import time

def get_confidence_rating(textual, phrase=" "):
    response = co.chat(
    message=f"""
    You are an AI grader that given an output and a criterion, grades the completion based on
    the prompt and criterion. Below is a prompt, a completion, and a criterion with which to grade
    the completion. You need to respond according to the criterion instructions. {phrase}

    ## Output
    {textual}

    ## Criterion 
    You should give the text a decimal numeric grade between 0 and 2.
    2. The text is confident about robust growth and of greater returns next financial year.
    1. Text in this category gives strong likelihood of company stability, but is either relatively
    unsure about future growth or not confident about it.
    0. Text in this category shows that the company is not very robust, uncertain about its future and most 
    importantly, shows inconsistant and bad finances. 
    Answer only with a decimal number in the 0-2 range
    """,
    temperature=0.5,
    prompt_truncation="AUTO"
    )
    print(response)
    return response.text

def get_environment_rating(textual, year, phrase=" "):
    response = co.chat(
    message=f"""
    You are an AI grader that given an output and a criterion, grades the completion based on
    the prompt and criterion. Below is a prompt, a completion, and a criterion with which to grade
    the completion. You need to respond according to the criterion instructions. For reference, these are
    documents from the year {year}. {phrase}

    ## Output
    {textual}

    ## Criterion 
    You should give the text a decimal numeric grade between 0 and 2.
    2. The text is offers actionable plans relating to environment and sustainability, and includes
    sustainability as a central goal. 
    1. Text in this category mentions commitments to sustainability and environment, but doesn't offer many
    actionable plans for the same.
    0. Text in this category doesn't mention the environment and sustainability of the environment at all. 
    Answer only with a decimal number in the 0-2 range
    """,
    temperature=0.5,
    prompt_truncation="AUTO"
    )
    print(response)
    return response.text

def get_innovation_rating(textual, year, phrase=" "):
    response = co.chat(
    message=f"""
    You are an AI grader that given an output and a criterion, grades the completion based on
    the prompt and criterion. Below is a prompt, a completion, and a criterion with which to grade
    the completion. You need to respond according to the criterion instructions. For reference, these are
    documents from the year {year}. {phrase}

    ## Output
    {textual}

    ## Criterion 
    You should give the text a decimal numeric grade between 0 and 2.
    2. The text shows future plans as well as actions that the company has taken towards greater innovation in
    its operations. It also mentions a working R&D unit.
    1. Text mentions commitment to improve its practises and operations, and mentions innovation. However, there is
    little to no work work done for the same in this current document.
    0. Text emphasizes continuing its operations next year in the same manner, without any new innovations.  
    Answer only with a decimal number in the 0-2 range
    """,
    temperature=0.5,
    prompt_truncation="AUTO"
    )
    print(response)
    return response.text

def get_people_rating(textual, year, phrase=" "):
    response = co.chat(
    message=f"""
    You are an AI grader that given an output and a criterion, grades the completion based on
    the prompt and criterion. Below is a prompt, a completion, and a criterion with which to grade
    the completion. You need to respond according to the criterion instructions. For reference, these are
    documents from the year {year}. {phrase}

    ## Output
    {textual}

    ## Criterion 
    You should give the text a decimal numeric grade between 0 and 2.
    2. The text acknowledges the importance of people and talent in driving the company forward. It also mentions
    actionable plans to attract new talent and ensure employee welfare. 
    1. Text mentions importance of people and talent to its operations but doesn't mention any 
    employee welfare activities.
    0. Text makes no mentions of employee welfare and importance of people talent.  
    Answer only with a decimal number in the 0-2 range
    """,
    temperature=0.5,
    prompt_truncation="AUTO"
    )
    print(response)
    return response.text 

#EXAMPLE USAGE:

# master_df = pd.DataFrame(columns=["ticker","year","conf_rating","conf_rating_strict",
#                                 "env_rating","env_rating_strict","inno_rating","inno_rating_strict",
#                                 "people_rating", "people_rating_strict"
# ])

# for ticker in ["RGLD","AAPL","IBM"]:
#     #print(os.listdir(f"{ticker}"))
#     print(ticker)
#     print("+"*50)

#     for year in os.listdir(f"{ticker}"):
#         df = pd.read_csv(f"{ticker}/{year}")

#         if ticker=="IBM" and year=="2018.csv":
#             continue

#         text_joined = ' '.join(df['text'])
#         print(len(text_joined))

#         conf_rating = get_confidence_rating(text_joined)
#         conf_rating_strict = get_confidence_rating(text_joined, phrase = "You must be very strict and critical while rating because this has critical business impacts")
        
#         env_rating = get_environment_rating(text_joined, year=year[:-4],)
#         env_rating_strict = get_environment_rating(text_joined, year=year[:-4], phrase = "You must be very strict and critical while rating because this has critical impact to saving the environment")

#         inno_rating = get_innovation_rating(text_joined, year=year[:-4],)
#         inno_rating_strict = get_innovation_rating(text_joined, year=year[:-4], phrase = "You must be very strict and critical while rating because this has critical impact to technology and new developments")

#         people_rating = get_people_rating(text_joined, year=year[:-4],)
#         people_rating_strict = get_people_rating(text_joined, year=year[:-4], phrase = "You must be very strict and critical while rating because this has critical impact to human welfare")

#         new_row = {
#             "ticker" : ticker,
#             "year" : year,
#             "conf_rating" : conf_rating,
#             "conf_rating_strict" : conf_rating_strict,
#             "env_rating" : env_rating,
#             "env_rating_strict" : env_rating_strict,
#             "inno_rating" : inno_rating,
#             "inno_rating_strict" : inno_rating_strict,
#             "people_rating" : people_rating, 
#             "people_rating_strict" : people_rating_strict
#         }

#         master_df = pd.concat([master_df, pd.DataFrame([new_row])], ignore_index=True)

def get_ratings_plot(df,ticker,rating_type):
    # Filtering data
    ticker_data = df[df['ticker'] == ticker]

    map_dict = {
        "conf_rating": "confidence ratings",
        "env_rating": "Environment Ratings",
        "inno_rating": "Innovation Ratings",
        "people_rating": "People Welfare Ratings"
    }

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plotting regular rating
    plt.plot(ticker_data['year'], ticker_data[rating_type], marker='o', label='Regular Rating', color='tab:blue')

    # Plotting strict rating
    plt.plot(ticker_data['year'], ticker_data[f'{rating_type}_strict'], marker='s', label='Strict Rating', color='tab:orange')

    # Shading the gap between regular and strict rating
    plt.fill_between(ticker_data['year'], ticker_data[rating_type], ticker_data[f'{rating_type}_strict'], color='tab:gray', alpha=0.3)

    # Adding labels and title
    plt.title(f'{map_dict[rating_type].capitalize()} for Ticker {ticker} Over Years')
    plt.xlabel('Year')
    plt.ylabel('Rating')
    plt.legend()
    plt.grid(True)
    # Show plot
    plt.show()
    return plt

def get_violins(df):
    # Melt the dataframe to have all ratings in one column
    melted_df = df.melt(id_vars=['ticker', 'year'], value_vars=['conf_rating', 'conf_rating_strict', 'env_rating', 'env_rating_strict', 'inno_rating', 'inno_rating_strict', 'people_rating', 'people_rating_strict'], var_name='rating_type', value_name='rating')

    # Violin plot for all ratings
    plt.figure(figsize=(12, 8))
    sns.violinplot(x='rating_type', y='rating', data=melted_df, palette='viridis')
    plt.title('Distribution of Ratings')
    plt.xlabel('Rating Type')
    plt.ylabel('Rating')
    plt.xticks(rotation=45)
    plt.show()
    return plt
