from prompts_api_calls import *
from data_engineering import get_10k_filings_by_ticker_with_years, pipeline_api
import streamlit as st
import pandas as pd
import numpy as np

st.title('Demo App - 10K Filing Analysis Using LLMs')

st.write("""This demo web app creates and displays LLM ratings of 10-K filings. 
:red[WARNING] As it can take a long time to run for new tickers **(~30 mins)** , it is 
recommended that only the cached tickers are used: `IBM` / `AAPL` / `RGLD`
""")

ticker = st.text_input("*Enter Ticker*", value="AAPL", max_chars=4)

SECTIONS_10K = (
    "BUSINESS",  # ITEM 1
    "RISK_FACTORS",  # ITEM 1A
    "UNRESOLVED_STAFF_COMMENTS",  # ITEM 1B
    "PROPERTIES",  # ITEM 2
    "LEGAL_PROCEEDINGS",  # ITEM 3
    "MINE_SAFETY",  # ITEM 4
    "MARKET_FOR_REGISTRANT_COMMON_EQUITY",  # ITEM 5
    # NOTE(robinson) - ITEM 6 is "RESERVED"
    "MANAGEMENT_DISCUSSION",  # ITEM 7
    "MARKET_RISK_DISCLOSURES",  # ITEM 7A
    "FINANCIAL_STATEMENTS",  # ITEM 8
    "ACCOUNTING_DISAGREEMENTS",  # ITEM 9
    "CONTROLS_AND_PROCEDURES",  # ITEM 9A
    # NOTE(robinson) - ITEM 9B is other information
    "FOREIGN_JURISDICTIONS",  # ITEM 9C
    "MANAGEMENT",  # ITEM 10
    "COMPENSATION",  # ITEM 11
    "PRINCIPAL_STOCKHOLDERS",  # ITEM 12
    "RELATED_PARTY_TRANSACTIONS",  # ITEM 13
    "ACCOUNTING_FEES",  # ITEM 14
    "EXHIBITS",  # ITEM 15
    "FORM_SUMMARY",  # ITEM 16
)

master_col_1, master_col_2 = st.columns(2)
popover = master_col_2.popover("Filter items")
#red = popover.checkbox("Show red items.", True)

list_exclude = []
for section in SECTIONS_10K:
    list_exclude.append(popover.checkbox(f"{section}",False,key=section))

if master_col_1.button('Create Visuals'):
    if ticker in ["AAPL","IBM","RGLD"]:
        df = pd.read_csv("mstr_df_012_1.csv")
        father_col_1, father_col_2 = st.columns(2)
        with father_col_1.container():
            plot_conf = get_ratings_plot(df=df,ticker=ticker,rating_type="conf_rating")
            st.pyplot(plot_conf.gcf())
            st.write("""
            This is the overall ratings given by an LLM asked to evaluate the company's future performance
            and general robustness. Two LLMs were used, where one LLM is asked to be stricter than the other.
            """)

        with father_col_1.container():
            plot_env = get_ratings_plot(df=df,ticker=ticker,rating_type="env_rating")
            st.pyplot(plot_env.gcf())
            st.write("""
            This is the rating given by an LLM asked to evaluate the company's commitment to protecting 
            the environment and its general sustainability practises. Two LLMs were used, 
            where one LLM is asked to be stricter than the other.
            """)

        with father_col_2.container():
            plot_inno = get_ratings_plot(df=df,ticker=ticker,rating_type="inno_rating")
            st.pyplot(plot_inno.gcf())
            st.write("""
            This is the rating given by an LLM asked to evaluate the company's adoption and future plans
            to adopt innovative practises in advance its operational efficiency. Two LLMs were used, 
            where one LLM is asked to be stricter than the other.
            """)
        
        with father_col_2.container():
            plot_people = get_ratings_plot(df=df,ticker=ticker,rating_type="people_rating")
            st.pyplot(plot_people.gcf())
            st.write("""
            This is the rating given by an LLM asked to evaluate the company's retaining 
            and training of talent, along with employee welfare practices. Two LLMs were used, 
            where one LLM is asked to be stricter than the other.
            """)
        
        ticker_data = df[df['ticker'] == ticker]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Confidence Rating", f"{ticker_data.conf_rating.mean()}")
        col2.metric("Average Environment Rating", f"{ticker_data.env_rating.mean()}")
        col3.metric("Average Innovation Rating", f"{ticker_data.inno_rating.mean()}")
        col4.metric("Average People Rating", f"{ticker_data.people_rating.mean()}")

        ticker_data = df[df['ticker'] == ticker]
        ticker_data['year'] = ticker_data['year'].str[:-4]
        df_percentage = ticker_data[['conf_rating', 'env_rating', 'inno_rating', 'people_rating']].div(ticker_data[['conf_rating', 'env_rating', 'inno_rating', 'people_rating']].sum(axis=1), axis=0) * 100

        # Plot stacked bar chart
        plt.figure(figsize=(10, 6))
        df_percentage.plot(kind='bar', stacked=True)
        plt.title('Year-on-Year Percentage of Rating Values')
        plt.xlabel('Year')
        plt.ylabel('Percentage')
        plt.xticks(range(len(ticker_data['year'])), ticker_data['year'])
        plt.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        st.pyplot(plt.gcf())

        st.write(""" Comparision of 10-K Filings of Different companies: 
        Multiple entires of various sections were entered into an LLM and the LLM 
        chose which one is better """)
        st.image("FINTECH_outputs/comparision.png")
        st.image("FINTECH_outputs/comparision_YOY.png")

    else:
        if not os.path.exists(f"ticker_static/{ticker}"):
            os.makedirs(f"ticker_static/{ticker}")
            print(f"Directory created successfully.")
            st.info("Executing the Data Engineering Pipeline")
            ten_k_filings_with_years = get_10k_filings_by_ticker_with_years(ticker)

            for year, filing_text in ten_k_filings_with_years:
                if int(year) < 2020:
                    continue
                print(f"Year: {year}")
                all_narratives = pipeline_api(filing_text, response_type="text/csv", m_section=["_ALL"])

                with open(f'ticker_static/{ticker}/{year}.csv', 'w') as out:
                    out.write(all_narratives)
            st.success("Created the Dataset")
        else:
            print(f"Directory already exists.")
            st.info("DataSet found in cache")
        
        if not os.path.exists(f"{ticker}.csv"):
            st.info("Accessing the API")

            master_df = pd.DataFrame(columns=["ticker","year","conf_rating","conf_rating_strict",
                                    "env_rating","env_rating_strict","inno_rating","inno_rating_strict",
                                    "people_rating", "people_rating_strict"
                ])

            for year in os.listdir(f"ticker_static/{ticker}"):
                df = pd.read_csv(f"ticker_static/{ticker}/{year}")
                # st.write(f"ticker_static/{ticker}/{year}")
                certain_values = []
                for sect,value in zip(SECTIONS_10K,list_exclude):
                    if value:
                        certain_values.append(sect)
                
                # st.write(df.shape)
                df = df[~df['section'].isin(certain_values)]
                
                text_joined = ' '.join(df['text'])
                print(len(text_joined))

                conf_rating = get_confidence_rating(text_joined.replace("  ",""))
                conf_rating_strict = get_confidence_rating(text_joined.replace("  ",""), phrase = "You must be very strict and critical while rating because this has critical business impacts")
                
                env_rating = get_environment_rating(text_joined.replace("  ",""), year=year[:-4],)
                env_rating_strict = get_environment_rating(text_joined.replace("  ",""), year=year[:-4], phrase = "You must be very strict and critical while rating because this has critical impact to saving the environment")

                inno_rating = get_innovation_rating(text_joined.replace("  ",""), year=year[:-4],)
                inno_rating_strict = get_innovation_rating(text_joined.replace("  ",""), year=year[:-4], phrase = "You must be very strict and critical while rating because this has critical impact to technology and new developments")

                people_rating = get_people_rating(text_joined.replace("  ",""), year=year[:-4],)
                people_rating_strict = get_people_rating(text_joined.replace("  ",""), year=year[:-4], phrase = "You must be very strict and critical while rating because this has critical impact to human welfare")

                new_row = {
                    "ticker" : ticker,
                    "year" : year,
                    "conf_rating" : conf_rating,
                    "conf_rating_strict" : conf_rating_strict,
                    "env_rating" : env_rating,
                    "env_rating_strict" : env_rating_strict,
                    "inno_rating" : inno_rating,
                    "inno_rating_strict" : inno_rating_strict,
                    "people_rating" : people_rating, 
                    "people_rating_strict" : people_rating_strict
                }

                master_df = pd.concat([master_df, pd.DataFrame([new_row])], ignore_index=True)

            master_df.to_csv(f"{ticker}.csv")

        df = pd.read_csv(f"{ticker}.csv")

        father_col_1, father_col_2 = st.columns(2)
        with father_col_1.container():
            plot_conf = get_ratings_plot(df=df,ticker=ticker,rating_type="conf_rating")
            st.pyplot(plot_conf.gcf())
            st.write("""
            This is the overall ratings given by an LLM asked to evaluate the company's future performance
            and general robustness. Two LLMs were used, where one LLM is asked to be stricter than the other.
            """)

        with father_col_1.container():
            plot_env = get_ratings_plot(df=df,ticker=ticker,rating_type="env_rating")
            st.pyplot(plot_env.gcf())
            st.write("""
            This is the rating given by an LLM asked to evaluate the company's commitment to protecting 
            the environment and its general sustainability practises. Two LLMs were used, 
            where one LLM is asked to be stricter than the other.
            """)

        with father_col_2.container():
            plot_inno = get_ratings_plot(df=df,ticker=ticker,rating_type="inno_rating")
            st.pyplot(plot_inno.gcf())
            st.write("""
            This is the rating given by an LLM asked to evaluate the company's adoption and future plans
            to adopt innovative practises in advance its operational efficiency. Two LLMs were used, 
            where one LLM is asked to be stricter than the other.
            """)
        
        with father_col_2.container():
            plot_people = get_ratings_plot(df=df,ticker=ticker,rating_type="people_rating")
            st.pyplot(plot_people.gcf())
            st.write("""
            This is the rating given by an LLM asked to evaluate the company's retaining 
            and training of talent, along with employee welfare practices. Two LLMs were used, 
            where one LLM is asked to be stricter than the other.
            """)
        
        ticker_data = df[df['ticker'] == ticker]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Average Confidence Rating", f"{ticker_data.conf_rating.astype(float).mean()}")
        col2.metric("Average Environment Rating", f"{ticker_data.env_rating.astype(float).mean()}")
        col3.metric("Average Innovation Rating", f"{ticker_data.inno_rating.astype(float).mean()}")
        col4.metric("Average People Rating", f"{ticker_data.people_rating.astype(float).mean()}")

        ticker_data = df[df['ticker'] == ticker]
        ticker_data['year'] = ticker_data['year'].str[:-4]
        df_percentage = ticker_data[['conf_rating', 'env_rating', 'inno_rating', 'people_rating']].div(ticker_data[['conf_rating', 'env_rating', 'inno_rating', 'people_rating']].sum(axis=1), axis=0) * 100

        # Plot stacked bar chart
        plt.figure(figsize=(10, 6))
        df_percentage.plot(kind='bar', stacked=True)
        plt.title('Year-on-Year Percentage of Rating Values')
        plt.xlabel('Year')
        plt.ylabel('Percentage')
        plt.xticks(range(len(ticker_data['year'])), ticker_data['year'])
        plt.legend(title='Rating', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        st.pyplot(plt.gcf())

    
        st.write("NO")
    st.write("THE END")


