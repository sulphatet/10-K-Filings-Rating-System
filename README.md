# 10-K-Filings-Rating-System
This repository contains the codebase to pull and clean the data from a SEC 10-K filing of a company, along with functionality to use an LLM based rater. 

# Usage
Replace `<YOUR API HERE>` with your Co:here API key in `prompts_api_calls.py`
Run `streamlit run main.py` to create an interactive window.

# Methodology
## Pulling the 10 - K Filings Data and cleaning it
The function `get_10k_filings_by_ticker_with_years` in `data_engineering.py` pulls the raw data as a string given the ticker. This uses the SEC-Edgar downloader.
We use [Unstructured-IO](https://github.com/Unstructured-IO/unstructured) and [pipeline-sec-filings](https://github.com/Unstructured-IO/pipeline-sec-filings) extensively for accessing the 10-K Filings and cleaning them.

### Narrative Texts
Our aim with the data cleaning pipeline is to identify "Narrative Text", i.e, text which is of consequence to the LLMs or any downstream ML task. We also aim to identify the following main sections of the 10-K filing:
```
Business
Risk Factors
Unresolved Staff Comments
Properties
Legal Proceedings
Mine Safety
Market for Registrant Common Equity
Item 6 is "Reserved"
Management Discussion
Market Risk Disclosures
Financial Statements
Accounting Disagreements
Controls and Procedures
Item 9B is Other Information
Foreign Jurisdictions
Management
Compensation
Principal Stockholders
Related Party Transactions
Accounting Fees
Exhibits
Form Summary
```

Use the `pipeline_api` function in `data_engineering.py`, input the output of the get_10_k function to return the cleaned text (check 'Example Usage" in the file. These are the main sections of the 10-K filing, we use a combination of nltk to tag text as Narrative and regex to identify headlines and titles `(regex query to match title: "(?i)item \d{1,3}(?:[a-z]|\([a-z]\))?(?:\.)?(?::)?"`. Following the method presented by unstructured-IO, we reduce the charecter size by around 100 times, which allows us to feed the data into the LLM's context window.

### The cleaned data looks like this:
| Section  | Element Type | Text                                                                                                                                                                                                                                                                                                  |
|----------|--------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| BUSINESS | NarrativeText| Payable Metal: Ounces or pounds of metal in concentrate payable to the operator after deduction of a percentage of metal in concentrate that is paid to a third-party smelter pursuant to smelting contracts.                                                                                        |
| BUSINESS | NarrativeText| Reserve: That part of a mineral deposit which could be economically and legally extracted or produced at the time of the reserve determination.                                                                                                                                                   |
| BUSINESS | NarrativeText| Royalty: The right to receive a percentage or other denomination of mineral production from a resource extraction operation.                                                                                                                                                                         |
#### Known Issues:
Not all sections are identified meaningfully, however, we are able to extract a large enough corpus of important narrative text to make our analysis significant.

## We use the following tickers listed on NASDAQ for our analysis throughout

- `IBM`
- `AAPL` Apple
- `RGLD` Royal Gold, Inc.

## Using an LLM to "Grade" the text 
The main aim of this project is to utilize LLMs to analyse the 10-K filings. We use the command-R model from Cohere to do this. The model has a 128k context window, large enough for most of our analysis. We ask the LLM to meaningfully analyse the text and generate a rating based on certain criterion. We choose 4 different criteria.

### General Structure of the Prompt (Prompt Engineering)
Here is the example prompt for the confidence rating: A rating for how confident should the user be in the company's future:

```
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
```
Other prompts and their function wrappers can be found in `prompts_api_calls.py`. 

#### Warning:
Utilizing this requires your cohere API key in the python file, the streamlit application is for demonstration <span style="color:red;">only</span> and may be limited by API usage rates. It is recommended to use only the cached tickers as input.

### (Absolute) Scoring Analysis and Dual LLM Approach
These analysis were done in 'absolute' terms, ie, the extracted text from the previous step of a *single* company is entered into the LLM for evaluation. For each of our criteria, we dynamically create two prompts, a regular prompt, and a *strict* prompt. In the strict prompt, the LLM is specifically asked to be extremely critical and strict while evaluating the text.

The following parameters (scoring critera) are used:

#### 1. Confidence Rating

Confidence Rating gives a measure of the company's future prospects. The LLM is instructed to reward robust future planning, growth and and confidence in the company future. 
This rating is useful to future investors and current stakeholders who are most interested in the profit trajectory.

<p align="center">
    <img src="outputs/conf_rating.png" alt="Confidence Rating for Apple" width="550"/>
</p>

#### 2. Environment Rating

Environment Rating gives a measure of the company's effects on the environment. The LLM is instructed to reward feasible planning and commitment to the environment. Companies that don't mention sustainability are penalized. 
This rating is useful to policy makers and Environmentalists who are most interested in the company's outlook towards the environment.

<p align="center">
    <img src="outputs/env_rating.png" alt="Environment Rating for Apple" width="550"/>
</p>

#### 3. Innovation Rating

Innovation Rating gives a measure of the company's commitment towards advancements, specially technological. The LLM is instructed to reward plans to optimize its operations using the latest innovations. The year is also included in the prompt to assist the LLM in making the decision. R&D labs and work mentioned is also rewarded. Companies that make no mention of any future improvements are penalized.
This rating is useful to investors and stakeholders from an operational standpoint, if the company is optimizing itself.

<p align="center">
    <img src="outputs/inno_rating.png" alt="Innovation Rating for Apple" width="550"/>
</p>

#### 4. People Rating

Innovation Rating gives a measure of the company's relationship with its employees. The LLM is instructed to reward companies which mention their talent, efforts to retain them and care for their welfare. The year is also included in the prompt to assist the LLM in making the decision. Companies that make no mention of employees are penalized.
This rating is useful to human rights activists and unions.

<p align="center">
    <img src="outputs/people_rating.png" alt="People Rating for Apple" width="550"/>
</p>

### Plots of all tickers:

<p align="center">
    <img src="outputs/ratings.png" alt="People Rating for Apple" width="750"/>
</p>

##### Plots depicting metadata of each rating:

<p align="center">
  <img src="outputs/heatmap.png" alt="Image 1" width="450" style="display:inline-block; margin-right: 20px;"/>
  <img src="outputs/violin_plot.png" alt="Image 2" width="475" style="display:inline-block;"/>
</p>

### Year-On-Year ratings of the three tickers:
Helpful to see the company priorities and their change every year.

<p align="center">
  <img src="outputs/AAPL_Stack.png" alt="Image 1" width="500" style="display:inline-block; margin-right: 20px;"/>
  <img src="outputs/IBM_stack.png" alt="Image 2" width="500" style="display:inline-block;"/>
</p>
<p align="center">
    <img src="outputs/RGLD_stack.png" alt="Ratings" width="500"/>
</p>

[Citation](https://arxiv.org/abs/2307.13106)

### Relative Scoring Analysis:
We input the text of all the tickers for one particular section and ask the LLM to judge the best one. The LLM thus has a more diverse set of data to assist in its decision making. We record the "Winner" for that particular section.

#### Prompt Used For Comparitive Analysis

```
response = co.chat(
                message=f"""
                You are an AI grader that given an output and a criterion, grades the completion based on
                the prompt and criterion. Below "Excerpt A", "Excerpt B" and "Excerpt B", 
                you must compare both excerpts and output which excerpt is better.

                ## Excerpt A
                {excerpt_1}

                ## Excerpt B
                {excerpt_2}

                ## Excerpt C
                {excerpt_3}

                ## Criterion
                Do not focus on the grammer, instead focus on the overall future plan and robust explainability.
                [Answer with either "A" or "B" or "C".
                A. If Excerpt A is the best, detailed, transparent with robust financials.
                B. If Excerpt B is the best, detailed, transparent with robust financials.
                C. If Excerpt C is the best, detailed, transparent with robust financials.
                .]

                """
```

<p align="center">
  <img src="outputs/comparision.png" alt="Image 1" width="500" style="display:inline-block; margin-right: 20px;"/>
  <img src="outputs/comparision_YOY.png" alt="Image 2" width="500" style="display:inline-block;"/>
</p>
<p align="center">
    <img src="outputs/section_comparision.png" alt="Ratings" width="500"/>
</p>

# Web App
Web Application is available at this [link](https://8501-01hwwe6a9rterwq0zjh91pnspz.cloudspaces.litng.ai/)
Please follow the instructions mentioned. Loading might initially take some time. 

This application was built using Streamlit, and is hosted on a lightning.ai studio. Streamlit as a library is designed for presenting dashboards, integrating it with the storage and compute of a lighting.ai studio, we are able to effectively create a pipeline and store its results within the cloud. 

![image](https://github.com/sulphatet/10-K-Filings-Rating-System/assets/73064995/e0111d5e-5c06-45bc-9205-490597860841)
