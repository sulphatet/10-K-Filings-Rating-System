# 10-K-Filings-Rating-System
This repository contains the codebase to pull and clean the data from a SEC 10-K filing of a company, along with functionality to use an LLM based rater. 

# Usage
Replace `<YOUR API HERE>` with your Co:here API key in `prompts_api_calls.py`
Run `streamlit run main.py` to create an interactive window.

# Methodology
## Pulling the 10 - K Filings Data and cleaning it
The function `get_10k_filings_by_ticker_with_years` in `data_engineering.py` pulls the raw data as a string given the ticker.
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




# Web App
Web Application is available at
