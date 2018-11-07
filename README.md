# Company Upstream and Downstream Analysis


## Background

- Listed companies regularly disclose annual reports in each quarter to disclose the company's operating conditions. But these reports are too long for humans to read and extract effective information. So using some technology of data mining is very popular nowadays for people who want to have better knowledge of listed companies.
- Most listed companies have their upstream industry and downstream industry. What’s more, their industries can have a huge effect on these companies, so it’s important to know upstream and downstream industries of each company. When we have that information, we can have a better analysis of the factors that influence the company’s profits and help us predict business revenue of certain companies.

## Requirement Analysis

- First, we need to build a dictionary that contains vocabularies that represents certain kinds of industries. With the dictionary, we can get words of industries from annual reports, this is the prerequisite of our analysis.
- Then, we need to identify whether these words are actually words that represent the company’s upstream and downstream industries.
- Finally, show results in a file or on a website.

## Solution
- First, we use word2vec as our basic technology to map words to vectors in order to do classification and other calculation, more will be illustrated later.
- We will use some corpus to collect words that represent industry. I’ll do manual annotation and then use SVM to classify words.
- Then, I will use LSTM to judge whether the word represents upstream or downstream industries.

## Running Environment
- Python 3.6
- Django

## Technology Used
- Word2vec:
- SVM:
- LSTM:
## Experiment

## To Be Finished

## Code in [BusinessInteligenceFinalWork](https://github.com/caozixuan/BusinessInteligenceFinalWork)