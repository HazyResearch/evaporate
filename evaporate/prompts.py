############################ SCHEMA ID PROMPTS ############################
SCHEMA_ID_PROMPTS = [
f"""Sample text:
<tr class="mergedrow"><th scope="row" class="infobox-label"><div style="text-indent:-0.9em;margin-left:1.2em;font-weight:normal;">•&nbsp;<a href="/wiki/Monarchy_of_Canada" title="Monarchy of Canada">Monarch</a> </div></th><td class="infobox-data"><a href="/wiki/Charles_III" title="Charles III">Charles III</a></td></tr>
<tr class="mergedrow"><th scope="row" class="infobox-label"><div style="text-indent:-0.9em;margin-left:1.2em;font-weight:normal;">•&nbsp;<span class="nowrap"><a href="/wiki/Governor_General_of_Canada" title="Governor General of Canada">Governor General</a></span> </div></th><td class="infobox-data"><a href="/wiki/Mary_Simon" title="Mary Simon">Mary Simon</a></td></tr>
<b>Provinces and Territories</b class='navlinking countries'>
<ul>
<li>Saskatchewan</li>
<li>Manitoba</li>
<li>Ontario</li>
<li>Quebec</li>
<li>New Brunswick</li>
<li>Prince Edward Island</li>
<li>Nova Scotia</li>
<li>Newfoundland and Labrador</li>
<li>Yukon</li>
<li>Nunavut</li>
<li>Northwest Territories</li>
</ul>

Question: List all relevant attributes about 'Canada' that are exactly mentioned in this sample text if any.
Answer: 
- Monarch: Charles III
- Governor General: Mary Simon
- Provinces and Territories: Saskatchewan, Manitoba, Ontario, Quebec, New Brunswick, Prince Edward Island, Nova Scotia, Newfoundland and Labrador, Yukon, Nunavut, Northwest Territories

----

Sample text:
Patient birth date: 1990-01-01
Prescribed medication: aspirin, ibuprofen, acetaminophen
Prescribed dosage: 1 tablet, 2 tablets, 3 tablets
Doctor's name: Dr. Burns
Date of discharge: 2020-01-01
Hospital address: 123 Main Street, New York, NY 10001

Question: List all relevant attributes about 'medications' that are exactly mentioned in this sample text if any.
Answer: 
- Prescribed medication: aspirin, ibuprofen, acetaminophen
- Prescribed dosage: 1 tablet, 2 tablets, 3 tablets

----

Sample text:
{{chunk:}}

Question: List all relevant attributes about '{{topic:}}' that are exactly mentioned in this sample text if any. 
Answer:"""
]


############################ PROMPTS FOR EXTRACTING A SPECIFIC FIELD BY DIRECTLY GIVING THE MODEL THE CONTEXT ############################
METADATA_EXTRACTION_WITH_LM = [
f"""Here is a file sample:

<th>Location</th>
<td><a href="/wiki/Cupertino">Cupertino</a>, <a href="/wiki/California">California</a>Since 1987</td>

Question: Return the full "location" span of this sample if it exists, otherwise output []. 
Answer: ['Cupertino, California Since 1987']

----

Here is a file sample:

{{chunk:}}

Question: Return the full "{{attribute:}}" span of this sample if it exists, otherwise output [].
Answer:""",
]


METADATA_EXTRACTION_WITH_LM_ZERO_SHOT = [
f"""Sample text:

{{chunk:}}

Question: What is the "{{attribute:}}" value in the text?
Answer:"""
]

EXTRA_PROMPT = [
f"""Here is a file sample:

<a href="/year/2012;price=$550;url=http%www.myname.com;?" target="_blank"></a>

Question: Return the full "price" from this sample if it exists, otherwise output []. 
Answer: ['$550']

----

Here is a file sample:

{{chunk:}}

Question: Return the full "{{attribute:}}" from this sample if it exists, otherwise output [].
Answer:""",
]

METADATA_EXTRACTION_WITH_LM_CONTEXT = [
f"""Here is a file sample:

A. 510(k) Number: 
k143467 

Question: Return the full "510(k) Number" from this sample if it exists and the context around it, otherwise output []. 
Answer: [510(k) Number: k143467]

----

Here is a file sample:

The iphone price increases a lot this there. Each iphone's price is as high as 1000$.

Question: Return the full "price" from this sample if it exists and the context around it, otherwise output []. 
Answer: [Each iphone's price is as high as 1000$]

----

Here is a file sample:

{{chunk:}}

Question: Return the full "{{attribute:}}" from this sample if it exists and the context around it, otherwise output [].
Answer:""",
]

IS_VALID_ATTRIBUTE = [
f"""Question: Could "2014" be a "year" value in a "students" database?
Answer: Yes

----

Question: Could "cupcake" be a "occupation" value in a "employee" database?
Answer: No

----

Question: Could "''" be a "animal" value in a "zoo" database?
Answer: No

----

Question: Could "police officer" be a "occupation" value in a "employee" database?
Answer: Yes

----

Question: Could "{{value:}}" be a "{{attr_str:}}" value in a {{topic:}} database?
Answer:"""
]


PICK_VALUE = [
f"""Examples:
- 32
- 2014
- 99.4
- 2012

Question: Which example is a "year"?
Answer: 2012, 2014

----

Examples:
- police officer
- occupation

Question: Which example is a "occupation"?
Answer: police officer

----

Examples:
{{pred_str:}}

Question: Which example is a "{{attribute:}}"?
Answer:"""
]


PICK_VALUE_CONTEXT = [
f"""Here are file samples:

-The purpose for submission is to obtain substantial equivalence determination for the illumigene HSV 1&2 DNA Amplification Assay.
-The purpose for submission of this document is not specified in the provided sample.
-The purpose for submission of this file is not specified.

Question: Extract "the purpose for submission" from the right sample , otherwise output []. 
Answer: to obtain substantial equivalence determination for the illumigene HSV 1&2 DNA Amplification Assay

----

Here are file samples:

{{pred_str:}}

Question: Return the full "{{attribute:}}" from this sample if it exists, otherwise output [].
Answer:""",
]



############################## PROMPTS TO GENERATE FUNCTIONS THAT PARSE FOR A SPECIFIC FIELD ##############################
METADATA_GENERATION_FOR_FIELDS = [
    # base prompt
f"""Here is a sample of text:

{{chunk:}}


Question: Write a python function to extract the entire "{{attribute:}}" field from text, but not any other metadata. Return the result as a list.


import re

def get_{{function_field:}}_field(text: str):
    \"""
    Function to extract the "{{attribute:}} field". 
    \"""
    """, 
    
    # prompt with flexible library imports
f"""Here is a file sample:

DESCRIPTION: This file answers the question, "How do I sort a dictionary by value?"
DATES MODIFIED: The file was modified on the following dates:
2009-03-05T00:49:05
2019-04-07T00:22:14
2011-11-20T04:21:49
USERS: The users who modified the file are:
Jeff Jacobs
Richard Smith
Julia D'Angelo
Rebecca Matthews
FILE TYPE: This is a text file.

Question: Write a python function called "get_dates_modified_field" to extract the "DATES MODIFIED" field from the text. Include any imports.

import re

def get_dates_modified_field(text: str):
    \"""
    Function to extract the dates modified.
    \"""
    parts= text.split("USERS")[0].split("DATES MODIFIED")[-1]
    pattern = r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
    return re.findall(pattern, text)

----

Here is a file sample:

<title>U.S. GDP Rose 2.9% in the Fourth Quarter After a Year of High Inflation - WSJ</title>
<meta property="og:url" content="https://www.wsj.com/articles/us-gdp-economic-growth-fourth-quarter-2022-11674683034"/>
<meta name="article.published" content="2023-01-26T10:30:00Z"/><meta itemProp="datePublished" content="2023-01-26T10:30:00Z"/>
<meta name="article.created" content="2023-01-26T10:30:00Z"/><meta itemProp="dateCreated" content="2023-01-26T10:30:00Z"/>
<meta name="dateLastPubbed" content="2023-01-31T19:17:00Z"/><meta name="author" content="Sarah Chaney Cambon"/>

Question: Write a python function called "get_date_published_field" to extract the "datePublished" field from the text. Include any imports.

from bs4 import BeautifulSoup

def get_date_published_field(text: str):
    \"""
    Function to extract the date published.
    \"""
    soup = BeautifulSoup(text, parser="html.parser")
    date_published_field = soup.find('meta', itemprop="datePublished")
    date_published_field = date_published_field['content']
    return date_published_field

----

Here is a sample of text:

{{chunk:}}

Question: Write a python function called "get_{{function_field:}}_field" to extract the "{{attribute:}}" field from the text. Include any imports."""
]


class Step:
    def __init__(self, prompt) -> None:
        self.prompt = prompt

    def execute(self):
        pass


