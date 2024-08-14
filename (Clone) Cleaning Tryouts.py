# Databricks notebook source
# MAGIC %pip install --quiet beautifulsoup4==4.9.3 dnspython==1.16.0 geoip2==4.1.0 html2text==2020.1.16 lxml==4.6.1 scikit-learn==1.0.2 tldextract==3.0.2 ua-parser==0.8.0 unidecode==1.1.1 /dbfs/FileStore/jars/5ee713c5_da3c_4d4e_a9f5_420dc4785e5c/fraudlib-3.0.19-py3-none-any.whl

# COMMAND ----------

import re
from lib.preprocess.email import get_body, get_subject, BodyConfig, REPLY1_REGEXES, REPLY2_REGEXES, extract_html_text_v2

# COMMAND ----------

import os
from IPython.core.display import HTML
from functools import reduce
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
import numpy as np
from difflib import SequenceMatcher
from tqdm.notebook import tqdm
import pandas as pd
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import FloatType, IntegerType, StructField, StructType, ArrayType, StringType, BooleanType
from pyspark.sql.functions import lit, struct


# COMMAND ----------

import fasttext
from collections import Counter

# COMMAND ----------

PHISHING_ENG_BUCKET = 'dbfs:/FileStore/claire/pre_gpt/english/phishing_score_1_english'
BEC_ENG_BUCKET = 'dbfs:/FileStore/claire/pre_gpt/english/bec_score_1_english'
BLACKMAIL_ENG_BUCKET = 'dbfs:/FileStore/claire/pre_gpt/english/blackmail_score_1_english'
SPAM_ENG_BUCKET = 'dbfs:/FileStore/claire/pre_gpt/english/spam_score_1_english'


# COMMAND ----------

bc = BodyConfig()

# COMMAND ----------

type(dir(bc)[0])

# COMMAND ----------

for atr in dir(bc):
  print(atr)
  # print(f"\t {bc.atr}")

# COMMAND ----------

# DBTITLE 1,Check bc attribute types
types = set()
for atr, v in bc.__dict__.items():
  # print(atr,":", v)

  # if type(v) is bool:
  #   print(atr,":", v)
  types.add(type(v))
  
print(types)
print()

for t in types:
  print(t)
  print()
  print('-'*20)
  for atr, v in bc.__dict__.items():
    if type(v) is t:
      print(atr,":", v)
      print('-'*20)
  print("="*20)

# COMMAND ----------

bc.notices_regexes

# COMMAND ----------

bc.signature_name_clean_regex

# COMMAND ----------

get_body_udf = udf(get_body)

# COMMAND ----------

@udf(IntegerType())
def count_absolute_length(text):
  return len(text.split())

# COMMAND ----------

bc = BodyConfig()

# COMMAND ----------

# MAGIC %md
# MAGIC # Phishing

# COMMAND ----------

phishing = spark.read.parquet(PHISHING_ENG_BUCKET).cache()
# phishing.count()

# COMMAND ----------

a = phishing.sample(fraction=0.01).collect()

# COMMAND ----------

# DBTITLE 1,Spot Check Examples
print(len(a))
for i,row in enumerate(a):
  print('INDEX:', i)
  print('detection')
  # print(f"\t {row['fraud_info']['taxonomy']}, {row['fraud_info']['classifier']}, {row['fraud_info']['score']}")
  print(f"\t {row['fraud_info_taxonomy']}, {row['fraud_info_classifier']}, {row['fraud_info_score']}")

  # print('sender')
  # # print(row)""
  # # if len(row['email']['sender']) > 0:
  # print(f"\t Email: {row['email_sender_emailAddress_address']}")
  # print(f"\t Name:  {row['email_sender_emailAddress_name']}")

  # print('from')
  # # if len(row['email']['from']) > 0:
  # # print(row)
  # print(f"\t Email: {row['email_from_emailAddress_address']}")
  # print(f"\t Name:  {row['email_from_emailAddress_name']}")

  # print('replyTo')
  # # print(row)
  # if len(row['email_replyTo_emailAddress_address']) > 0:
  #   num = len(row['email_replyTo_emailAddress_address'])
  #   for j in range(num):
  #     print(f"\t Email: {row['email_replyTo_emailAddress_address'][j]}")
  #     print(f"\t Name:  {row['email_replyTo_emailAddress_name'][j]}")

  # print('to')
  # if len(row['email_toRecipients_emailAddress_address']) > 0:
  #   num = len(row['email_toRecipients_emailAddress_address'])
  #   for j in range(num):
  #     print(f"\t Email: {row['email_toRecipients_emailAddress_address'][j]}")
  #     print(f"\t Name:  {row['email_toRecipients_emailAddress_name'][j]}")

  # print('cc')
  # if len(row['email_ccRecipients_emailAddress_address']) > 0:
  #   num = len(row['email_ccRecipients_emailAddress_address'])
  #   for j in range(num):
  #     print(f"\t Email: {row['email_ccRecipients_emailAddress_address'][j]}")
  #     print(f"\t Name:  {row['email_ccRecipients_emailAddress_name'][j]}")

  # print('subject')
  # print(f"\t {row['email_subject']}")

  # print('text_length')
  # print(f"\t {row['text_length']}")

  # print('attachments')
  # # print(row)
  # if row['email_hasAttachments']:
  #   num = len(row['email_attachments_name'])
  #   for j in range(num):
  #     print(f"\t {row['email_attachments_name'][j]}")

  print('body_original')
  # print(row)
  display(HTML(f"\t {row['email_body_content']}"))
  print()

  print('body_wei')
  print('-'*20)
  print(row['raw_text'])
  print('='*20)
  print()

  print('body_barracuda')
  barr_text = get_body(row['email_body_content'])

  print("len(barr_text) =", len(barr_text))
  print('-'*20)
  print(barr_text[0])



  print("#"*1000)

# COMMAND ----------

# DBTITLE 1,Spot Check English Detection
non_eng = {
  'lang':[38],
  'gibberish':[0, 30, 39, 56]
}

model = fasttext.load_model('/dbfs/FileStore/claire/models/lid_176.bin')

def check_english(text, model=None):
  if model is None:
    model = fasttext.load_model('/dbfs/FileStore/claire/models/lid_176.bin')
  pred = model.predict(text.replace('\n','')) #(('__label__en',), array([0.98101348]))
  label = pred[0][0]
  score = float(pred[1][0])
  return label, score

for k, indices in non_eng.items():
  for i in indices:
    row = a[i]
    text = get_body(row['email_body_content'])[0]
    print(text)
    print('Language:', check_english(text, model))
    print()


# COMMAND ----------

rand = phishing.sample(fraction=0.01).collect()

# COMMAND ----------

# DBTITLE 1,Add [link] back to message
print(len(rand))
for i,row in enumerate(rand):
  print('INDEX:', i)
  print('detection')
  # print(f"\t {row['fraud_info']['taxonomy']}, {row['fraud_info']['classifier']}, {row['fraud_info']['score']}")
  print(f"\t {row['fraud_info_taxonomy']}, {row['fraud_info_classifier']}, {row['fraud_info_score']}")

  print('body_original')
  # print(row)
  # display(HTML(f"\t {row['email_body_content']}"))
  print()

  print('body_barracuda')
  barr_text = get_body(row['email_body_content'])

  print("len(barr_text) =", len(barr_text))
  print('-'*20)
  print(barr_text[0])
  print('-'*20)
  print(barr_text[1])

  # row['barr_text'] = barr_text[0]
  # row['barr_link'] = barr_text[1]

  barr_text_linked = barr_text[0]

  for l in barr_text[1]:
    # print(l, type(l))
    barr_text_linked = barr_text_linked.replace(l.text, l.text+' [Link]')

  print(barr_text_linked)

  print("#"*200)

# COMMAND ----------

# DBTITLE 1,Don't remove threads?
# bc.signature_name_removal = True # Not this
# bc.closing_lines_per_check = 100 # Not this

print(len(rand))
for i,row in enumerate(rand):
  print('INDEX:', i)
  print('detection')
  # print(f"\t {row['fraud_info']['taxonomy']}, {row['fraud_info']['classifier']}, {row['fraud_info']['score']}")
  print(f"\t {row['fraud_info_taxonomy']}, {row['fraud_info_classifier']}, {row['fraud_info_score']}")

  print('body_original')
  # print(row)
  display(HTML(f"\t {row['email_body_content']}"))
  print()

  print('body_barracuda')
  barr_text = get_body(row['email_body_content'], config=bc)

  print("len(barr_text) =", len(barr_text))
  print('-'*20)
  print(barr_text[0])
  print('-'*20)
  print(barr_text[1])

  print("#"*200)

# COMMAND ----------

# DBTITLE 1,Detect the Thread first?
REPLY1_REGEXES = [re.compile(r'(?:^[\*> ]*?(?:From|Sent|To|Cc|Subject):\*?[ ]?.+\n){2,5}',
 re.IGNORECASE|re.MULTILINE|re.UNICODE),
 re.compile(r'(?:^[\*> ]*?(?:Von|Datum|An|Betreff):\*?[ ]?.+\n){2,5}',
 re.IGNORECASE|re.MULTILINE|re.UNICODE),
 re.compile(r'(?:^[\*> ]*?(?:Van|Datum|Aan|Onderwerp):\*?[ ]?.+\n){2,5}',
 re.IGNORECASE|re.MULTILINE|re.UNICODE),
 re.compile(r'(?:^[\*> ]*?(?:De|Date|À|Objet)[ \xa0]?:\*?[ ]?.+\n){2,5}',
 re.IGNORECASE|re.MULTILINE|re.UNICODE),
 re.compile(r'(?:^[\*> ]*?(?:De|Fecha|Para|Asunto):\*?[ ]?.+\n){2,5}',
 re.IGNORECASE|re.MULTILINE|re.UNICODE),
 re.compile(r'(?:^[\*> ]*?(?:Da|Data|A|Oggetto):\*?[ ]?.+\n){2,5}',
 re.IGNORECASE|re.MULTILINE|re.UNICODE),
 re.compile(r'(?:^[\*> ]*?(?:De|Data|Para|Assunto):\*?[ ]?.+\n){2,5}',
 re.IGNORECASE|re.MULTILINE|re.UNICODE),
 re.compile(r'(?:^[\*> ]*?(?:Från|Skickat|Till|Kopia|Ämne):\*?[ ]?.+\n){2,5}',
 re.IGNORECASE|re.MULTILINE|re.UNICODE),
 re.compile(r'(?:^[\*> ]*?(?:发件人|日期|收件人|主题):\*?[ ]?.+\n){2,5}',
 re.IGNORECASE|re.MULTILINE|re.UNICODE),
 re.compile(r'(?:^[\*> ]*?(?:差出人|日付|宛先|件名):\*?[ ]?.+\n){2,5}',
 re.IGNORECASE|re.MULTILINE|re.UNICODE)]

# COMMAND ----------

REPLY2_REGEXES = [re.compile(r'^[> ]*?(?!On.*?On\s.+?wrote:)(On\s(.+?)wrote:)',
 re.IGNORECASE|re.UNICODE|re.MULTILINE),
 re.compile(r'^[> ]*?(?!Am.*?Am\s.+?schrieb:)(Am\s(.+?)schrieb:)',
 re.IGNORECASE|re.UNICODE|re.MULTILINE),
 re.compile(r'^[> ]*?(?!Op.*?Op\s.+?geschreven:)(Op\s(.+?)geschreven:)',
 re.IGNORECASE|re.UNICODE|re.MULTILINE),
 re.compile(r'^[> ]*?(?!Le.*?Le\s.+?a écrit[ \xa0]?:)(Le\s(.+?)a écrit[ \xa0]?:)',
 re.IGNORECASE|re.UNICODE|re.MULTILINE),
 re.compile(r'^[> ]*?(?!El.*?El\s.+?escribió:)(El\s(.+?)escribió:)',
 re.IGNORECASE|re.UNICODE|re.MULTILINE),
 re.compile(r'^[> ]*?(?!Il giorno.*?Il giorno\s.+?ha scritto:)(Il giorno\s(.+?)ha scritto:)',
 re.IGNORECASE|re.UNICODE|re.MULTILINE),
 re.compile(r'^[> ]*?(?!Em.*Em\s.+?escreveu:)(Em\s(.+?)escreveu:)',
 re.IGNORECASE|re.UNICODE|re.MULTILINE),
 re.compile(r'^[> ]*?(?!在.*?在\s.+?写入:)(在\s(.+?)写入:)',
 re.IGNORECASE|re.UNICODE|re.MULTILINE),
 re.compile(r'^\d{2,4}.+?\sを書き込みました:', re.IGNORECASE|re.UNICODE),
 re.compile(r'^[> ]*?begin forwarded message:[ ]*?$',
 re.IGNORECASE|re.UNICODE|re.MULTILINE),
 re.compile(r'^[> ]*?anfang der weitergeleiteten nachricht:[ ]*?$',
 re.IGNORECASE|re.UNICODE|re.MULTILINE)]

# COMMAND ----------

REPLY_REGEX = REPLY1_REGEXES + REPLY2_REGEXES
len(REPLY_REGEX), type(REPLY_REGEX[0])
REPLY_REGEX

# COMMAND ----------

def detect_thread(email, regex_list=REPLY_REGEX):
  thread = False
  for r in regex_list:
    if re.search(r, email) is not None:
      thread = True
      break
  return thread

# COMMAND ----------

# FN: 0, 45, 56
e = rand[56]['email_body_content']
e = extract_html_text_v2(e)

print(detect_thread(e))
for r in REPLY1_REGEXES+REPLY2_REGEXES:
  print(r)
  print(re.findall(r, e))
  # print(re.search(r, e))

# print(e)
# re.findall(r'(?:^[\*> ]*?(?:From|Sent|To|Cc|Subject):\*?[ ]?.+\n){2,5}', e, re.IGNORECASE|re.MULTILINE|re.UNICODE)
# re.findall(r'\A(?:^[\*> ]*?(?:From|Sent|To|Cc|Subject):\*?[ ]?.+\n){2,5}', e, re.IGNORECASE|re.MULTILINE|re.UNICODE)
# re.findall(r'^[> ]*?(?!On.*?On\s.+?wrote:)(On\s(.+?)wrote:)', e ,re.IGNORECASE|re.UNICODE|re.MULTILINE)
# get_body(e)

# COMMAND ----------

prin

# COMMAND ----------

# DBTITLE 1,Spot check thread detections
# Detect the threads first
len(rand)

for i,row in enumerate(rand):
  print('INDEX:', i)
  print('detection')
  # print(f"\t {row['fraud_info']['taxonomy']}, {row['fraud_info']['classifier']}, {row['fraud_info']['score']}")
  print(f"\t {row['fraud_info_taxonomy']}, {row['fraud_info_classifier']}, {row['fraud_info_score']}")

  print('body_original')
  # print(row)
  display(HTML(f"\t {row['email_body_content']}"))
  print()
  
  # Detect using Regex
  thread = detect_thread(email=extract_html_text_v2(row['email_body_content']))
  print("EMAIL IS THREAD:", thread)
  print('#'*300)

# COMMAND ----------

email = rand[6]['email_body_content']

# COMMAND ----------

print(a)
print(b)

# COMMAND ----------

phishing = phishing.withColumn('raw_text_barr', get_body_udf('email_body_content'))
