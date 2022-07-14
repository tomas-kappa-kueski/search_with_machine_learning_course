import os
import argparse
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import csv

import re

# Useful if you want to perform stemming.
import nltk
stemmer = nltk.stem.PorterStemmer()

categories_file_name = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

queries_file_name = r'/workspace/datasets/train.csv'
#output_file_name = r'/workspace/datasets/labeled_query_data.txt'
output_file_name = r'/workspace/datasets/fasttext/labeled_queries.txt'

parser = argparse.ArgumentParser(description='Process arguments.')
general = parser.add_argument_group("general")
general.add_argument("--min_queries", default=1,  help="The minimum number of queries per category label (default is 1)")
general.add_argument("--output", default=output_file_name, help="the file to output to")

args = parser.parse_args()
output_file_name = args.output

if args.min_queries:
    min_queries = int(args.min_queries)

# The root category, named Best Buy with id cat00000, doesn't have a parent.
root_category_id = 'cat00000'

tree = ET.parse(categories_file_name)
root = tree.getroot()

# Parse the category XML file to map each category id to its parent category id in a dataframe.
categories = []
parents = []
for child in root:
    id = child.find('id').text
    cat_path = child.find('path')
    cat_path_ids = [cat.find('id').text for cat in cat_path]
    leaf_id = cat_path_ids[-1]
    if leaf_id != root_category_id:
        categories.append(leaf_id)
        parents.append(cat_path_ids[-2])
parents_df = pd.DataFrame(list(zip(categories, parents)), columns =['category', 'parent'])

# Read the training data into pandas, only keeping queries with non-root categories in our category tree.
df = pd.read_csv(queries_file_name)[['category', 'query']]
df = df[df['category'].isin(categories)]

# IMPLEMENT ME: Convert queries to lowercase, and optionally implement other normalization, like stemming.
def normalization(text, apply_stemmer=True):
    # normalization
    text_norm = re.sub('\W+',' ', text.strip().replace('_','').replace(',','')
        ).lower().translate(str.maketrans('âáàãêéíóôõúüñç','aaaaeeiooouunc') )
    # Apply Snowball stemmer
    if apply_stemmer:
        text_norm = ' '.join( map( stemmer.stem, text_norm.split(' ') ) )
    return text_norm

# IMPLEMENT ME: Roll up categories to ancestors to satisfy the minimum number of queries per category.
df['query'] = df['query'].map( normalization )
df_cat_count = df.groupby('category').size().reset_index(name='cat_count')
# Calculate categorys to be grouped because the lower number of querys
category_lower_min = len(df_cat_count[ df_cat_count['cat_count']< min_queries ])

while category_lower_min >0:
    # join all
    df = df[['query','category']]
    df = df.merge(df_cat_count, how='inner', on=['category'])
    df = df.merge(parents_df, how='inner', on=['category'])
    # paste the parent when the nb of querys are low
    df['category' ][ df['cat_count']<min_queries ] = df[ df['cat_count']<min_queries ]['parent']
    df = df[df['category'].isin(categories)]
    # recalculate query count per category
    df_cat_count = df.groupby('category').size().reset_index(name='cat_count')
    category_lower_min = len(df_cat_count[ df_cat_count['cat_count']< min_queries ])


# Create labels in fastText format.
df['label'] = '__label__' + df['category']

# Output labeled query data as a space-separated file, making sure that every category is in the taxonomy.
df = df[df['category'].isin(categories)]
df['output'] = df['label'] + ' ' + df['query']
df[['output']].to_csv(output_file_name, header=False, sep='|', escapechar='\\', quoting=csv.QUOTE_NONE, index=False)
