import pandas as pd 
import fasttext 

def get_synonyms( file_words= '/workspace/datasets/fasttext/top_words.txt'
                , file_model='/workspace/datasets/fasttext/title_model.bin'
                , file_output='/workspace/datasets/fasttext/synonyms.csv'
                , threshold=0.65):
    with open( file_words ) as top_words:
        ls_words = top_words.read().splitlines()

    model = fasttext.load_model( file_model )

    ls_synonyms = []
    for word in ls_words:
        synonyms = []
        synonyms.append( word )
        for nn in model.get_nearest_neighbors(word):
            if nn[0] >= threshold:
                synonyms.append(nn[1])
        ls_synonyms.append( ",".join(synonyms) )
    
    with open( file_output , mode='wt' ) as output:
        output.write( '\n'.join(ls_synonyms) ) 
    
if __name__ == '__main__':
    get_synonyms()