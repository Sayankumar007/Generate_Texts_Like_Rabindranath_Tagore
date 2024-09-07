import os
import re
import random

def clean_data(text):
    whitespace = re.compile(u"[\s\u0020\u00a0\u1680\u180e\u202f\u205f\u3000\u2000-\u200a]+", re.UNICODE)
    punctSeq = u"['\"“”‘’]+|[.…-]+|[:;]+"
    bengali_numeral_pattern = r'[১-৯০]+' 
    # bangla_fullstop = u"\u0964"                     # in case you don't need it..
    

    corpus = re.sub(bengali_numeral_pattern," ",text)
    corpus = re.sub(r'\s*\d+\s*', ' ', corpus)

    corpus = re.sub('\n'," ",corpus)
    corpus = re.sub(punctSeq," ",corpus)
    corpus = whitespace.sub(" ",corpus).strip()
    
    return corpus




def load_data(input_folder, output_file):

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write("")
    
    # Read all files from the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            print(f'processing "{filename}"  ...')
            with open(os.path.join(input_folder, filename), 'r', encoding='utf-8') as file:
                text = file.read()
                # Preprocess the text
                cleaned_text = clean_data(text)
    
                # Save the combined text to the output file
                with open(output_file, 'a', encoding='utf-8') as file:
                    file.write(cleaned_text)
                    
    




# Set the input folder and output file paths
input_folder = './dataset'
output_file = './data.txt'

# Read, preprocess, and save the texts
load_data(input_folder, output_file)

# sample use of data.txt
with open(output_file, 'r', encoding='utf-8') as file:
    corpus = file.read()

# printing random 500 samnples from the corpus..
idx = random.randint(0, len(corpus)-500)
print(f'\n\nSample corpus -> \n\n{corpus[idx: idx+500]}')