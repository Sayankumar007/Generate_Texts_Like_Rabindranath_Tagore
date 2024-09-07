import torch
import torch.nn.functional as F
# import torchtext
import random
from nltk.tokenize import word_tokenize
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import TensorDataset

class TextPreProcessor:
    def __init__(self, file_path, unk_rate=0.1, specials=['<pad>', '<unk>']):
        
        print(f'Initializing TextPreProcessor.. \nReading file... {file_path}')
        with open(file_path, 'r', encoding='utf-8') as file:
            self.corpus = file.read()
            
        self.doc = self.tokenize()

        self.unk_rate = unk_rate
        self.specials = specials
        self.word_vocab = self.build_vocab(special_first=True)
        
        print("Creating the word-to-index map...")
        self.word_to_index = {token: idx for idx, token in enumerate(self.word_vocab.get_itos())}

        print("Creating the index-to-word map...")
        self.index_to_word = {idx: token for idx, token in enumerate(self.word_vocab.get_itos())} 
        
        self.total_words = len(self.word_vocab)


    def tokenize(self):
        bangla_fullstop = u"\u0964"  # unicode code for bengali fullstop
        print(f'\nTokenizing Corpus of Length {len(self.corpus)}...')
        
        # Tokenize sentences based on the Bengali fullstop symbol
        sentences = self.corpus.split(bangla_fullstop)
        
        # Remove empty strings and add the period symbol back to each sentence
        sentences = [sentence.strip() + " " + bangla_fullstop for sentence in sentences if sentence.strip()]
        
        # Tokenize each sentence into words
        doc = [word_tokenize(sentence) for sentence in sentences]
        
        print("\n\nBefore Tokenization : ", self.corpus[-386:],"\n")
        print("After Tokenization : ", doc[-3:])
        
        return doc
        
 

       
    def build_vocab(self, special_first=True):
        specials = self.specials if special_first else []
        print("\nBuilding Vocabulary... ")
        word_vocab = build_vocab_from_iterator(
            self.doc,
            min_freq=1,
            specials=specials,
        )
        print(f'\n\nLength of Vocabulary : {len(word_vocab)}\nSample Vocabulary --> {list(word_vocab.get_itos())[:10]}')
        return word_vocab



    def seq2grams(doc):
        n_grams = []
        print(f'Converting Document to n_grams...')
        for sentence in doc:            # for each sentence in the corpus
            for i in range(1, len(sentence)):  # from [1st] word, [1st,2nd] word, [1st,2nd,3rd] word upto last word inde 
                sequence = sentence[:i+1]      # make sequences [1,2], [1,2,3], [1,2,3,4] and so on
                n_grams.append(sequence)        # add the sequence to the main array
        return n_grams



    def add_random_unk_tokens(self, ngram):
        for idx, word in enumerate(ngram[:-1]):
            if random.uniform(0, 1) < self.unk_rate:
                ngram[idx] = '<unk>'
        return ngram
    
    

    def text_to_numerical_sequence(self, tokenized_text):
        tokens_list = []
        if tokenized_text[-1] in self.word_to_index:
            for token in tokenized_text[:-1]:
                num_token = self.word_to_index[token] if token in self.word_to_index  else self.word_to_index['<unk>']
                tokens_list.append(num_token)
            num_token = self.word_to_index[tokenized_text[-1]]
            tokens_list.append(num_token)
            return tokens_list
        return None



    def create_dataset(self):
        
        document = self.seq2grams(self.doc)
        print(f'\n\nSample Document after n-gram -->\n{document[:5]}')
        
        document_unk = []
        print(f'Inserting <unk> token in the n_gram document randomly...')
        for data in document:
            document_unk.append(self.add_random_unk_tokens(data))
            
        
        print(f'\n\nSample n-Gram with random <unk> tokens -->\n{document_unk[:5]}')
        
        # Efficiently create data list without redundant calls
        data = [seq for seq in (self.text_to_numerical_sequence(sequence) for sequence in document_unk)]

        print(f'\n\nTotal input sequences: {len(data)}\n')
        print('Sample document :\n', data[7:9])
        
        
        print(f'\n\nCreating Dataset From the Document Corpus...')
        X = [sequence[:-1] for sequence in data]
        y = [sequence[-1] for sequence in data]
        
        
        longest_sequence = max(len(sequence) for sequence in X)
        
        
        print(f'\n\nPadding Features to Maximum length {longest_sequence} ...')
        padded_X = [F.pad(torch.tensor(sequence), (longest_sequence - len(sequence),0), value=0) for sequence in X]
        
        print(f'\n\nConverting Data into torch Tensors...')
        padded_X = torch.stack(padded_X)
        y = torch.tensor(y)

        
        dataset = TensorDataset(padded_X, y)
        
        

        return dataset, self.total_words, longest_sequence