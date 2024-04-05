import torch
import pandas as pd
from konlpy.tag import Okt

okt = Okt()
VOCAB = pd.read_csv('../data/VOCAB.csv')
MAX_LENGTH = 55
def main(path):
    model = torch.load(path)

    text = input('Enter any words :: \n')
    tokens = okt.morphs(text)

    if len(tokens) > MAX_LENGTH:
        tokens = tokens[:MAX_LENGTH]
    else:
        while len(tokens) < MAX_LENGTH:
            tokens.append('<PAD>')

    for idx in range(len(tokens)):
        token = tokens[idx]
        code = VOCAB[VOCAB[0] == token]
        tokens[idx] = code

    predict = model(text)
    print(predict)

main('./first_model.pkl')
