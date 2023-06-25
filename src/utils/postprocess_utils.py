import numpy as np
import pandas as pd
TAG = ['B-bin', 'I-bin', 'B-soc','I-soc','B-met', 'I-met','B-cmp','I-cmp','B-eco', 'I-eco','B-inst','I-inst','B-act','I-act','B-qua','I-qua']


def remove_pad(df, vocab, vocab_lables,all_t, all_p):
    """
    убираем паддинги
    """
    data = {'sentence': df['sentence'], 
            'real_cat': np.array_split(all_t, len(df['sentence'])), 
            'pred_cat':np.array_split(all_p, len(df['sentence']))}
    df_test_see = pd.DataFrame(data)
    for ind in df_test_see.index:
        sent = df_test_see['sentence'][ind]
        try:
            len_pad = sent.index(next(filter(lambda x: x!=0, sent)))
        except StopIteration:
            pass
        sent = sent[len_pad:]
        sent = vocab.lookup_tokens(sent)
        df_test_see.at[ind, 'sentence'] = sent 
        sent = df_test_see['real_cat'][ind]
        sent = sent[len_pad:]
        sent = vocab_lables.lookup_tokens(sent)
        df_test_see.at[ind, 'real_cat'] = sent 
        sent = df_test_see['pred_cat'][ind]
        sent = sent[len_pad:]
        sent = vocab_lables.lookup_tokens(sent)
        df_test_see.at[ind, 'pred_cat'] = sent 
    return df_test_see


def make_show(row,lbl):
    """
    function for vizualize models reusults on test data
    inputs:
    row - a sentence, 
    lbl - true markup or prediction markup
    return text and spans 
    """
    spans = []
    coords = 0
    c1 = 0
    c2 = 0
    tag = ''
    text = ''
    last = 0
    for i in range(len(row)):
        text += row[i]+' '
        if lbl[i] in TAG:
            if 'B-' in lbl[i]:
                c1 = coords
                c2 = coords + len(row[i])
                tag = lbl[i][2:].upper()
                coords+=1+len(row[i])
            elif 'I-' in lbl[i]:
                c2+=1+len(row[i])
                coords+=1+len(row[i])
        elif lbl[i] =='O':
            coords += len(row[i])+1
        if (c1,c2,tag) !=(0, 0, ''):
            if last == c1:
                spans = spans[:-1]
            spans.append((c1,c2,tag))
            last = c1
    return text, spans
