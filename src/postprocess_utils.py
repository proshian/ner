import numpy as np
import pandas as pd


# убираем паддинги 
def remove_pad(df, vocab, vocab_lables,all_t,all_p):
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