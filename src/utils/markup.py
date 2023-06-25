import pandas as pd
import os
import tqdm
import re
from typing import List
import numpy as np

# имена сущностей
NAMES = ['BIN', 'SOC', 'MET', 'CMP', 'ECO', 'INST', 'ACT', 'QUA']
# тэги для разметки
TAG = ['B-bin', 'I-bin', 'B-soc','I-soc','B-met', 'I-met','B-cmp','I-cmp','B-eco', 'I-eco','B-inst','I-inst','B-act','I-act','B-qua','I-qua']

def make_data(path: str, filenames: List[str]) -> pd.DataFrame:
    """
    Creates dataframe with word sequences and corresponding IOB tags

    args:
        path: path to dir with .txt and .ann files
        filenames: list of filenames without extensions from `path` directory
            that will be used to create a dataset. Usually it's a list of all
            unique filenames without extensions, but it may be a subset
    returns:
        dataframe with two columns
            First column: a sequence of words.
            Second column: sequence of IOB tags
    """
    res_df = pd.DataFrame(columns = ['word', 'tag'])
    for filename in tqdm(filenames):
        df1 = make_ann(filename, path)
        df2 = make_text(filename,df1,path)
        res_df = pd.concat([res_df, df2], ignore_index=True)
    return res_df


# продготовка аннотаций
def make_ann(filename: str, path: str, on_bad_lines: str = "skip") -> pd.DataFrame:
    """
    Creates an annotation dataframe for {filename}.txt with columns:
        words: words sequence 
        coords: list of two numbers: first and last charracter
            indexes of named entity in {filename}.txt
        class: named entity type
        
        также есть другие столбцы, но они вроде бы не используются
    
    args:
        path: path to dir with .txt and .ann files
        filename: filename without extension
        on_bad_lines: принимает значения из {'error', 'warn', 'skip'}
            Благодаря on_bad_lines='skip' пропустим строки исходного .ann файла,
            в которых число элементов не соотвествует количеству столбцов.
    returns: 
        annotation dataframe for {filename}.txt
    """
    filepath = os.path.join(path, f"{filename}.ann")

    ann_df = pd.read_csv(filepath, sep='\t', engine='python',
                         header=None, on_bad_lines=on_bad_lines)
    # Дадим столбцам названия.
    ann_df.rename(columns = {1:'class_and_coords', 2:'words'}, inplace = True )
    # Разделим в разные столбцы классы и координаты.
    # ann_df.insert(2, 'coords' , ann_df['class'])  # Я не понял, зачем это нужно
    ann_df['class'] = ann_df['class_and_coords'].apply(lambda x: x.split(" ")[0])
    ann_df['coords'] = ann_df['class_and_coords'].apply(lambda x: x.split(" ")[1:])
    ann_df = ann_df.drop('class_and_coords', axis = 1)

    # Удалим строки с аннтоациями отношений
    ann_df = ann_df.dropna()
    ann_df.reset_index(drop= True , inplace= True )

    # Чистка текстовых данных?
    ann_df['words'] = ann_df['words'].apply(lambda x: my_split(x.split(" ")))
    ann_df['words'] = ann_df['words'].apply(lambda x: del_all(x))
    ann_df['words'] = ann_df['words'].apply(lambda x: [item.strip() for item in x if item not in ['','»', '«',':',' ']])

    # coords 1 нужен для сортировки
    ann_df.insert(2, 'coords1' , ann_df['coords'])
    ann_df['coords1'] = ann_df['coords1'].apply(lambda x: int(x[0]))
    ann_df = ann_df.sort_values(by='coords1')
    ann_df.reset_index(drop= True , inplace= True )
    return ann_df


def make_text(file, df, path):
    """
    делаем разметку данных и чистим их
    
    """
    # открываем файл и записываем его в dataframe
    with open(path+'/'+file+'.txt') as f:
        lines = f.readlines()
    text_df = pd.DataFrame({'word':lines})
    
    # считаем длины строк для удобства дальнейшей разметки
    text_df.insert(1, 'len' , text_df['word'].copy())
    text_df['len'] = text_df['len'].apply(lambda x: len(x)) # считаем длину строки
    new_lens = [text_df['len'][0]] # считаем длину предыдущих строк + длина новой строки
    for i in range(1, len(list(text_df['len']))):
        new_lens.append(sum(list(text_df['len'])[:i+1]))
    text_df.insert(2, 'new_len' , new_lens)
    
    # добавляем столбец для разметки
    text_df.insert(3, 'tag', 0)
    
    # удаляем \n
    text_df['word'] = text_df['word'].apply(lambda x: re.split('\n',x)[0])
    #удаляем строрки с []
    idx = [i for i in range(len(text_df)) if len(text_df['word'][i])==0]
    text_df = text_df.drop(index=idx)
    text_df.reset_index(drop = True, inplace= True)
    
    # делаем разметку
    # находим индекс строки для каждой аннотации
    df = find_rows(text_df, df)
    
    # преобразовываем строку в массив
    text_df['word'] = text_df['word'].apply(lambda x: my_split([item for item in re.split(' ',x) if item != '']))
    text_df['word'] = text_df['word'].apply(lambda x: del_all(x))
    text_df['word'] = text_df['word'].apply(lambda x: [item.strip() for item in x if item not in ['',' ']])
    
    # момент разметки
    text_df = make_markup(text_df, df)
    text_df = last_changes(text_df)
    
    # удаляем вспомогательные столбцы
    del text_df['len']
    del text_df['new_len']
    text_df['word'] = text_df['word'].apply(lambda x: [item.lower() for item in x if item not in ['',' ']])
    idx = [i for i in range(len(text_df)) if len(text_df['word'][i])==0]
    text_df = text_df.drop(index=idx)
    text_df.reset_index(drop = True, inplace= True)
    # удаляем оставшуюся пунктуацию
    text_df = punctuation(text_df)
    return text_df


# отделяем знаки пунктуации и другие от слов
def my_split(lst):
    for k in range(3):
        for i in range(len(lst)-1,-1,-1):
            if lst[i] in ['',' ']:
                lst.pop(i)
            else:
                lst[i] = lst[i].replace('\xa0', ' ').replace('\t', ' ').replace('…','')
                idx = []
                lens = len(lst[i])
                for item in ['+', ')', '»',';','.',',', '"','(', '«',':',' ', '-\t','\\', '/','”','“','-','–','_________','*','№','%']:
                     idx.append(lst[i].find(item))
                for item in ['\xa0','"','.']:
                    if lst[i].endswith(item) and lens-1 not in idx:
                        idx.append(lens-1)                    
                idx.sort(reverse=True)
                for item in idx:
                    if item!=-1 and item!=0:
                        lst.insert(i+1, lst[i][item])
                        lst.insert(i+2, lst[i][item+1:])
                        lst[i] = lst[i][:item]
                    elif item!=-1 and item == 0:
                        lst.insert(i+1, lst[i][item+1:])
                        lst[i] = lst[i][item]
                    elif item!= -1 and item == lens-1:
                        lst.insert(i+1, lst[i][item])
                        lst[i] = lst[i][:item] 
    return lst


def del_all(lst):
    start = lst.copy()
    for i in range(len(lst)-1,-1,-1):
        split = lst[i]
        split = split.split(' ')
        if len(split) > 1:
            for j in range(len(split)):
                lst.insert(i+1+j, split[j])
            lst.pop(i)
    return lst


def last_changes(df):
    """
    размечаем строки без сущностей и оставшиеся слова не являющиеся сущностями
    """
    for i in range(len(df)):
        if df['tag'][i] == 0:
            df['tag'][i] = ['O'] * len(df['word'][i])
        else:
            llst = []
            for item in df['tag'][i]:
                if item in TAG:
                    llst.append(item)
                else:
                    llst.append('O')
            df['tag'][i] = llst
        assert (len(df['word'][i]) == len(df['tag'][i]))
    return df


def find_rows(txt_df, ann_df):
    """
    находим индекс строки для каждой аннотации
    """
    ann_df.insert(4, 'idx' , 0)
    for i in range(len(ann_df)):
        for j in range(len(txt_df)):
            start = txt_df['new_len'][j]-txt_df['len'][j]
            end = txt_df['new_len'][j]
            if int(ann_df['coords'][i][0]) in np.arange(start, end):
                ann_df['idx'][i] = j
                break
            else:
                pass
    return ann_df
    
    
def make_markup(text_df, ann_df):
    """
    делаем BIO разметку
    """
    for i in range(len(ann_df)):
        words_count = len(ann_df['words'][i])
        lens = [len(item) for item in ann_df['words'][i]] 
        row = text_df.iloc[[ann_df['idx'][i]]]
        if row['tag'].item() == 0:
                        text_df['tag'][ann_df['idx'][i]] = text_df['word'][ann_df['idx'][i]].copy()
                        row['tag'] = row['word']
        for k in range(words_count):
                if k == 0:
                    idx = (row['tag'].item()).index(ann_df['words'][i][k])
                    text_df['tag'][ann_df['idx'][i]][idx] = 'B-'+ ann_df['class'][i].lower()
                else:
                    idx = row['tag'].item().index(ann_df['words'][i][k])
                    text_df['tag'][ann_df['idx'][i]][idx] = 'I-'+ ann_df['class'][i].lower()
    return text_df


def remove_punctuation(sentence):
    """
    вспомогательная функция для удаления пунктуационных знаков
    """
    cleaned_sentence = re.sub(r'[?!\'"#]', '', sentence)
    cleaned_sentence = re.sub(r'[-.,;:(){}\/<>№»«|-|_]', '', cleaned_sentence)
    return cleaned_sentence


def punctuation(df):
    """
    удаляем пунктуационные знаки
    """
    for i in range(len(df)):
        lst = df['word'][i]
        for j in range(len(lst)-1,-1,-1):
            res = remove_punctuation(lst[j])
            if res =='' or res == ' ':
                df['word'][i].pop(j)
                df['tag'][i].pop(j)
    return df