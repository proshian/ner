from utils import *
from sklearn.model_selection import train_test_split


PATH1 = r'data\train_part_1'
PATH2 = r'data\train_part_2'
PATH3 = r'data\train_part_3'
PATH4 = r'data\test_ner_only'


if __name__ == "__main__":
    # Filenames without extensions
    filenames = [filename[:-4] for filename in os.listdir(PATH1) if filename[-4:] == '.txt']
    train_df = make_data(PATH1, filenames)

    filenames =  [filename[:-4] for filename in os.listdir(PATH3) if filename[-4:] == '.txt']
    train_df_2 = make_data(PATH3, filenames)
    train_df = pd.concat([train_df, train_df_2], ignore_index=True)
    
    filenames =  [filename[:-4] for filename in os.listdir(PATH3) if filename[-4:] == '.txt']
    train_df_3 = make_data(PATH3, filenames)
    train_df = pd.concat([train_df, train_df_3], ignore_index=True)

    train_df, val_df = train_test_split(train_df, test_size=0.1)
    train_df.reset_index(drop = True, inplace= True)
    val_df.reset_index(drop = True, inplace= True)
    