import pandas as pd
import numpy as np
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
import functools

####### STUDENTS FILL THIS OUT ######
#Question 3
def reduce_dimension_ndc(df, ndc_df):
    '''
    df: pandas dataframe, input dataset
    ndc_df: pandas dataframe, drug code dataset used for mapping in generic names
    return:
        df: pandas dataframe, output dataframe with joined generic drug name
    '''
    reduce_dim_df = pd.merge(df, ndc_df[['NDC_Code', 'Non-proprietary Name']], left_on='ndc_code', right_on='NDC_Code', how='left')
    reduce_dim_df = reduce_dim_df.rename(columns={'Non-proprietary Name': 'generic_drug_name'})
    reduce_dim_df = reduce_dim_df.drop(['NDC_Code', 'ndc_code'], axis=1)
    
    # Replace generic names that are similar but listed as different
    reduce_dim_df['generic_drug_name'] = reduce_dim_df['generic_drug_name'].replace({'Insulin Human': 'Human Insulin', 'Glyburide-metformin Hydrochloride': 'Glyburide And Metformin Hydrochloride', 'Glipizide And Metformin Hcl': 'Glipizide And Metformin Hydrochloride'})    

    return reduce_dim_df

#Question 4
def select_first_encounter(df):
    '''
    df: pandas dataframe, dataframe with all encounters
    return:
        - first_encounter_df: pandas dataframe, dataframe with only the first encounter for a given patient
    '''
    first_encounter_df = df.sort_values(['patient_nbr', 'encounter_id']).groupby('patient_nbr').first().reset_index()
    return first_encounter_df


#Question 6
# def patient_dataset_splitter(df, PREDICTOR_FIELD, patient_key='patient_nbr'):
#     '''
#     df: pandas dataframe, input dataset that will be split
#     patient_key: string, column that is the patient id

#     return:
#      - train: pandas dataframe,
#      - validation: pandas dataframe,
#      - test: pandas dataframe,
     
     
#     Approximately 60%/20%/20% train/validation/test split
#     Randomly sample different patients into each data partition
#     IMPORTANT Make sure that a patient's data is not in more than one partition, so that we can avoid possible data leakage.
#     Make sure that the total number of unique patients across the splits is equal to the total number of unique patients in the original dataset
#     Total number of rows in original dataset = sum of rows across all three dataset partitions
#     '''
#     for col in ['race', 'primary_diagnosis_code']:
#         df[col] = df[col].astype('str')
#     y = df[PREDICTOR_FIELD]
#     X = df.drop(PREDICTOR_FIELD, axis=1)
#     random_state=55
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=random_state, stratify=y)
#     X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=random_state, stratify=y_test)
    
#     train = pd.concat([X_train, y_train], axis=1)
#     train = train.T.drop_duplicates().T
# #     train = train.loc[:,~train.duplicated()]
#     validation = pd.concat([X_val, y_val], axis=1)
#     validation = validation.T.drop_duplicates().T
# #     validation = validation.loc[:,~validation.duplicated()]
#     test = pd.concat([X_test, y_test], axis=1)
#     test = test.T.drop_duplicates().T
# #     test = test.loc[:,~test.duplicated()]
    
#     return train, validation, test

def patient_dataset_splitter(df, patient_key='patient_nbr'):
    '''
    df: pandas dataframe, input dataset that will be split
    patient_key: string, column that is the patient id

    return:
     - train: pandas dataframe,
     - validation: pandas dataframe,
     - test: pandas dataframe,
    '''
    train_perc, val_perc, test_perc = 0.6, 0.2, 0.2
    df = df.iloc[np.random.permutation(len(df))]
    unique_values = df[patient_key].unique()
    total_values = len(unique_values)
    train_size = int(train_perc * total_values)
    val_size = int(val_perc * total_values)
    test_size = int(test_perc * total_values)
    train = df[df[patient_key].isin(unique_values[:train_size])]
    validation = df[df[patient_key].isin(unique_values[train_size: train_size+val_size])]
    test = df[df[patient_key].isin(unique_values[train_size+val_size:])]
    return train, validation, test

#Question 7

def create_tf_categorical_feature_cols(categorical_col_list,
                              vocab_dir='./diabetes_vocab/'):
    '''
    categorical_col_list: list, categorical field list that will be transformed with TF feature column
    vocab_dir: string, the path where the vocabulary text files are located
    return:
        output_tf_list: list of TF feature columns
    '''
    output_tf_list = []
    for c in categorical_col_list:
        vocab_file_path = os.path.join(vocab_dir,  c + "_vocab.txt")
        '''
        Which TF function allows you to read from a text file and create a categorical feature
        You can use a pattern like this below...
        tf_categorical_feature_column = tf.feature_column.......

        '''
        vocab = tf.feature_column.categorical_column_with_vocabulary_file(key=c, vocabulary_file=vocab_file_path, num_oov_buckets=1)
        tf_categorical_feature_column = tf.feature_column.indicator_column(vocab)
        if tf_categorical_feature_column not in output_tf_list:
            output_tf_list.append(tf_categorical_feature_column)
    return output_tf_list

#Question 8
def normalize_numeric_with_zscore(col, mean, std):
    '''
    This function can be used in conjunction with the tf feature column for normalization
    '''
    return (col - mean)/std



def create_tf_numeric_feature(col, MEAN, STD, default_value=0):
    '''
    col: string, input numerical column name
    MEAN: the mean for the column in the training data
    STD: the standard deviation for the column in the training data
    default_value: the value that will be used for imputing the field

    return:
        tf_numeric_feature: tf feature column representation of the input field
    '''
    normalizer = functools.partial(normalize_numeric_with_zscore, mean=MEAN, std=STD) #normalize_numeric_with_zscore(col, mean=MEAN, std=STD)
    tf_numeric_feature = tf.feature_column.numeric_column(key=col, default_value=default_value, normalizer_fn=normalizer, dtype=tf.float64)
    return tf_numeric_feature

#Question 9
def get_mean_std_from_preds(diabetes_yhat):
    '''
    diabetes_yhat: TF Probability prediction object
    '''
    m = diabetes_yhat.mean()
    s = diabetes_yhat.stddev()
    return m, s

# Question 10
# def get_student_binary_prediction(df, pred_field, actual_field, threshold):
def get_student_binary_prediction(df, pred_field, threshold):
    '''
    df: pandas dataframe prediction output dataframe
    col: str,  probability mean prediction field
    return:
        student_binary_prediction: pandas dataframe converting input to flattened numpy array and binary labels
    '''
    student_binary_prediction = df[pred_field].apply(lambda x: 1 if x >= 5 else 0).values
    return student_binary_prediction
#     df['score'] = df[pred_field].apply(lambda x: 1 if x>=threshold else 0)
#     df['label_value'] = df[actual_field].apply(lambda x: 1 if x>=threshold else 0)
#     return df[['score', 'label_value']]
#     return df['score']
