"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.17.6
"""
from typing import Dict
import torch

import pandas as pd
from string import punctuation, ascii_lowercase


def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """strip whitespace from strings

    Args:
        df (pd.DataFrame): dataframe name

    Returns:
        pd.DataFrame: stripped whitespace from text
    """
    return df.applymap(lambda x: x.strip() if type(x) == str else x)


def _remove_empty_row(df: pd.DataFrame) -> pd.DataFrame:
    """remove empty rows

    Args:
        df (pd.DataFrame): dataframe name

    Returns:
        pd.DataFrame: removed empty rows
    """
    return df.dropna(axis=0).reset_index(drop=True)


def _lowercase_text(df: pd.DataFrame) -> pd.DataFrame:
    """lowercase strings

    Args:
        df (pd.DataFrame): dataframe name

    Returns:
        pd.DataFrame: lowercased text
    """
    return df.applymap(lambda x: x.lower() if type(x) == str else x)


def _remove_number(df: pd.DataFrame) -> pd.DataFrame:
    """remove number from strings

    Args:
        df (pd.DataFrame): dataframe name

    Returns:
        pd.DataFrame: removed number from text
    """
    # get all strings columns
    cols = df.select_dtypes(include=[object]).columns
    # remove numbers
    df[cols] = df[cols].replace('\\d+', '', regex=True)
    # dataframe with removed numbers
    return df


def _remove_punctuation(df: pd.DataFrame) -> pd.DataFrame:
    """remove punctuation from strings

    Args:
        df (pd.DataFrame): dataframe name

    Returns:
        pd.DataFrame: removed punctuation from text
    """
    # get all strings columns
    cols = df.select_dtypes(include=[object]).columns
    # remove punctuations
    for i in cols:
        df[i] = df[i].str.replace('[{}]'.format(punctuation), '', regex=True)
    # dataframe with removed punctuations
    return df


def _strip_accent(df: pd.DataFrame) -> pd.DataFrame:
    """strip accent from strings

    Args:
        df (pd.DataFrame): dataframe name

    Returns:
        pd.DataFrame: stripped accent from text
    """
    # get all strings columns
    cols = df.select_dtypes(include=[object]).columns
    # strip accent
    df[cols] = df[cols].apply(lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))
    # dataframe with stripped accent
    return df


def _remove_duplicate_row(df: pd.DataFrame) -> pd.DataFrame:
    """remove duplicate rows

    Args:
        df (pd.DataFrame): dataframe name

    Returns:
        pd.DataFrame: removed duplicated rows
    """
    return df.drop_duplicates().reset_index(drop=True)


def preprocess_belgium_name(belgium_name: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for belgium_name.

    Args:
        belgium_name (pd.DataFrame): Raw data

    Returns:
        pd.DataFrame: Preprocessed data, with `Name / Gender` :
        - stripped whitespace
        - removed empty row
        - lowercased text
        - removed number
        - removed punctuation
        - stripped accent
        - removed duplicated row
    """
    belgium_cleaned = (
        belgium_name
        .pipe(_strip_whitespace)
        .pipe(_remove_empty_row)
        .pipe(_lowercase_text)
        .pipe(_remove_number)
        .pipe(_remove_punctuation)
        .pipe(_strip_accent)
        .pipe(_remove_duplicate_row)
    )
    return belgium_cleaned


def preprocess_canadian_name(canadian_name: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for canadian_name.

    Args:
        canadian_name (pd.DataFrame): Raw data

    Returns:
        pd.DataFrame: Preprocessed data, with `Name / Gender` :
        - stripped whitespace
        - removed empty row
        - lowercased text
        - removed number
        - removed punctuation
        - stripped accent
        - removed duplicated row
    """
    canadian_cleaned = (
        canadian_name
        .pipe(_strip_whitespace)
        .pipe(_remove_empty_row)
        .pipe(_lowercase_text)
        .pipe(_remove_number)
        .pipe(_remove_punctuation)
        .pipe(_strip_accent)
        .pipe(_remove_duplicate_row)
    )
    return canadian_cleaned


def preprocess_french_name(fr_french_name: pd.DataFrame, idf_french_name: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for french_name.

    Args:
        fr_french_name (pd.DataFrame): Raw data
        idf_french_name (pd.DataFrame): Raw data

    Returns:
        pd.DataFrame: Preprocessed data, with `Name / Gender` :
        - stripped whitespace
        - removed empty row
        - lowercased text
        - removed number
        - removed punctuation
        - stripped accent
        - removed duplicated row
    """
    # concat rows of two dataframe
    french_name = pd.concat([fr_french_name, idf_french_name], axis=0).reset_index(drop=True)

    #pandas pipeline
    french_cleaned = (
        french_name
        .pipe(_strip_whitespace)
        .pipe(_remove_empty_row)
        .pipe(_lowercase_text)
        .pipe(_remove_number)
        .pipe(_remove_punctuation)
        .pipe(_strip_accent)
        .pipe(_remove_duplicate_row)
    )
    return french_cleaned


def preprocess_american_name(nyc_american_name: pd.DataFrame, usa_american_name: pd.DataFrame) -> pd.DataFrame:
    """Preprocesses the data for american_name.

    Args:
        nyc_american_name (pd.DataFrame): Raw data
        usa_american_name (pd.DataFrame): Raw data

    Returns:
        pd.DataFrame: Preprocessed data, with `Name / Gender` :
        - stripped whitespace
        - removed empty row
        - lowercased text
        - removed number
        - removed punctuation
        - stripped accent
        - removed duplicated row
    """
    # concat rows of two dataframe
    american_name = pd.concat([nyc_american_name, usa_american_name], axis=0).reset_index(drop=True)

    # pandas pipeline
    american_cleaned = (
        american_name
        .pipe(_strip_whitespace)
        .pipe(_remove_empty_row)
        .pipe(_lowercase_text)
        .pipe(_remove_number)
        .pipe(_remove_punctuation)
        .pipe(_strip_accent)
        .pipe(_remove_duplicate_row)
    )
    return american_cleaned


def _name_to_tensor(name: str) -> torch.tensor:
    """transpose string name into torch tensor

    Args:
        name (str): name to encode

    Returns:
        torch.tensor: name as one-hot vector representation
    """
    # fetch all ascii lowercase letters
    n_letters = len(ascii_lowercase)
    # each letter is an one-hot vector
    tensor = torch.zeros(len(name), 1, n_letters)

    # create a dictionnary of one-hot vectors
    for idx, letter in enumerate(name):
        # map character to dictionnary value
        letter_idx = ascii_lowercase.find(letter)
        # set vector to one if letter correspond to dictionnary
        tensor[idx][0][letter_idx] = 1

    # return tensor
    return tensor


def create_model_input_table(
    belgium_name: pd.DataFrame,
    canadian_name: pd.DataFrame,
    french_name: pd.DataFrame,
    american_name: pd.DataFrame,
    parameters: Dict
) -> pd.DataFrame:
    """concat rows from different datasets into one

    Args:
        belgium_name (pd.DataFrame): dataframe name
        canadian_name (pd.DataFrame): dataframe name
        french_name (pd.DataFrame): dataframe name
        american_name (pd.DataFrame): dataframe name
        parameters (Dict) : yaml configuration

    Returns:
        pd.DataFrame: dataframe concatened
    """
    df =  pd.concat([belgium_name, canadian_name, french_name, american_name], axis=0).reset_index(drop=True)
    df[parameters["name_encoded"]] = df[parameters["name"]].apply(lambda x: _name_to_tensor(x) if type(x) == str else x)
    return df