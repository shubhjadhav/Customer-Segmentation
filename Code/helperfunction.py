import pandas as pd
import os

data_path = "../Data/"

def get_file_path(file_name):
    dir_path = os.getcwd()
    print("Current work Directory", dir_path)
    file_path = dir_path + os.sep + file_name
    print("File Path is ", file_path)
    return file_path
def create_df(file_name):
    file_path = get_file_path(file_name)
    invoice_df = pd.read_csv(data_path + file_name + '.csv', encoding='unicode_escape')
    invoice_df['InvoiceDate'] = pd.to_datetime(invoice_df['InvoiceDate'])
    return invoice_df


def print_observation(text):
    print("OBSERVATION: ", text)