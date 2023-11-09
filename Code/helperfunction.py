import pandas as pd

data_path = "../Data/"


def create_df(file_name):
    invoice_df = pd.read_csv(data_path + file_name + '.csv', encoding='unicode_escape')
    invoice_df['InvoiceDate'] = pd.to_datetime(invoice_df['InvoiceDate'])
    return invoice_df
