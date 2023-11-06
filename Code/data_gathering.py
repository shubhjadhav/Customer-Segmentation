import pandas as pd
from tabulate import tabulate

data_path = "../Data/"

invoice_df = pd.read_csv(data_path+'invoice_data.csv', encoding='unicode_escape')
invoice_df['InvoiceDate'] = pd.to_datetime(invoice_df['InvoiceDate'])

print(f"\nTotal observations (records): {invoice_df.shape[0]}\n")
print(tabulate(invoice_df.head(), headers='keys', tablefmt='psql'))
