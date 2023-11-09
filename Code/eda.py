import helperfunction as hf
from tabulate import tabulate

invoice_df = hf.create_df('invoice_data')

print(f"\nTotal observations (records): {invoice_df.shape[0]}\n")
print(tabulate(invoice_df.head(), headers='keys', tablefmt='psql'))
