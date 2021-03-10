import pandas as pd
import csv
lula = pd.read_csv('lula.csv')



with open('lula.csv', 'rU') as myfile:
    filtered = (line.replace('\r', '') for line in myfile)
    for row in csv.reader(filtered):
        print(row)

print(lula.head())