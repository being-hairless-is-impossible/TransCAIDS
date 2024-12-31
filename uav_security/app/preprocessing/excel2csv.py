import pandas

def excel2csv(file_path, output_path):
    df = pandas.read_excel(file_path)
    df.to_csv(output_path, index=False)
    print("Conversion completed successfully!")
    return df

if __name__ == '__main__':
    file = '/data/physical_ready.xlsx'
    excel2csv(file,
              "/data/physical_ready.csv")