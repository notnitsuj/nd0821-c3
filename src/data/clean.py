import csv


def clean_csv(input_file, output_file):
    with open(input_file, "r", newline="", encoding="utf-8") as infile:
        reader = csv.reader(infile)
        cleaned_data = []

        for row in reader:
            cleaned_row = [cell.replace(" ", "") for cell in row]
            cleaned_data.append(cleaned_row)

    with open(output_file, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(cleaned_data)


if __name__ == "__main__":
    input_csv = "census.csv"
    output_csv = "cleaned_census.csv"
    clean_csv(input_csv, output_csv)
