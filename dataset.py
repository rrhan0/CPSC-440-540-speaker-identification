import csv
import ast
import nltk.tokenize
nltk.download('punkt')


def write_quotes(quote_path, novel_path, output_csv_name):
    with open(quote_path, 'r', newline='') as input_csv, \
            open(novel_path, 'r', newline='') as novel, \
            open(output_csv_name, 'w', newline='') as quotes_csv:
        reader = csv.DictReader(input_csv)

        fieldnames = ['quoteText']
        quotes = csv.DictWriter(quotes_csv, fieldnames=fieldnames)
        quotes.writeheader()

        text = novel.read()
        text = text.replace("\n", " ")

        for row in reader:
            quote_byte_spans = ast.literal_eval(row.get('quoteByteSpans'))
            start = quote_byte_spans[0][0] - 1
            end = quote_byte_spans[-1][-1] + 1
            quote = text[start:end]
            quotes.writerow({'quoteText': quote})
        print(f'Done {output_csv_name}')


def write_context(quote_path, novel_path, output_csv_name, context_window=0):
    with open(quote_path, 'r', newline='') as input_csv, \
            open(novel_path, 'r', newline='') as novel, \
            open(output_csv_name, 'w', newline='') as context_csv:
        reader = csv.DictReader(input_csv)

        fieldnames = ['left_context', 'right_context']
        context = csv.DictWriter(context_csv, fieldnames=fieldnames)
        context.writeheader()

        text = novel.read()
        text = text.replace("\n", " ")

        for row in reader:
            quote_byte_spans = ast.literal_eval(row.get('quoteByteSpans'))
            start = quote_byte_spans[0][0] - 1
            end = quote_byte_spans[-1][-1] + 1

            left_sentences = nltk.sent_tokenize(text[0:start])
            right_sentences = nltk.sent_tokenize(text[end+1:len(text)])

            left_context = ' '.join(left_sentences[-context_window:])
            right_context = ' '.join(right_sentences[:context_window])
            context.writerow({'left_context': left_context, 'right_context': right_context})
        print(f'Done {output_csv_name}')


def main():
    write_quotes('PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/novel_text.txt',
                 'PrideAndPrejudice_quotes.csv')

    # for i in [1, 2, 4, 8, 16]:
    #     write_context('PrideAndPrejudice/quotation_info.csv',
    #                   'PrideAndPrejudice/novel_text.txt',
    #                   f'PrideAndPrejudice_context{i}.csv',
    #                   context_window=i)

    write_quotes('Emma/quotation_info.csv',
                 'Emma/novel_text.txt',
                 'Emma_quotes.csv')

    # for i in [1, 2, 4, 8, 16]:
    #     write_context('Emma/quotation_info.csv',
    #                   'Emma/novel_text.txt',
    #                   f'Emma_context{i}.csv',
    #                   context_window=i)


if __name__ == '__main__':
    main()