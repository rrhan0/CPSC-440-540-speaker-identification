import pandas as pd
import ast


def earliest_alias(df, inferred_speaker):
    earliest_name = ''
    earliest_index = float('inf')
    if isinstance(inferred_speaker, float):
        return ''

    for index, row in df.iterrows():
        main_name = row['Main Name']
        aliases = row['Aliases']

        for alias in aliases:
            alias_index = inferred_speaker.find(alias)
            if alias_index != -1 and alias_index < earliest_index:
                earliest_name = main_name
                earliest_index = alias_index

    return earliest_name


def analyze(results_path, quotes_path, character_path, metric):
    # Read the CSV files
    results = pd.read_csv(results_path)
    quote_info = pd.read_csv(quotes_path)
    characters = pd.read_csv(character_path)
    characters['Aliases'] = characters['Aliases'].apply(ast.literal_eval)

    # Determine the length of the shorter dataframe
    min_length = min(len(results), len(quote_info))
    # Truncate the longer dataframe
    results = results.iloc[:min_length]
    quote_info = quote_info.iloc[:min_length]
    results_info = pd.concat([results, quote_info], axis=1)

    table = metric(characters, results_info)

    # return results_info

    count_alias_present = table['correct'].sum()
    total_count = table.shape[0]
    anaphoric_correct_count = (
            table[table['correct']]['quoteType'] == 'Anaphoric').sum()
    anaphoric_count = (table['quoteType'] == 'Anaphoric').sum()
    implicit_correct_count = (
            table[table['correct']]['quoteType'] == 'Implicit').sum()
    implicit_count = (table['quoteType'] == 'Implicit').sum()
    explicit_correct_count = (
            table[table['correct']]['quoteType'] == 'Explicit').sum()
    explicit_count = (table['quoteType'] == 'Explicit').sum()
    return count_alias_present, total_count, anaphoric_correct_count, anaphoric_count, implicit_correct_count, implicit_count, explicit_correct_count, explicit_count


def strong_metric(characters, results_info):
    results_info_copy = results_info.copy()
    results_info_copy['correct'] = results_info_copy.apply(
        lambda row: earliest_alias(characters, row['inferred_speaker']) == row['speaker'], axis=1)
    return results_info_copy


def weak_metric(characters, results_info):
    # Append the columns together
    results_info_character = pd.merge(results_info, characters, left_on='speaker', right_on='Main Name', how='left')
    f = lambda row: any(not isinstance(row['inferred_speaker'], float) and alias in row['inferred_speaker'] for alias in row['Aliases'])
    results_info_character['correct'] = results_info_character.apply(f, axis=1)

    return results_info_character


def main():
    pp_paths = [('context0/Mistral 7b NO INST/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context1/Mistral 7b NO INST/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context2/Mistral 7b NO INST/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context4/Mistral 7b NO INST/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context8/Mistral 7b NO INST/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context16/Mistral 7b NO INST/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv')
                ]
    emma_paths = [('context0/Mistral 7b NO INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context1/Mistral 7b NO INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context2/Mistral 7b NO INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context4/Mistral 7b NO INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context8/Mistral 7b NO INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context16/Mistral 7b NO INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv')
                  ]

    context = [0, 1, 2, 4, 8, 16]
    total_acc = []
    anaphoric_acc = []
    implicit_acc = []
    explicit_acc = []
    for c, pp, emma in zip(context, pp_paths, emma_paths):
        count_alias_present_pp, \
            total_count_pp, \
            anaphoric_correct_count_pp, \
            anaphoric_count_pp, \
            implicit_correct_count_pp, \
            implicit_count_pp, \
            explicit_correct_count_pp, \
            explicit_count_pp = analyze(*pp, strong_metric)

        count_alias_present_emma, \
            total_count_emma, \
            anaphoric_correct_count_emma, \
            anaphoric_count_emma, \
            implicit_correct_count_emma, \
            implicit_count_emma, \
            explicit_correct_count_emma, \
            explicit_count_emma = analyze(*emma, strong_metric)
        total = total_count_pp + total_count_emma
        anaphoric_count = anaphoric_count_pp + anaphoric_count_emma
        implicit_count = implicit_count_pp + implicit_count_emma
        explicit_count = explicit_count_pp + explicit_count_emma

        total_correct_count = count_alias_present_pp + count_alias_present_emma
        anaphoric_correct_count = anaphoric_correct_count_pp + anaphoric_correct_count_emma
        implicit_correct_count = implicit_correct_count_pp + implicit_correct_count_emma
        explicit_correct_count = explicit_correct_count_pp + explicit_correct_count_emma

        total_acc.append(total_correct_count / total * 100)
        anaphoric_acc.append(anaphoric_correct_count / anaphoric_count * 100)
        implicit_acc.append(implicit_correct_count / implicit_count * 100)
        explicit_acc.append(explicit_correct_count / explicit_count * 100)

    formatted_list = ["{:.1f}".format(num) for num in total_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in anaphoric_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in implicit_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in explicit_acc]
    print(" & ".join(formatted_list) + " \\\\")
    print()

    pp_paths = [('context0/Mistral 7b INST/PrideAndPrejudice.csv',
            'PrideAndPrejudice/quotation_info.csv',
            'PrideAndPrejudice/character_info.csv'),
             ('context1/Mistral 7b INST/PrideAndPrejudice.csv',
              'PrideAndPrejudice/quotation_info.csv',
              'PrideAndPrejudice/character_info.csv'),
             ('context2/Mistral 7b INST/PrideAndPrejudice.csv',
              'PrideAndPrejudice/quotation_info.csv',
              'PrideAndPrejudice/character_info.csv'),
             ('context4/Mistral 7b INST/PrideAndPrejudice.csv',
              'PrideAndPrejudice/quotation_info.csv',
              'PrideAndPrejudice/character_info.csv'),
             ('context8/Mistral 7b INST/PrideAndPrejudice.csv',
              'PrideAndPrejudice/quotation_info.csv',
              'PrideAndPrejudice/character_info.csv'),
             ('context16/Mistral 7b INST/PrideAndPrejudice.csv',
              'PrideAndPrejudice/quotation_info.csv',
              'PrideAndPrejudice/character_info.csv')
             ]
    emma_paths = [('context0/Mistral 7b INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context1/Mistral 7b INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context2/Mistral 7b INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context4/Mistral 7b INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context8/Mistral 7b INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context16/Mistral 7b INST/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv')
                  ]

    context = [0, 1, 2, 4, 8, 16]
    total_acc = []
    anaphoric_acc = []
    implicit_acc = []
    explicit_acc = []
    for c, pp, emma in zip(context, pp_paths, emma_paths):
        count_alias_present_pp, \
            total_count_pp, \
            anaphoric_correct_count_pp, \
            anaphoric_count_pp, \
            implicit_correct_count_pp, \
            implicit_count_pp, \
            explicit_correct_count_pp, \
            explicit_count_pp = analyze(*pp, strong_metric)

        count_alias_present_emma, \
            total_count_emma, \
            anaphoric_correct_count_emma, \
            anaphoric_count_emma, \
            implicit_correct_count_emma, \
            implicit_count_emma, \
            explicit_correct_count_emma, \
            explicit_count_emma = analyze(*emma, strong_metric)
        total = total_count_pp + total_count_emma
        anaphoric_count = anaphoric_count_pp + anaphoric_count_emma
        implicit_count = implicit_count_pp + implicit_count_emma
        explicit_count = explicit_count_pp + explicit_count_emma

        total_correct_count = count_alias_present_pp + count_alias_present_emma
        anaphoric_correct_count = anaphoric_correct_count_pp + anaphoric_correct_count_emma
        implicit_correct_count = implicit_correct_count_pp + implicit_correct_count_emma
        explicit_correct_count = explicit_correct_count_pp + explicit_correct_count_emma

        total_acc.append(total_correct_count / total * 100)
        anaphoric_acc.append(anaphoric_correct_count / anaphoric_count * 100)
        implicit_acc.append(implicit_correct_count / implicit_count * 100)
        explicit_acc.append(explicit_correct_count / explicit_count * 100)

    print(f'PP: total:{total_count_pp}, anaphoric:{anaphoric_count_pp}, implicit:{implicit_count_pp}, explicit:{explicit_count_pp}')
    print(f'EMMA: total:{total_count_emma}, anaphoric:{anaphoric_count_emma}, implicit:{implicit_count_emma}, explicit:{explicit_count_emma}')
    formatted_list = ["{:.1f}".format(num) for num in total_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in anaphoric_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in implicit_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in explicit_acc]
    print(" & ".join(formatted_list) + " \\\\")
    print()

    pp_paths = [('context0/Llama 7b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context1/Llama 7b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context2/Llama 7b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context4/Llama 7b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context8/Llama 7b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context16/Llama 7b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv')
                ]
    emma_paths = [('context0/Llama 7b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context1/Llama 7b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context2/Llama 7b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context4/Llama 7b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context8/Llama 7b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context16/Llama 7b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv')
                  ]

    context = [0, 1, 2, 4, 8, 16]
    total_acc = []
    anaphoric_acc = []
    implicit_acc = []
    explicit_acc = []
    for c, pp, emma in zip(context, pp_paths, emma_paths):
        count_alias_present_pp, \
            total_count_pp, \
            anaphoric_correct_count_pp, \
            anaphoric_count_pp, \
            implicit_correct_count_pp, \
            implicit_count_pp, \
            explicit_correct_count_pp, \
            explicit_count_pp = analyze(*pp, strong_metric)

        count_alias_present_emma, \
            total_count_emma, \
            anaphoric_correct_count_emma, \
            anaphoric_count_emma, \
            implicit_correct_count_emma, \
            implicit_count_emma, \
            explicit_correct_count_emma, \
            explicit_count_emma = analyze(*emma, strong_metric)
        total = total_count_pp + total_count_emma
        anaphoric_count = anaphoric_count_pp + anaphoric_count_emma
        implicit_count = implicit_count_pp + implicit_count_emma
        explicit_count = explicit_count_pp + explicit_count_emma

        total_correct_count = count_alias_present_pp + count_alias_present_emma
        anaphoric_correct_count = anaphoric_correct_count_pp + anaphoric_correct_count_emma
        implicit_correct_count = implicit_correct_count_pp + implicit_correct_count_emma
        explicit_correct_count = explicit_correct_count_pp + explicit_correct_count_emma

        total_acc.append(total_correct_count / total * 100)
        anaphoric_acc.append(anaphoric_correct_count / anaphoric_count * 100)
        implicit_acc.append(implicit_correct_count / implicit_count * 100)
        explicit_acc.append(explicit_correct_count / explicit_count * 100)

    print(
        f'PP: total:{total_count_pp}, anaphoric:{anaphoric_count_pp}, implicit:{implicit_count_pp}, explicit:{explicit_count_pp}')
    print(
        f'EMMA: total:{total_count_emma}, anaphoric:{anaphoric_count_emma}, implicit:{implicit_count_emma}, explicit:{explicit_count_emma}')
    formatted_list = ["{:.1f}".format(num) for num in total_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in anaphoric_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in implicit_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in explicit_acc]
    print(" & ".join(formatted_list) + " \\\\")
    print()

    pp_paths = [('context0/Llama 13b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context1/Llama 13b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context2/Llama 13b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context4/Llama 13b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context8/Llama 13b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv'),
                ('context16/Llama 13b/PrideAndPrejudice.csv',
                 'PrideAndPrejudice/quotation_info.csv',
                 'PrideAndPrejudice/character_info.csv')
                ]
    emma_paths = [('context0/Llama 13b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context1/Llama 13b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context2/Llama 13b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context4/Llama 13b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context8/Llama 13b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv'),
                  ('context16/Llama 13b/Emma.csv',
                   'Emma/quotation_info.csv',
                   'Emma/character_info.csv')
                  ]

    context = [0, 1, 2, 4, 8, 16]
    total_acc = []
    anaphoric_acc = []
    implicit_acc = []
    explicit_acc = []
    for c, pp, emma in zip(context, pp_paths, emma_paths):
        count_alias_present_pp, \
            total_count_pp, \
            anaphoric_correct_count_pp, \
            anaphoric_count_pp, \
            implicit_correct_count_pp, \
            implicit_count_pp, \
            explicit_correct_count_pp, \
            explicit_count_pp = analyze(*pp, strong_metric)

        count_alias_present_emma, \
            total_count_emma, \
            anaphoric_correct_count_emma, \
            anaphoric_count_emma, \
            implicit_correct_count_emma, \
            implicit_count_emma, \
            explicit_correct_count_emma, \
            explicit_count_emma = analyze(*emma, strong_metric)
        total = total_count_pp + total_count_emma
        anaphoric_count = anaphoric_count_pp + anaphoric_count_emma
        implicit_count = implicit_count_pp + implicit_count_emma
        explicit_count = explicit_count_pp + explicit_count_emma

        total_correct_count = count_alias_present_pp + count_alias_present_emma
        anaphoric_correct_count = anaphoric_correct_count_pp + anaphoric_correct_count_emma
        implicit_correct_count = implicit_correct_count_pp + implicit_correct_count_emma
        explicit_correct_count = explicit_correct_count_pp + explicit_correct_count_emma

        total_acc.append(total_correct_count / total * 100)
        anaphoric_acc.append(anaphoric_correct_count / anaphoric_count * 100)
        implicit_acc.append(implicit_correct_count / implicit_count * 100)
        explicit_acc.append(explicit_correct_count / explicit_count * 100)

    print(
        f'PP: total:{total_count_pp}, anaphoric:{anaphoric_count_pp}, implicit:{implicit_count_pp}, explicit:{explicit_count_pp}')
    print(
        f'EMMA: total:{total_count_emma}, anaphoric:{anaphoric_count_emma}, implicit:{implicit_count_emma}, explicit:{explicit_count_emma}')
    formatted_list = ["{:.1f}".format(num) for num in total_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in anaphoric_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in implicit_acc]
    print(" & ".join(formatted_list) + " \\\\")

    formatted_list = ["{:.1f}".format(num) for num in explicit_acc]
    print(" & ".join(formatted_list) + " \\\\")
    print()



if __name__ == '__main__':
    main()