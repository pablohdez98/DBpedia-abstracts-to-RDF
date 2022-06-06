FOLDER_PATH = 'log_files/'


def tracking_log(input, level):
    with open(FOLDER_PATH + 'tracking.txt', 'a', encoding='utf8') as f:
        if level == 0:
            f.write('ORIGINAL SENTENCE: ' + input + '\n')
        if level == 1:
            f.write('  COREFEREE: ' + input.text + '\n')
        if level == 2:
            f.write('  LIST OF SENTENCES:\n')
            for i, sent in enumerate(input):
                f.write(f'    {i}: {sent}\n')
        if level == 3:
            f.write('  SIMPLE SENTENCES:\n')
            for i, sent in enumerate(input):
                f.write(f'    {i}: {sent}\n')
        if level == 4:
            f.write('  TEXT TRIPLES:\n')
            for i, sent in enumerate(input):
                f.write(f'    {i}: {sent.__repr__()}\n')
        if level == 5:
            f.write('  RDF TRIPLES:\n')
            for i, sent in enumerate(input):
                f.write(f'    {i}: {sent}\n')
    f.close()


def triple_with_no_uri_log(timestr, triple, item):
    with open(FOLDER_PATH + 'literals_log.txt', 'a', encoding='utf-8') as f:
        f.write(f'{timestr} >>SENTENCE: {triple.sent}\n')
        f.write(f'{triple.subj} | {triple.pred} | {triple.objct}\n')
        f.write(f'Literal: <{item}>\n')
    f.close()