import os
from collections import Counter

def read_data(fname, count, word2idx, mem_size):
    if os.path.isfile(fname):
        with open(fname) as f:
            lines = f.readlines()
    else:
        raise("[!] Data %s not found" % fname)

    words = []
    for line in lines:
        for word in line.split():
            if not word.isdigit():
                words.append(word)

    if len(count) == 0:
        count.append(['<eos>', 0])

    count[0][1] += len(lines)
    count.extend(Counter(words).most_common())

    if len(word2idx) == 0:
        word2idx['<eos>'] = 0

    for word, _ in count:
        if word not in word2idx:
            word2idx[word] = len(word2idx)

    data = list()
    query = list()
    target = list()
    dataIdx = list()

    for line in lines:
        temp = list()

        if line.__contains__('?'): #Query or Not
            idx = len(data)
            if idx < mem_size:
                continue

            dataIdx.append(idx)
            q = line.split("\t")

            for word in q[0].split():
                if not word.isdigit():
                    index = word2idx[word]
                    temp.append(index)
            query.append(temp)

            temp = list()
            for word in q[1].split(','):
                if not word.isdigit():
                    index = word2idx[word]
                    temp.append(index)
            target.append(temp)

        else: # Sentence
            for word in line.split():
                if not word.isdigit():
                    index = word2idx[word]
                    temp.append(index)
            temp.append(word2idx['<eos>'])
            data.append(temp)

    print("Read %s words from %s" % (len(data), fname))
    return data, query, target, dataIdx
