import plac
from tqdm import tqdm
import re
import logging
from distutils.util import strtobool

import worker_consumer as wc
from flashtext import KeywordProcessor

logger = logging.getLogger('ngram_replacer')


def reader(source):
    with open(source, 'r') as fin:
        for line in fin:
            yield line


def writer(x, output_stream, pbar, **kwargs):
    for line in x:
        if line is not None and len(line) > 0:
            print(line, file=output_stream, flush=True)

    pbar.update(1)


def helper(line, vocab, ngrams, pattern, keep_orig, keep_orig_if_unchanged,
           sep, **kwargs):
    line = line.strip()
    line_orig = str(line)
    res = list()
    if keep_orig:
        res.append(line_orig)

    if ngrams is not None:
        #  line = pattern.sub(lambda m: ngrams[re.escape(m.group(0))], line)
        line = ngrams.replace_keywords(line)

    if vocab is not None:
        line = ' '.join((w for w in line.split() if w in vocab or sep in w))

    if line != line_orig:
        res.append(line)
    elif keep_orig_if_unchanged and not keep_orig:
        res.append(line_orig)

    return res


@plac.annotations(
    input=("Sentences to replace", 'option', 'i', str),
    output=("Output file", 'option', 'o', str),
    ngrams=("File containing space separated ngrams", 'option', 'n', str),
    sep=("n-gram separator to use in the output", 'option', 'sp', str),
    vocab=("Unigrams to keep", 'option', 'v', str),
    keep_orig=("Keep the original line as well", 'option', None, strtobool),
    keep_orig_if_unchanged=("Keep the original line if no bigram found", 'option', None, strtobool),
    threads=("Number of parallel threads", 'option', 't', int),
)
def main(input, output, sep='&#32;', ngrams=None, vocab=None, keep_orig=False,
         keep_orig_if_unchanged=False, threads=1):
    assert ngrams or vocab

    logger.info('Loading vocab and ngrams...')
    if vocab:
        vocab_cleaner = KeywordProcessor(case_sensitive=True)
        with open(vocab, 'r') as fin:
            vocab = {l.strip() for l in fin}

    pattern=None
    keyword_processor = None
    if ngrams:
        keyword_processor = KeywordProcessor(case_sensitive=True)
        with open(ngrams) as fin:
            for l in fin:
                l = l.strip()
                keyword_processor.add_keyword(l, l.replace(' ', sep))

    logger.info('Replacing...')
    with open(output, 'w') as fout, tqdm() as pbar:
        wc.server(helper, writer, reader(input),
                  output_stream=fout, pbar=pbar,
                  vocab=vocab, ngrams=keyword_processor, pattern=pattern,
                  keep_orig=keep_orig, keep_orig_if_unchanged=keep_orig_if_unchanged,
                  sep=sep,
                  num_threads=threads)

if __name__ == '__main__':
    plac.call(main)
