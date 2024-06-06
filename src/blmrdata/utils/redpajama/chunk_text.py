# List of control characters = https://donsnotes.com/tech/charsets/ascii.html

import re
import math
import time

control_char_pattern = re.compile(
    r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]'
)
page_break_pattern = re.compile(
    r'([\x0C]|\n{2,})'
)
control_char_pattern_except_page_break = re.compile(
    r'[\x00-\x08\x0B\x0E-\x1F\x7F�]'
)

def count_control_characters(line):
    return len(control_char_pattern_except_page_break.findall(line))

def no_alpha_char(line):
    return re.match(r'^[\W\d]+$', line)

def compute_line_features(line):
    """
    A line is suspicious if:
    - It is is_empty
    - It contains more than 50% of control characters
    - It contains a single word
    - It contains a word with more than 20 characters

    Returns a tuple (
        is_suspicious,
        is_empty,
        is_short,
        line_length,
    )
    """
    line_length = len(line) + 1 # + 1 for trailing line break
    line = line.strip()
    is_empty = not line
    if is_empty:
        return (True, is_empty, True, line_length)
    num_chars = len(line)
    num_control_chars = count_control_characters(line)
    if num_control_chars > 4 or (num_chars > 1 and num_control_chars > 0.3 * num_chars):
        return (True, False, False, line_length)
    line_no_punctuation = re.sub(r'[^\w\s]', '', line)
    words = line_no_punctuation.split()
    num_words = len(words)
    if num_words <= 1:
        return (True, False, True, line_length)
    if no_alpha_char(line):
        return (True, False, True, line_length)
    if num_control_chars > 0.5 * num_words:
        return (True, False, False, line_length)
    # max_len = max(map(len, words))
    # if max_len > 25:
    #     return (True, False, False, line_length)
    return (False, False, False, line_length)

def chunk_text_simple(text, chunk_size=10_000):
    """
    Split text into chunks of text of a given size

    Parameters:
    - text: str
        The text to process
    - chunk_size: int
        The size of the chunks
    """
    assert isinstance(text, str)
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append([i, i+chunk_size])
    return chunks, [False]*len(chunks)

def chunk_text_smart(text,
    min_suspicious_lines=4,
    min_chunk_size=200,
    max_chunk_size=10000,
    min_empty_lines_for_page_break=3,
    log_timing=False
    # remove_control_characters_line_breaks=True,
    ):
    """
    Split text into chunks of text that are likely to be pages, or good/bad chunks inside a page.

    Parameters:
    - text: str
        The text to process
    - min_suspicious_lines: int
        The minimum number of suspicious lines to consider a chunk as bad
    - min_chunk_size: int
        Chunks lower than this size will be considered as possibly bad
    - max_chunk_size: int
        The (target) maximum size of a chunk
    - min_empty_lines_for_page_break: int
        The minimum number of empty lines to consider a page break
    - log_timing: bool
        Whether to log the time taken by each step
    """

    assert isinstance(text, str)
    lines = text.split("\n")
    tic = time.time()
    lines_features = list(map(compute_line_features, lines))
    if log_timing:
        print(f"Time to compute line statistics: {time.time()-tic:.2f}s")
    
    tic = time.time()
    suspicious_chunks = []
    line_breaks = []
    # num_suspicious = 0
    num_really_suspicious = 0
    num_empty = 0
    end_char = 0
    num_char_since_last_empty = 0
    num_char_since_last_suspicious = 0
    for iline, ((suspicious, is_empty, is_short, line_length), line) in enumerate(zip(lines_features, lines)):
        if iline == 0:
            start_char = 0
        else:
            start_char = end_char

            if re.match(page_break_pattern, line) and start_char > 0 and (not line_breaks or start_char > line_breaks[-1]):
                line_breaks.append(start_char)

        end_char += line_length
        really_suspicious = suspicious and not is_short

        # Set counters
        if is_empty:
            num_empty += 1
            num_char_since_last_empty += line_length
        else:
            num_empty = 0
            num_char_since_last_empty = 0

        if really_suspicious or (suspicious and num_really_suspicious):
            num_really_suspicious += 1
            num_char_since_last_suspicious += line_length
        else:
            # Reset the counters
            # num_suspicious = 0
            num_really_suspicious = 0
            num_char_since_last_suspicious = 0

        # Start to discord the lines if the number of suspicious lines is greater than the threshold
        if num_really_suspicious >= min_suspicious_lines:
            if num_really_suspicious == min_suspicious_lines or not len(suspicious_chunks):
                first_line = max(0, iline+1-num_really_suspicious)
                assert first_line >= 0
                start_char = start_char + line_length - num_char_since_last_suspicious
                suspicious_chunks.append([first_line, iline, start_char, end_char])
                # print("\n".join(lines[first_line:iline+1]))
            else:
                suspicious_chunks[-1][1] = iline
                suspicious_chunks[-1][-1] = end_char
                # print(lines[iline])
            
        else:
            # See if probable page break
            if is_empty:
                num_empty += 1
                if num_empty >= min_empty_lines_for_page_break:
                    start_char = start_char + line_length - num_char_since_last_empty
                    if start_char > 0 and (not line_breaks or start_char > line_breaks[-1]):
                        line_breaks.append(start_char)

    if log_timing:
        print(f"Accumulated line statistics: {time.time()-tic:.2f}s")
    suspicious_chunks = [(start, end) for _,_,start,end in suspicious_chunks]

    # Interleave good and bad chunks
    tic = time.time()
    max_char_indices = len(text)
    if max_char_indices == 0:
        return [], []
    # line_breaks.append(max_char_indices)
    final_chunks = []
    bad_chunks = []
    current_page = 0
    current_page_start = 0
    current_page_end = line_breaks[current_page] if current_page<len(line_breaks) else max_char_indices
    assert current_page_end > current_page_start, f"{current_page_end} <= {current_page_start} ({max_char_indices=}, ({line_breaks=} {max_char_indices=})"

    def append_good_chunks():
        nonlocal current_page, current_page_start, current_page_end, start_char
        current_page_end = min(current_page_end, start_char)
        while start_char >= current_page_end and current_page_start < max_char_indices:
            if current_page_end > current_page_start:
                assert current_page_end > current_page_start, f"{current_page_end} <= {current_page_start} ({max_char_indices=}, ({line_breaks=} {max_char_indices=})"
                final_chunks.append([current_page_start, current_page_end])
                bad_chunks.append(False)
            if current_page >= len(line_breaks):
                break
            current_page += 1
            current_page_start = current_page_end
            current_page_end = line_breaks[current_page] if current_page<len(line_breaks) else max_char_indices

    start_char = 0
    for start_char, end_char in suspicious_chunks:
        append_good_chunks()

        # Add bad chunk
        final_chunks.append([start_char, end_char])
        bad_chunks.append(True)
        current_page_start = max(current_page_start, end_char)
        while current_page_end <= current_page_start and current_page_end < max_char_indices:
            # Skip pages included in the bad chunk
            current_page += 1
            current_page_end = line_breaks[current_page] if current_page<len(line_breaks) else max_char_indices
        assert current_page_end > current_page_start or current_page_end == max_char_indices, f"{current_page_end} <= {current_page_start} ({max_char_indices=}, {current_page=}/{len(line_breaks)})"

    # Add last pages if needed
    if not final_chunks or final_chunks[-1][1] < max_char_indices:
        start_char = max_char_indices + 1
        append_good_chunks()

    if log_timing:
        print(f"Chunk post-processing 2: {time.time()-tic:.2f}s")

    # Glue small chunks with previous chunk or next one when it's bad
    tic = time.time()
    new_final_chunks = []
    new_bad_chunks = []
    for i_chunk in range(len(final_chunks)):
        start_char, end_char = final_chunks[i_chunk]
        is_bad = bad_chunks[i_chunk]
        chunk_length = end_char - start_char
        # If this is a small chunk that is not bad
        if chunk_length < min_chunk_size and not is_bad:
            previous_is_bad = i_chunk > 0 and bad_chunks[i_chunk-1]
            has_next = i_chunk < len(final_chunks)-1
            # next_is_bad = i_chunk < len(final_chunks)-1 and bad_chunks[i_chunk+1]
            if previous_is_bad:
                # Glue to previous bad page
                assert len(new_final_chunks)
                new_final_chunks[-1][1] = end_char
            elif has_next:
                # Glue to next bad page
                final_chunks[i_chunk+1][0] = start_char
        elif chunk_length > max_chunk_size:
            # Cut the chunk in pieces
            n_chunks = int(math.ceil(chunk_length / max_chunk_size))
            chunk_size = int(math.ceil(chunk_length / n_chunks))
            new_start_char = start_char 
            new_end_char = 0
            for i in range(n_chunks):
                if new_end_char >= end_char:
                    break
                new_end_char = min(new_start_char + chunk_size, end_char)
                has_linebreak = "\n" in text[new_end_char:end_char]
                while new_end_char < end_char and text[new_end_char] not in ("\n" if has_linebreak else " "):
                    new_end_char += 1
                new_final_chunks.append([new_start_char, new_end_char])
                new_bad_chunks.append(is_bad)
                new_start_char = new_end_char
        else:
            new_final_chunks.append([start_char, end_char])
            new_bad_chunks.append(is_bad)
    final_chunks = new_final_chunks
    bad_chunks = new_bad_chunks
    if log_timing:
        print(f"Chunk post-processing 3: {time.time()-tic:.2f}s")
    assert len(bad_chunks) == len(final_chunks)

    # # Checks
    # last_end_char = 0
    # for i, (start_char, end_char) in enumerate(final_chunks):
    #     if start_char != last_end_char:
    #         import pdb; pdb.set_trace()
    #     if end_char <= start_char:
    #         import pdb; pdb.set_trace()
    #     last_end_char = end_char

    return final_chunks, bad_chunks


def preview_chunked_text(text, remove_bad_chunks=True, **kwargs):
    chunks, bad_chunks = chunk_text_smart(text, **kwargs)
    if remove_bad_chunks:
        assert len(chunks) == len(bad_chunks)
        chunks = [text[start_char:end_char] for (start_char, end_char), is_bad in zip(chunks, bad_chunks) if not is_bad]
    else:
        chunks = [text[start_char:end_char] for start_char, end_char in chunks]
    return "\n>>>> Chunk Break <<<<\n".join(chunks)


if __name__ == "__main__":

    import os
    import shutil
    import argparse
    import time
    import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=str, help="File to process", nargs="+")
    args = parser.parse_args()

    files = args.files
    out_folder = None

    for f, file in enumerate(tqdm.tqdm(files)):
        file = os.path.abspath(file)
        basename = os.path.basename(file)
        basename, ext = os.path.splitext(basename)
        local_out_folder = out_folder or os.path.join(os.path.dirname(file), "cleaned")
        os.makedirs(local_out_folder, exist_ok=True)
        if out_folder:
            copy_file = f"{local_out_folder}/{basename}_raw{ext}"
            shutil.copy(file, copy_file)
            out_file = f"{local_out_folder}/{basename}_cleaned{ext}"
        else:
            out_file = f"{local_out_folder}/{basename}{ext}"

        with open(file, "r") as f:
            text = f.read()
        tic = time.time()
        text_len = len(text)
        text = preview_chunked_text(text)
        processing_time = time.time() - tic
        print(f"Processing time: {processing_time:.2f}s for {text_len} characters. ({text_len/processing_time:.2f} char/s) -> {out_file}")

        with open(out_file, "w") as f:
            f.write(text)

