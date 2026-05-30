from stable_whisper.text_output import sec2srt

def optimize_text(text, max_lines_per_segment, line_penalty, longest_line_char_penalty):
    text = text.strip()
    words = text.split()

    psum = [0]
    for w in words:
        psum += [psum[-1] + len(w) + 1]  

    bestScore = 10 ** 30
    bestSplit = None

    def backtrack(level, wordsUsed, maxLineLength, split):
        nonlocal bestScore, bestSplit

        if wordsUsed == len(words):
            score = level * line_penalty + maxLineLength * longest_line_char_penalty
            if score < bestScore:
                bestScore = score
                bestSplit = split
            return

        if level + 1 == max_lines_per_segment:
            backtrack(
                level + 1, len(words),
                max(maxLineLength, psum[len(words)] - psum[wordsUsed] - 1),
                split + [words[wordsUsed:]]
            )
            return

        for levelWords in range(1, len(words) - wordsUsed + 1):
            backtrack(
                level + 1, wordsUsed + levelWords,
                max(maxLineLength, psum[wordsUsed + levelWords] - psum[wordsUsed] - 1),
                split + [words[wordsUsed:wordsUsed + levelWords]]
            )

    backtrack(0, 0, 0, [])

    if not bestSplit:
        return text
        
    if len(bestSplit) > max_lines_per_segment or any(len(line) == 1 for line in bestSplit):
        return text

    optimized = '\n'.join(' '.join(words) for words in bestSplit)
    return optimized

def segment2optimizedsrtblock(segment: dict, idx: int, max_lines_per_segment, line_penalty, longest_line_char_penalty, strip=True) -> str:
    return f'{idx}\n{sec2srt(segment["start"])} --> {sec2srt(segment["end"])}\n' \
           f'{optimize_text(segment["text"], max_lines_per_segment, line_penalty, longest_line_char_penalty)}'

def segments2blocks(segments, max_lines_per_segment, line_penalty, longest_line_char_penalty):
    return '\n\n'.join(
        segment2optimizedsrtblock(s, i, max_lines_per_segment, line_penalty, longest_line_char_penalty, strip=True)
        for i, s in enumerate(segments)
    )