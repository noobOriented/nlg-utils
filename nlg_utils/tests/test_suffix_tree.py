from ..suffix_tree import SuffixTree


def test_longest_substring_length():
    suffix_tree = SuffixTree()
    word = 'aabbaabxbbbxb$'
    suffix_tree.add(word)

    assert suffix_tree.longest_substring_length("abccxxbaab") == 4


def test_all_occurrences():
    suffix_tree = SuffixTree()
    word = 'aabbaabxbbbxb$'
    suffix_tree.add(word)

    assert all(
        word[start: start + 2] == 'ab'
        for start in suffix_tree.all_occurrences('ab')
    )
