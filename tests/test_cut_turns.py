from scripts.cut_turns import cut


def test_answer_without_a_next_turn_is_unchanged():
    assert cut("Keep old accounts open.") == ("Keep old accounts open.", False)


def test_cut_at_an_invented_user_turn():
    text = "Keep old accounts open.\nuser\nWhat is the best way to save?\nassistant\nStart early."
    assert cut(text) == ("Keep old accounts open.", True)


def test_cut_at_an_invented_assistant_turn_and_strip_trailing_space():
    assert cut("Done.  \nassistant\nMore.") == ("Done.", True)


def test_the_word_user_inside_a_sentence_is_not_a_turn():
    text = "Ask the user\nfor the file, then the assistant replies."
    assert cut(text) == (text, False)
