import logging
from os.path import exists
from os import makedirs
from generate_input_file import load_questions_from_file

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    questions = {}
    questions['train'], voc, idf = load_questions_from_file('train', -1)
    #questions['validate'], voc, idf = load_questions_from_file('validate', -1)
    #questions['test'], voc, idf = load_questions_from_file('test', -1)
    logging.info("Questions loaded.")

    logging.info("Working on train...")

    q_with_answer = 0
    q_with_answers = 0
    for question in questions['train']:
        if len(question.correct_answer) > 1:
            q_with_answers += 1
        elif len(question.correct_answer) == 1:
            q_with_answer += 1

    print("For train, 1 answer: %d, many answers: %d" % (q_with_answer, q_with_answers))