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

answerCounts = {0: 1245, 1: 745, 2: 103, 3: 20, 7: 2, 4: 1, 5: 1, 6: 1}
sum = 0
for k,v in answerCounts.iteritems():
    sum += v

pmf = []

for k,v in answerCounts.iteritems():
    pmf.append(float(v)/sum)
cdf = []
for i,v in enumerate(pmf):
    sum = 0
    for j in range(0,i+1):
        sum += pmf[j]
    cdf.append(sum)

print pmf
print cdf

#[0.5878186968838527, 0.35174693106704435, 0.04863078375826251, 0.009442870632672332, 0.00047214353163361664, 0.00047214353163361664, 0.00047214353163361664, 0.0009442870632672333]
#[0.5878186968838527, 0.939565627950897, 0.9881964117091595, 0.9976392823418319, 0.9981114258734655, 0.9985835694050991, 0.9990557129367327, 0.9999999999999999]
