import logging
from os.path import exists
from os import makedirs
from generate_input_file import load_questions_from_file
from collections import Counter

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    questions = {}
    questions['train'], voc, idf = load_questions_from_file('train', -1)
    questions['validate'], voc, idf = load_questions_from_file('validate', -1)
    questions['test'], voc, idf = load_questions_from_file('test', -1)
    logging.info("Questions loaded.")

    logging.info("Working on train...")

    n_answers = Counter()
    q_with_answer = 0
    q_with_answers = 0
    for split in ['train','validate','test']:
        for question in questions[split]:
            if len(question.correct_answer) > 1:
                q_with_answers += 1
            elif len(question.correct_answer) == 1:
                q_with_answer += 1

            n_answers[len(question.answers)] += 1

    print('distribution of number of answers',n_answers)
    print("For all, 1 answer: %d, many answers: %d" % (q_with_answer, q_with_answers))

answerCounts = {0: 1245, 1: 745, 2: 103, 3: 20, 7: 2, 4: 1, 5: 1, 6: 1}
for d in [answerCounts, n_answers]:
    sum = 0
    for k,v in d.iteritems():
        sum += v

    pmf = []

    for k,v in d.iteritems():
        pmf.append(float(v)/sum)
    cdf = []
    for i,v in enumerate(pmf):
        sum = 0
        for j in range(0,i+1):
            sum += pmf[j]
        cdf.append(sum)

    print pmf
    print cdf
    print len(cdf)

#answer_counts
#[0.5878186968838527, 0.35174693106704435, 0.04863078375826251, 0.009442870632672332, 0.00047214353163361664, 0.00047214353163361664, 0.00047214353163361664, 0.0009442870632672333]
#[0.5878186968838527, 0.939565627950897, 0.9881964117091595, 0.9976392823418319, 0.9981114258734655, 0.9985835694050991, 0.9990557129367327, 0.9999999999999999]
#8
#n_answers
# [0.027266754270696452, 0.05387647831800263, 0.06701708278580815, 0.0821287779237845, 0.07917214191852825, 0.07325886990801576, 0.0735873850197109, 0.0683311432325887, 0.05420499342969776, 0.051248357424441525, 0.04894875164257556, 0.04402102496714849, 0.03777923784494087, 0.03580814717477004, 0.02956636005256242, 0.026938239159001315, 0.028580814717477004, 0.01938239159001314, 0.011498028909329829, 0.020367936925098553, 0.013797634691195795, 0.00821287779237845, 0.011498028909329829, 0.007227332457293035, 0.0016425755584756898, 0.0036136662286465177, 0.00492772667542707, 0.0042706964520367935, 0.0045992115637319315, 0.007227332457293035]
# [0.027266754270696452, 0.08114323258869909, 0.14816031537450725, 0.23028909329829172, 0.30946123521682, 0.3827201051248358, 0.4563074901445467, 0.5246386333771353, 0.5788436268068331, 0.6300919842312747, 0.6790407358738502, 0.7230617608409986, 0.7608409986859395, 0.7966491458607095, 0.8262155059132719, 0.8531537450722732, 0.8817345597897501, 0.9011169513797632, 0.912614980289093, 0.9329829172141916, 0.9467805519053873, 0.9549934296977658, 0.9664914586070956, 0.9737187910643886, 0.9753613666228643, 0.9789750328515109, 0.983902759526938, 0.9881734559789748, 0.9927726675427067, 0.9999999999999997]
# 30