import json
import numpy as np
import stanza
#用于得到依赖关系
def get_dependency(tokens):
    nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse,constituency',
                          tokenize_pretokenized=True,download_method=None)

    result = []
    result2 = []
    POS = []
    Pre_head = []
    for idx,token in enumerate(tokens):
        # sente =' '.join(['Pairing', 'it', 'with', 'an', 'iPhone', 'is', 'a', 'pure', 'pleasure', '-', 'talk', 'about', 'painless', 'syncing', '-', 'used', 'to', 'take', 'me', 'forever', '-', 'now', 'it', "'s", 'a', 'snap', '.'])
        # sente =' '.join(['she', 'is', 'a', 'beautiful', 'woman','.'])
        # sente = "It is really easy to use and it is quick to start up ."
        # sente = "great food but the service was dreadful!"


        sente = ' '.join(token)
        # if idx==627:
        #     print(idx)
        # token_list= ['The', 'wait', 'staff', 'is', 'pleasant', ',', 'fun', ',', 'and', 'for', 'the', 'most', 'part', 'gorgeous', '(', 'in', 'the', 'wonderful', 'aesthetic', 'beautification', 'way', ',', 'not', 'in', 'that', "she's-way-cuter-than-me-that-b", '@', '']
        # token_dicts = [{'id': str(i + 1), 'text': token} for i, token in enumerate(token)]
        try:
            doc = nlp(sente)
        except:
            print(idx,token)
        # break
        # 获取第一个句子的依存分析结果
        sent = doc.sentences
        dependencies = sent[0].dependencies
        dd = []
        pos = []
        for dependency in dependencies:
            this_word = dependency[2]

            token_id = this_word.id
            token_head_id = this_word.head
            token_dependency_label = this_word.deprel

            # 将依存关系转换成['root',1,2]形式
            if token_head_id == 0:
                dd.append(['root', token_head_id,token_id])
            else:
                dd.append([token_dependency_label, token_head_id, token_id])
            pos.append(this_word.pos)
        result.append(dd)
        dd2 = [""+e[0] for e in dd]
        POS.append(pos)
        result2.append(dd2)
        prehead = [e[1] for e in dd]
        Pre_head.append(prehead)
    return result2,result,POS,Pre_head
def tackle_dataset(dataset_list):
    dataset = dataset_list
    sentences = []
    for data in dataset_list:
        sentence = data['sentences']
        sentences.append(sentence)
    predicted_dependencies, dependencies,POS,prehead = get_dependency(sentences)
    # predicted_dependencies, dependencies,POS,prehead = get_dependency(dataset)
    t = []
    for id,data in enumerate(dataset_list):
        depend = dependencies[id]

        # 构建邻接矩阵
        matrix = np.zeros((len(data['sentences']), len(data['sentences'])), dtype=int)
        for idx, token in enumerate(depend):
            try:
                matrix[token[2]-1][token[1]-1] = 1
                matrix[token[1]-1][token[2]-1] = 1
                matrix[idx][idx] = 1
            except:
                matrix[token[2] - 1][token[1] - 1] = 1
                matrix[token[1] - 1][token[2] - 1] = 1
                matrix[idx][idx] = 1
        dic = {}
        dic['token'] = data
        dic['graph'] = matrix.tolist()
        t.append(dic)
    return t

def read_sentence_depparsed(file_path):
    with open(file_path, 'r',encoding='utf-8') as f:
        data = json.load(f)
        return data

dataset = '14res'
# train = list(read_sentence_depparsed(f'./scierc_1_sim/{dataset}/train.json'))
# test = list(read_sentence_depparsed(f'./scierc_1_sim/{dataset}/test.json'))
test = list(read_sentence_depparsed(f'./scierc_1_sim/text4test.json'))
# dev = list(read_sentence_depparsed(f'./scierc_1_sim/{dataset}/dev.json'))

# train = tackle_dataset(train)
test = tackle_dataset(test)
# dev = tackle_dataset(dev)

# with open(f'scierc_1_sim/{dataset}/train_test.json','w', encoding = 'utf-8') as f:
#     json.dump(train,f,ensure_ascii=False)
# with open(f'scierc_1_sim/{dataset}/test_test.json','w', encoding = 'utf-8') as f:
#     json.dump(test,f,ensure_ascii=False)
# with open(f'scierc_1_sim/{dataset}/dev_test.json','w', encoding = 'utf-8') as f:
#     json.dump(dev,f,ensure_ascii=False)
with open(f'./scierc_1_sim/text4testt.json','w', encoding = 'utf-8') as f:
    json.dump(test,f,ensure_ascii=False)