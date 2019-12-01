import gpt_2_simple as gpt2
from emotion_classification import predict

import collections, math


def gen_story(mode, input_text, out_length=1023):
    sess = gpt2.start_tf_sess()

    check_dir = 'tf_model/355M_' + mode
    gpt2.load_gpt2(sess, checkpoint_dir=check_dir)
    text = gpt2.generate(sess, return_as_list=True, 
        checkpoint_dir=check_dir, prefix=input_text, length=out_length, include_prefix=True)

    # tf.reset_default_graph()
    # sess.close()

    return text[0]


def identify_emotion(input_text):
    df_prob = predict(input_text)

    labels_dict = collections.OrderedDict()
    labels_dict = {"love":3819.0/4000, "empty":826.0/4000, "surprise":2181.0/4000, "relief":1524.0/4000, 
                "neutral":8619.0/4000, "anger":110.0/4000, "sadness":5162.0/4000, "hate":1322.0/4000,
                "enthusiasm":753.0/4000, "worry":8452.0/4000, "boredom":179.0/4000, "happiness":5180.0/4000, 
                "fun":1773.0/4000}

    # calculate weight
    z = []
    for item in labels_dict:
        z.append((1.0/labels_dict[item])**0.35)
    z_exp = [math.exp(i) for i in z]
    sum_z_exp = sum(z_exp)
    softmax = [round(i / sum_z_exp, 3)*10 for i in z_exp]

    # multiply by weight
    for i in range(len(labels_dict)):
        labels = list(labels_dict.items())[i][0]
        labels_dict[labels] = df_prob[i] * softmax[i]

    # print(labels_dict)
    labels = sorted(labels_dict.items(), key=lambda d: d[1], reverse = True)
    # print(labels[0:2])
    
    # return top1 or top2 if difference is small
    top1_type = labels[0][0]
    top1_val = labels[0][1]

    top2_type = labels[1][0]
    top2_val = labels[1][1]

    if (top1_val - top2_val) / top1_val < 0.125:
        return (top1_type, top2_type)
    else:
        return (top1_type, )


