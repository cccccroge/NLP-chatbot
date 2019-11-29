import gpt_2_simple as gpt2
import tensorflow as tf

def gen_story(mode, input_text, out_length=1023):
    
    sess = gpt2.start_tf_sess()

    check_dir = 'tf_model/355M_' + mode
    gpt2.load_gpt2(sess, checkpoint_dir=check_dir)
    text = gpt2.generate(sess, return_as_list=True, 
        checkpoint_dir=check_dir, prefix=input_text, length=out_length, include_prefix=True)

    # tf.reset_default_graph()
    # sess.close()

    return text[0]