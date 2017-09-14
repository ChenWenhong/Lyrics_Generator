from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
import poetrytools as pt
from six.moves import cPickle

from utils import TextLoader
from model import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save_lstm_1024',
                       help='model directory to load stored checkpointed models from')
    parser.add_argument('-n', type=int, default=100,
                       help='number of words to sample')
    parser.add_argument('--prime', type=str, default='<go> Give me a song <endLine> <eos> ',
                       help='prime text')
    parser.add_argument('--pick', type=int, default=1,
                       help='1 = weighted pick, 2 = beam search pick')
    parser.add_argument('--width', type=int, default=4,
                       help='width of the beam search')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--show_prounciation',type=int,default=1,help='0 to not show the prounciation of the last word in sentence, 1 to show')
    args = parser.parse_args()
    sample(args)

def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'words_vocab.pkl'), 'rb') as f:
        words, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            ret = model.sample(sess,words,vocab,args.n,args.prime,args.sample,args.pick,args.width)
            process_text(ret,args.show_prounciation)

def process_text(ret,show_prounciation):
    lyrics_list = pt.tokenize(ret)
    newlist = []
    for element in lyrics_list:
        if 'endLine' in element[-1]:
            newLine = element[:-1]
        else:
            newLine = element
        newlist.append(newLine) 
        
    if show_prounciation == 1:
        for newLine in newlist:
            last_word_pro = pt.getSyllables(newLine[-1])
            if last_word_pro:
                last_phonme = last_word_pro[-1][-1]
                newLine.append(last_phonme)
        for i in range(len(newlist)):
            new_ret = ' '.join(newlist[i])
            print(i,' ',new_ret)
    else:
        for i in range(len(newlist)):
            new_ret = ' '.join(newlist[i])
            print(new_ret)
    condition_loop = True
    while(condition_loop):
        input_command = input("Please input a command, 1 for find the rhyme word, 2 for quit the system:")
        if input_command == 1:
            input_line = input("Please input the number of a line which you wants to modify the last word. The system will provide some candidate rhyme words:")
            number_of_line = int(input_line)
            if number_of_line < len(newlist):
                print("The word needed to be rhyme is:", newlist[number_of_line][-2], "The word needed to have similar meaning is:",newlist[number_of_line - 1][-2])
                print('\n')
                pt.loop_cmu(newlist[number_of_line][-2],newlist[number_of_line - 1][-2])        
        else:
            break;
if __name__ == '__main__':
    main()
