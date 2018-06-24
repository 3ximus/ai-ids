"""This file contains the class Stats and Logger module

AUTHORS:

Joao Meira <joao.meira@tekever.com>
Fabio Almeida <fabio.4335@gmail.com>
"""

from sklearn.metrics import accuracy_score, confusion_matrix
import time, numpy, threading, os


class Logger:
    error = '[\033[1;31mERROR\033[m]'
    warning = '[\033[1;33mWARNING\033[m]'
    normal = '[\033[1;34m LOG \033[m]'

    def __init__(self):
        if not os.path.isdir('log'): os.makedirs('log')
        self.log_file = open('log/classifier_2levels_run_%d.log' % time.time(),'a')
        self.lock = threading.Lock()

    def log(self, color, message, verbose=False):
        ''' Log messages to a the Logger file and to screen if verbose'''
        with self.lock:
            string = '%s %s %s' % (color, threading.current_thread().getName(), message)
            if verbose or color == self.error: print(string)
            self.log_file.write(string + '\n')

class Stats:
    '''Holds stats from predictions. Can be updated multiple times to include more stats on tests with same labels'''

    def __init__(self, node):
        self.node = node
        self.n = self.total_correct = 0
        self.confusion_matrix = numpy.matrix([[0 for x in range(len(node.outputs))] for x in range(len(node.outputs))])
        self.lock = threading.Lock()

    def update(self, y_predicted, y_test):
        '''Update stats values with more results. Thread safe.

            Parameters
            ----------
            - y_predicted     numpy list of predict NN outputs
            - y_test          numpy list of target outputs
        '''

        with self.lock:
            self.total_correct += accuracy_score(y_test, y_predicted, normalize=False) # counts only elements not classified as [0,0..,0]
            # TODO FIXME Using numpy.argmax puts unclassified entries ([0,0,...,0]) as label zero, see previous code to count those
            x = numpy.argmax(y_test, axis=1) if len(y_test.shape) == 2 else y_test
            y = numpy.argmax(y_predicted, axis=1) if len(y_predicted.shape) == 2 else y_predicted
            self.confusion_matrix += numpy.matrix(confusion_matrix(x, y, labels=list(range(len(self.node.attack_keys)))))
            self.n = self.confusion_matrix.sum()

    def __repr__(self):
        with self.lock:
            # confusion matrix
            lmsize = max(map(len, self.node.attack_keys[:-1])) # for output formatting
            rep_str = " Real\\Pred |" + ''.join([('%' + str(lmsize) + 's ') % label for label in self.node.attack_keys]) + "\n"
            for i, label in enumerate(self.node.attack_keys):
                rep_str += "%10s |" % label
                for j in range(len(self.node.attack_keys)):
                    rep_str += (("\033[1;32m" if i == j else '') + "%" + str(lmsize) + "d\033[m ") % self.confusion_matrix[i,j]
                rep_str += "\n"

            # stats
            if len(self.node.attack_keys) == 2: # MALIGN OR BENIGN
                if numpy.argmax(self.node.outputs['MALIGN']) == 1:
                    tn, fp, fn, tp = numpy.ravel(self.confusion_matrix)
                else:
                    tp, fn, fp, tn = numpy.ravel(self.confusion_matrix)

                rep_str += "Overall Acc = \033[34m%4f\033[m\n" % (float(tp+tn)/float(self.n))
                if self.node.verbose:
                    if tp+fn:
                        rep_str += "Recall = %4f\n" % (float(tp)/float(tp+fn))
                        rep_str += "Miss Rate = %4f\n" % (float(fn)/(tp+fn))
                    if tn+fp:
                        rep_str += "Specificity = %4f\n" % (float(tn)/float(tn+fp))
                        rep_str += "Fallout = %4f\n" % (float(fp)/float(tn+fp))
                    if tp+fp: rep_str += "Precision = %4f\n" % (float(tp)/float(tp+fp))
                    if tp+fp+fn: rep_str += "F1 score = %4f\n" % (float(2*tp)/float(2*tp+fp+fn))
                    if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn): rep_str += "Mcc = %4f\n" % (float((tp*tn)-(fp*fn))/float(numpy.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))

            # unidentified
            diag = sum(numpy.diag(self.confusion_matrix))
            if diag - self.total_correct:
                rep_str += "Unidentified flows marked as \"%s\": \033[1;33m#%d\033[m\n" % \
                    (self.node.attack_keys[0], diag - self.total_correct)
        return rep_str
