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

    def __init__(self, log_path):
        if not os.path.isdir(log_path): os.makedirs(log_path)
        self.log_file = open(log_path + '/classifier_%d.log' % time.time(),'a')
        self.lock = threading.Lock()

    def log(self, message, color='', verbose=False):
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

    @staticmethod
    def calculate_metrics(tp, tn, fp, fn, total, rep_str):
        rep_str += "Overall Acc = \033[34m%4f\033[m\n" % (float(tp+tn)/total)
        if tp+fn:
            rep_str += "Recall = %4f\n" % (float(tp)/(tp+fn))
            rep_str += "Miss Rate = %4f\n" % (float(fn)/(tp+fn))
        if tn+fp:
            rep_str += "Specificity = %4f\n" % (float(tn)/(tn+fp))
            rep_str += "Fallout = %4f\n" % (float(fp)/(tn+fp))
        if tp+fp: rep_str += "Precision = %4f\n" % (float(tp)/(tp+fp))
        if tp+fp+fn: rep_str += "F1 score = %4f\n" % (float(2*tp)/(2*tp+fp+fn))
        if (tp+fp)*(tp+fn)*(tn+fp)*(tn+fn): rep_str += "Mcc = %4f\n" % (float((tp*tn)-(fp*fn))/numpy.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        return rep_str

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
                rep_str = self.calculate_metrics(tp, tn, fp, fn, self.n, rep_str)

            elif len(self.node.attack_keys) == 3: # Layer-1 three labels
                dos_dos, dos_ps, dos_bf, ps_dos, ps_ps, ps_bf, bf_dos, bf_ps, bf_bf = numpy.ravel(self.confusion_matrix)
                # fastdos-portscan
                tp_dos_ps = dos_dos
                tn_dos_ps = ps_ps
                fp_dos_ps = ps_dos
                fn_dos_ps = dos_ps
                total_dos_ps = dos_dos + ps_ps + ps_dos + dos_ps
                rep_str+= "\033[1;33mfastdos-portscan accuracy:\033[m\n"
                rep_str = self.calculate_metrics(tp_dos_ps, tn_dos_ps, fp_dos_ps, fn_dos_ps, total_dos_ps, rep_str)
                # fastdos-bruteforce
                tp_dos_bf = dos_dos
                tn_dos_bf = bf_bf
                fp_dos_bf = bf_dos
                fn_dos_bf = dos_bf
                total_dos_bf = dos_dos + bf_bf + bf_dos + dos_bf
                rep_str+= "\033[1;33mfastdos-bruteforce accuracy:\033[m\n"
                rep_str = self.calculate_metrics(tp_dos_bf, tn_dos_bf, fp_dos_bf, fn_dos_bf, total_dos_bf, rep_str)
                # portscan-bruteforce
                tp_ps_bf = ps_ps
                tn_ps_bf = bf_bf
                fp_ps_bf = bf_ps
                fn_ps_bf = ps_bf
                total_ps_bf = ps_ps + bf_bf + bf_ps + ps_bf
                rep_str+= "\033[1;33mportscan-bruteforce accuracy:\033[m\n"
                rep_str = self.calculate_metrics(tp_ps_bf, tn_ps_bf, fp_ps_bf, fn_ps_bf, total_ps_bf, rep_str)

            # unidentified
            diag = sum(numpy.diag(self.confusion_matrix))
            if diag - self.total_correct:
                rep_str += "Unidentified flows marked as \"%s\": \033[1;33m#%d\033[m\n" % \
                    (self.node.attack_keys[0], diag - self.total_correct)
        return rep_str
