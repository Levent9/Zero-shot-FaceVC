import os
import errno
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)
        self.logdir = logdir
        # mulit process
        #https://stackoverflow.com/questions/12468022/python-fileexists-error-when-making-directory

        os.makedirs(logdir,exist_ok=True)
        self.logfile = os.path.join(self.logdir, 'train.log')

    def log_training(self, metrics, iteration):
        for k, v in metrics.items():
            self.add_scalar('training/%s' % k, v, iteration)    

    def log_validation(self, keys, values, iteration):
        for k, v in zip(keys, values):
            self.add_scalar('validation/%s' %k, v, iteration) 
            
               
    def log_epoch(self, train_metrics, val_metrics, 
        train_keys, valid_keys, train_time, total_time, epoch, ali=None):
        
        for k,v in zip(train_keys, train_metrics):
            self.add_scalar("training/%s" %k, v, epoch)

        for k,v in zip(valid_keys, val_metrics):
            self.add_scalar('validation/%s' %k, v, epoch)

        if (not os.path.exists(self.logfile)) or epoch == 0:
            with open(self.logfile, 'w') as fp:
                fp.write('Epoch, Eptime, ')
                fp.write(', '.join(train_keys))
                fp.write(', ')
                fp.write(', '.join(valid_keys))
                fp.write('\n')
        
        with open(self.logfile, 'a') as fp:
            print('Epoch {}, Train {:.1f} min, Total {:.1f} min'.format(
                epoch, train_time/60, total_time/60))
            # print('Train Loss: {:.4f}'.format(train_metrics[0]))
            # print('Valid Loss: {:.4f}'.format(val_metrics[0]))
            for idex, i in enumerate(valid_keys):
                print('Valid ' + i+': {:.4f}'.format(val_metrics[idex]))

            fp.write('E {}, {:.1f}, '.format(epoch, total_time/60))
            fp.write(', '.join(['{:.4f}'.format(v) for v in train_metrics]))
            fp.write(', ')
            fp.write(', '.join(['{:.4f}'.format(v) for v in val_metrics]))
            fp.write('\n')
        
        # if not ali is None:
        #     ali = ali.detach().cpu().numpy() #
        #     plot_alignment(ali, self.ali_path+'/epoch-att-%d.pdf'%epoch)