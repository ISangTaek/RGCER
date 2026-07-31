import torch, time
import numpy as np

M, N = np.inf, -np.inf
class PerformanceMeter(object):
    def __init__(self, task_dict):
        
        self.task_dict = task_dict
        self.task_num = len(self.task_dict)
        self.task_name = list(self.task_dict.keys())
        self.weight = {task: self.task_dict[task]['weight'] for task in self.task_name}

        if next(iter(self.weight.values())) == [1, 1]:
            self.best_val_result = {task:[N, N] for task in self.task_name}
            self.best_test_result = {task:[N, N] for task in self.task_name}
            self.best_val_result['average'] = [N, N]
            self.best_test_result['average'] = [N, N]
        if next(iter(self.weight.values())) == [1, -1]:
            self.best_val_result = {task:[N, M] for task in self.task_name}
            self.best_test_result = {task:[N, M] for task in self.task_name}
            self.best_val_result['average'] = [N, M]
            self.best_test_result['average'] = [N, M]
        if next(iter(self.weight.values())) == [-1, 1]:
            self.best_val_result = {task:[M, N] for task in self.task_name}
            self.best_test_result = {task:[M, N] for task in self.task_name}
            self.best_val_result['average'] = [M, N]
            self.best_test_result['average'] = [M, N]
        if next(iter(self.weight.values())) == [-1, -1]:
            self.best_val_result = {task:[M, M] for task in self.task_name}
            self.best_test_result = {task:[M, M] for task in self.task_name}
            self.best_val_result['average'] = [M, M]
            self.best_test_result['average'] = [M, M]
        
        self.best_val_epoch = {task:[0, 0] for task in self.task_name}
        self.best_test_epoch = {task:[0, 0] for task in self.task_name}
        self.best_val_epoch['average'] = [0, 0]
        self.best_test_epoch['average'] = [0, 0]

        self.losses = {task: self.task_dict[task]['loss_fn'] for task in self.task_name}
        self.metrics = {task: self.task_dict[task]['metrics_fn'] for task in self.task_name}
        self.results = {task:[] for task in self.task_name}
        self.loss_item = np.zeros(self.task_num)
        self.init_display()
        # init visdom

        # summery and save TEXT and VALID results in batch
        self.cache_result = {task:{'pred':[], 'gts':[]}for task in self.task_name}

    def record_time(self, mode='begin'):
        if mode == 'begin':
            self.beg_time = time.time()
        elif mode == 'end':
            self.end_time = time.time()
        else:
            raise ValueError('No support time mode {}'.format(mode))

    def append_result(self, preds, gts, task_name = None):
        with torch.no_grad():
            self.cache_result[task_name]['pred'].append(preds)
            self.cache_result[task_name]['gts'].append(gts)
        
    def update(self, preds, gts, task_name=None):
        with torch.no_grad():
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)
        
    def get_score(self):
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                self.results[task] = self.metrics[task].score_fun()
                self.loss_item[tn] = self.losses[task].average_loss()
    
    def init_display(self):
        print('='*40)
        print('LOG FORMAT | ', end='')
        for tn, task in enumerate(self.task_name):
            print(task+'_LOSS ', end='')
            for m in self.task_dict[task]['metrics']:
                print(m+' ', end='')
            print('| ', end='')
        print('TIME')
    
    def display(self, mode, epoch):
        if epoch is not None:
            if mode == 'train':
                print('Epoch: {:03d} '.format(epoch), end='\n')
            if mode == 'val':
                self.update_best_result(self.results, self.best_val_result, epoch, self.best_val_epoch)
            if mode == 'test':
                self.update_best_result(self.results, self.best_test_result, epoch, self.best_test_epoch)

        print('{}: '.format(mode), end='')
        for tn, task in enumerate(self.task_name):
            print('{:.4f} '.format(self.loss_item[tn]), end='')
            for i in range(len(self.results[task])):
                print('{:.4f} '.format(self.results[task][i]), end='')
            print('| ', end='')
        print('Time: {:.4f}'.format(self.end_time-self.beg_time), end='')
        print(' | ', end='\n')
        # self.vis(epoch=epoch, mode = mode, results = self.results, switch = False)
    def display_best_result(self):
        print('='*40)
        print('Best Valid Result: Epoch {}, result {}'.format(self.best_val_epoch, self.best_val_result))
        print('='*40)
        print('Best Test Result: Epoch {}, result {}'.format(self.best_test_epoch, self.best_test_result))
        print('='*40)
    
    def update_best_result(self, new_result, best_result, epoch, best_epoch):

        A, C, E, A_weight, C_weight, E_weight, count = 0, 0, 0, 0, 0, 0, 0
        for task in list(new_result.keys()):
            for matric in range(len(new_result[task])):

                A_weight = self.weight[task][matric]
                A = new_result[task][matric]
                B = best_result[task][matric]
                if A_weight * A > A_weight * B:
                    best_result[task][matric] = A
                    best_epoch[task][matric] = epoch

                if count == 0 and matric == 0:
                    C_weight = self.weight[task][matric]
                if count == 0 and matric == 1:
                    E_weight = self.weight[task][matric]
                if matric == 0:
                    C = C + new_result[task][matric]
                if matric == 1:
                    E = E + new_result[task][matric]

            count = count + 1

        C, E = C / count, E / count
        D, F = best_result['average'][0], best_result['average'][1]
        if C_weight * C > C_weight * D:
            best_result['average'][0] = C
            best_epoch['average'][0] = epoch
        if E_weight * E > E_weight * F:
            best_result['average'][1] = E
            best_epoch['average'][1] = epoch
    
    def reinit(self):
        for task in self.task_name:
            self.losses[task].reinit()
            self.metrics[task].reinit()
        self.loss_item = np.zeros(self.task_num)
        self.results = {task:[] for task in self.task_name}