from multiprocessing import Process, Queue
from sklearn.linear_model import LinearRegression
from time import sleep
import copy
import numpy as np

class PredictSlowAlgorithm(object):
    def __init__(self, alg_object, method_name, attribute_names, method_args=[], method_kwargs={}):
        self.parameters_history = []
        self.parameters_q = Queue()
        self.suggestion_q = Queue()
        self.interrupt_q = Queue()
        self.alg_object_q = Queue()
        self.alg_object = alg_object
        self.method_name = method_name
        self.method_args = method_args
        self.method_kwargs = method_kwargs
        self.attribute_names = attribute_names
        self.p1 = None

    def method(self, max_iter, incorporate_predictor_suggestions=False, interruptible=False):
        for _ in xrange(max_iter):
            method = getattr(self.alg_object, self.method_name)
            method(*self.method_args, **self.method_kwargs)
            if interruptible:
                try:
                    if self.interrupt_q.get_nowait() == 0:
                        self.alg_object_q.put(self.alg_object)
                        return
                except:
                    pass
            self.parameters_q.put([getattr(self.alg_object, attribute_name) for attribute_name in self.attribute_names])
            if incorporate_predictor_suggestions:
                try:
                    new_params = self.suggestion_q.get_nowait()
                except:
                    new_params = None
                if new_params:
                    for attribute_name, new_param in zip(self.attribute_names, new_params):
                        setattr(self.alg_object, attribute_name, new_param)
                        self.parameters_q.put(None)
        self.alg_object_q.put(self.alg_object)
        sleep(0.5)
        self.parameters_q.put('break')

    def start_method(self, max_iter, incorporate_predictor_suggestions=False, interruptible=False):
        self.p1 = Process(target=self.method, args=(max_iter, incorporate_predictor_suggestions, interruptible))
        self.p1.start()

    def join_method(self):
        self.p1.join()

    def interrupt_method(self):
        self.interrupt_q.put(0)

    def update_parameters_history(self, max_iter=None, wait=False):
        n = 0
        while n != max_iter:
            try:
                self.parameters_history.append(self.parameters_q.get_nowait())
                n += 1
                if self.break_out():
                    return
            except:
                if wait:
                    def wait_with_checking():
                        if not self.break_out():
                            try:
                                self.parameters_history.append(self.parameters_q.get(timeout=1))
                            except:
                                wait_with_checking()
                        else:
                            return
                    wait_with_checking()
                if (len(self.parameters_history) > 0 and self.parameters_history[-1] == None) or (len(self.parameters_history) > 1 and self.parameters_history[-2] == None) and not self.break_out():
                    self.parameters_history = []
                return
            if (len(self.parameters_history) > 0 and self.parameters_history[-1] == None) or (len(self.parameters_history) > 1 and self.parameters_history[-2] == None) and not self.break_out():
                self.parameters_history = []
        print "Not all parameters on the queue were added to parameters history."

    def suggest_parameters(self, lst_parameters):
        try:
            self.suggestion_q.get_nowait()
        except:
            pass
        self.suggestion_q.put(lst_parameters)

    def create_new_PSA_from_estimate(self, lst_parameters):
        new_obj = copy.deepcopy(self.alg_object)
        for attribute_name, param in zip(self.attribute_names, lst_parameters):
            setattr(new_object, attribute_name, param)
        return PredictSlowAlgorithm(new_obj, self.method_name, self.attribute_names, method_args=self.method_args, method_kwargs=self.method_kwargs)

    def train_independent_estimators(self, num_rows_param_hist, n_steps_future):
        param_hist = np.array(self.parameters_history)
        estimators = []
        for x in xrange(param_hist.shape[1]):
            hist = param_hist[:,x]
            n_rows = len(hist) - n_steps_future - num_rows_param_hist + 1
            X = []
            y = []
            for i in xrange(n_rows):
                X.append(hist[i:i+num_rows_param_hist])
                y.append(hist[i+num_rows_param_hist+n_steps_future-1])
            X = np.array(X)
            y = np.array(y)
            estimator = LinearRegression()
            estimator.fit(X,y)
            estimators.append(estimator)
        return estimators

    def predict_parameters_independently(self, estimators, num_rows_param_hist):
        param_hist = np.array(self.parameters_history)
        if len(estimators) != param_hist.shape[1]:
            raise ValueError('There must an estimator (trained or not) for each parameter being tracked.')
        X = param_hist[-num_rows_param_hist:]
        yhat = []
        for i, estimator in enumerate(estimators):
            yhat.append(estimator.predict(X[:,i].reshape(1,-1))[0])
        return yhat

    def break_out(self):
        return len(self.parameters_history) > 0 and self.parameters_history[-1] == 'break'

    def run_algorithm_with_predictions(self, max_iter, len_hist_used, n_steps_future, wait_to_model):
        psa.start_method(max_iter=max_iter, incorporate_predictor_suggestions=True)
        estimators = None
        while True:
            psa.update_parameters_history(wait=True)
            if psa.break_out():
                break
            print psa.parameters_history
            if len(psa.parameters_history) > wait_to_model:
                estimators = psa.train_independent_estimators(len_hist_used,n_steps_future)
            if estimators:
                psa.update_parameters_history()
                if psa.break_out():
                    break
                if len(psa.parameters_history) >= len_hist_used:
                    new_params = psa.predict_parameters_independently(estimators, len_hist_used)
                    print new_params
                    psa.suggest_parameters(new_params)
            if psa.break_out():
                break
        inc = psa.alg_object_q.get_nowait()
        psa.join_method()

class Increment(object):
    def __init__(self):
        self.parameter1 = 0
        self.parameter2 = 0

    def increment(self):
        sleep(0.1)
        print 'Tick'
        self.parameter1 += 1
        self.parameter2 -= 1

if __name__ == '__main__':
    inc = Increment()
    psa = PredictSlowAlgorithm(inc, 'increment', ['parameter1', 'parameter2'])
    psa.run_algorithm_with_predictions(25, 3, 8, 15)
