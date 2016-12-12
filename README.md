# predict_slow_algorithm
Takes a series that is being extended slowly, and approximates the value to be expected a few iterations in the future. Can be applied to any optimization procedure, for example, to skip ahead a few steps every so often.

Exmaple usage:

Suppose Increment is a class defined as follows:

    from time import sleep
    
    class Increment(object):
        def __init__(self):
            self.parameter1 = 0
            self.parameter2 = 0

        def increment(self):
            sleep(0.1)
            print 'Tick'
            self.parameter1 += 1
            self.parameter2 -= 1

The following code would iteratively run the increment method, tracking the parameter1 and parameter2 attributes. The PredictSlowAlgorithm object would run inc's increment method 25 times. For the first fifteen iterations, it would not intervene; it would only train its predictor. Thereafter, after every three iterations of inc.increment, it would jump 8 steps into the future, updating inc.parameter1 and inc.parameter2 accordingly.

    inc = Increment()
    psa = PredictSlowAlgorithm(inc, 'increment', ['parameter1', 'parameter2'])
    psa.run_algorithm_with_predictions(max_iter=25, len_hist_used=3, n_steps_future=8, wait_to_model=15)
