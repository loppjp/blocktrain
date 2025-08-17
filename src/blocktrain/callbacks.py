class ICallback:

    def on_train_step_start(self): pass
    def on_train_step_end(self): pass

    def on_train_epoch_start(self): pass
    def on_train_epoch_end(self): pass

    def on_eval_step_start(self): pass
    def on_eval_step_end(self): pass

    def on_training_start(self): pass
    def on_training_end(self): pass

    def on_eval_start(self): pass
    def on_eval_end(self): pass