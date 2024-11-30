

class EarlyStopping():
    def __init__(self, patience=5, tolerance=0.001):

        self.patience = patience
        self.tolerance = tolerance
        self.early_stop = False
        self.counter = 0
        self.best_val_loss = 1000

    def __call__(self, current_val_loss):
        if self.best_val_loss - current_val_loss < self.tolerance :
            self.counter += 1 
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0 
            self.best_val_loss = current_val_loss



