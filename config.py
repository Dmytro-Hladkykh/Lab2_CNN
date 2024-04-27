class Config:
    def __init__(self):
        self.batch_size = 64
        self.lr = 0.001
        self.num_epochs = 25
        self.train_val_split = 0.7
        self.train_new_model = True
        self.model_path = "trained_model.pth"
        self.num_batches = 5  
        self.combination_method = "physical"  
        self.selected_batches = [1, 3, 4]  