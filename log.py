from tensorboardX import SummaryWriter

class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)

    def logs_training(self, reduced_losses, grad_norms, learning_rates, durations, iteration):
        for i in range(len(reduced_losses)):
            self.log_training(reduced_losses[i], grad_norms[i], learning_rates[i], durations[i], iteration+i)

    def logs_validation(self, reduced_losses, iterations):
        for i in range(1, len(reduced_losses)):
            self.log_validation(reduced_losses[i], iterations[i])

    def log_training(self, total_loss, f1, accuracy, iteration):
        self.add_scalar("training.loss", total_loss, iteration)
        self.add_scalar("training.F1", f1, iteration)
        self.add_scalar("training.accuracy", accuracy, iteration)

    def log_validation(self, dev_loss, dev_f1, dev_accuracy, test_loss, test_f1, test_accuracy, iteration):

        self.add_scalar("validation.loss", dev_loss, iteration)
        self.add_scalar("validation.F1", dev_f1, iteration)
        self.add_scalar("validation.accuracy", dev_accuracy, iteration)
        self.add_scalar("test.loss", test_loss, iteration)
        self.add_scalar("test.F1", test_f1, iteration)
        self.add_scalar("test.accuracy", test_accuracy, iteration)
