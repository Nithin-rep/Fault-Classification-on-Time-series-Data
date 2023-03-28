import os
import yaml
import torch
import datetime
import optuna.samplers

from data.hyd_data.data_pro_files.dataloaders import HydDataLoader
from data.hyd_data.data_pro_files.writing_all_sensor_data \
    import reading_all_sensors
from tunning.optuna_opt import Optimizer
from training.hyd_te_trainer import Trainer
from AutoEncoding.Autoencoder import AutoEnc
from data.te_data.te_dataloader import TeData
from net.net_hyd_te import Hyd_TE_Net, Hyd_TE_QuatNet


class HydraulicTE:
    def __init__(self, device, train_data, train_label, test_data, test_label,
                 val_data, val_label, loc_main, yaml_file, val_acc_list,
                 timeline):
        """Giving access for common parameters.

        Parameters
        ----------
        device : str
            The processor(cpu or gpu) selection to perform calculations.
        train_data : Tensor
            Training data.
        train_label : Tensor
            Labels for all the training data.
        test_data : Tensor
            Testing data.
        test_label : Tensor
            Labels for all the testing data.
        val_data : Tensor
            Validation data.
        val_label : Tensor
            Labels for all the validation data.
        loc_main : str
            Directory where results are to be stored.(with timeline)
        yaml_file : yaml
            Configurations setting for the variables or switches in the code.
        val_acc_list : list
            List in which best validation accuracy of the trials get appended.
        timeline : str
            To differenciate between the directories (using datetime libraray).

        """

        self.device = device

        self.train_parameter = train_data
        self.outcomes_train = train_label

        self.test_parameter = test_data
        self.outcomes_test = test_label

        self.val_parameter = val_data
        self.outcomes_val = val_label

        self.loc_main = loc_main
        self.yaml_file = yaml_file
        self.val_acc_list = val_acc_list
        self.timeline = timeline

    def main(self, combinations):
        """Calculates the best validation accuracy of the configuration.

        Parameters
        ----------
        combinations : dict
            Hyperparameters to be used in the trial.

        Returns
        -------
        val_accuracy : float
            Best validation accuracy of the trial.

        """

        num_epoch = self.yaml_file["total_epochs"]
        workers = self.yaml_file["n_workers"]
        batch_size = 2 **(combinations["batch_size"])

        # To hold the parameters (for normal or auto-encoded input switch)
        train_data = self.train_parameter
        test_data = self.test_parameter
        val_data = self.val_parameter

        loc = os.path.join(self.loc_main, "trial_" +
                           str(len(self.val_acc_list)))


        # Switch in the network architecture (normal or quaternion)
        if self.yaml_file["switch_to_quaternion"]:
            net = Hyd_TE_QuatNet(combinations, self.yaml_file).to(self.device)

        else:
            net = Hyd_TE_Net(combinations, self.yaml_file).to(self.device)


        # To calculate the trainable parameters and view network

        torch.save(net, os.path.join(loc_main, 'net'))
        # print(net)
        # print('conv1_parameters', sum(p.numel() 
        #       for p in net.conv_layers[0].parameters() if p.requires_grad))


        # print('Conv2_parameters', sum(p.numel()
        #         for p in net.conv_layers[4].parameters() if p.requires_grad))
        
        
        # print('Linear1_parameters', sum(p.numel()
        #       for p in net.linear_layers[0].parameters() if p.requires_grad))
        # print('Linear2_parameters', sum(p.numel()
        #       for p in net.linear_layers[3].parameters() if p.requires_grad))
        # print('Linear3_parameters', sum(p.numel()
        #       for p in net.linear_layers[6].parameters() if p.requires_grad))


        # print('Linear4_parameters', sum(p.numel()
        #         for p in net.linear_layers[9].parameters() if p.requires_grad))


        # To calculate the parameters in neural network which require gradients
        grad = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Total Parameters which need grad are {}".format(grad))

        optim_choice = getattr(torch.optim, combinations["optimizer"])
        optimizer = optim_choice(net.parameters(), lr=combinations["lr"])


        # Training AutoEncoder
        if yaml_file['autoencoder']:

            training_dataloader = torch.utils.data.DataLoader(
                                    train_data, batch_size=batch_size,
                                    pin_memory=False, num_workers=workers,
                                    shuffle=True)

            testing_dataloader = torch.utils.data.DataLoader(
                                    test_data, batch_size=batch_size,
                                    pin_memory=False, num_workers=workers,
                                    shuffle=True)

            if yaml_file['mode'] == 'hyd':
                val_dataloader = torch.utils.data.DataLoader(
                                        val_data, batch_size=batch_size,
                                        pin_memory=False, num_workers=workers,
                                        shuffle=True)

            elif yaml_file['mode'] == 'TE':
                val_dataloader = testing_dataloader

            output_features = yaml_file["ae_conv_output_features"]
            ae_lr = float(yaml_file["ae_lr"])

            encoding = AutoEnc(yaml_file, output_features, ae_lr,
                               ae_trial_loc=loc,
                               ae_time_loc=self.loc_main,
                               device=device)

            #  normal input data is replaced by autoencoder generted input data
            train_data, test_data, val_data, test_loss = encoding.autoencoder(
                training_dataloader, testing_dataloader, val_dataloader)


            # need to be modified later to use autoencoded data in architecture
            # used to avoid the training and tuning and simply getting
            # auto-encoder regression loss

            return test_loss


        # If autoencoder; encoder generated results are used in dataset.
        # Else actual inputs are used
        training_tensor = torch.utils.data.TensorDataset(train_data,
                                                         self.outcomes_train)

        testing_tensor = torch.utils.data.TensorDataset(test_data,
                                                        self.outcomes_test)

        validation_tensor = torch.utils.data.TensorDataset(
                                val_data, self.outcomes_val)

        train_dataloader = torch.utils.data.DataLoader(training_tensor,
                                                       batch_size=batch_size,
                                                       pin_memory=False,
                                                       num_workers=workers,
                                                        shuffle=True
                                                       )

        test_dataloader = torch.utils.data.DataLoader(testing_tensor,
                                                      batch_size=batch_size,
                                                      pin_memory=False,
                                                      num_workers=workers,
                                                        shuffle=True
                                                      )

        val_dataloader = torch.utils.data.DataLoader(validation_tensor,
                                                     batch_size=batch_size,
                                                     pin_memory=False,
                                                     num_workers=workers,
                                                       shuffle=True
                                                     )


        moduler = Trainer(net, batch_size, optimizer, self.device,
                          loc, self.yaml_file)

        (val_accuracy, epoch, epoch_counter, self.val_acc_list) = (
            moduler.train_val_fit(train_dataloader, batch_size, val_dataloader,
                                  num_epoch, self.val_acc_list))

        evaluation = moduler.evaluate(test_dataloader)

        if self.yaml_file['mode']=='hyd':
            moduler.hyd_plots(epoch_counter, evaluation)
        elif self.yaml_file['mode']=='TE':
            moduler.te_plots(epoch_counter, evaluation)

        moduler.saving(evaluation, combinations, grad)

        return val_accuracy


    def train(self, main, data_loc):
        """Runs a quick check on configurations provided in yaml file.

        Parameters
        ----------
        main : method
            Calculates the top validation accuracy of the configurations.
        data_loc : str
            The location of the data in the local device

        """
        if self.yaml_file['autoencoder']:
            test_loss = main(self.yaml_file['quick_comb_param'])
            print(f'The least test loss reached is {test_loss}')

        else:
            val_accuracy = main(self.yaml_file['quick_comb_param'])
            print("\nTop validation accuracy achieved is {}".format(val_accuracy))

            with open(os.path.join("parameters_and_graphs",
                                   data_loc,
                                   "quick_check", self.timeline,
                                   "val_acc.txt"), 'w') as w:
                w.write("Accuracy: {}\n".format(str(val_accuracy)))

    def tune(self, Optimizer, main, data_loc):
        """ Tuning optuna based Optimizer class for study

        Parameters
        ----------
        Optimizer : class
            Performs series of trials to find the best hyperparameters.
        main : method
            Calculates the top validation accuracy of the configurations.
        data_loc : str
            The location of the data in the local device

        """

        sampling = getattr(optuna.samplers, yaml_file["sampling_type"])

        optimizer = Optimizer(sampler=sampling(
            n_startup_trials=self.yaml_file["n_startup_trials"]),
            study_path=self.loc_main,
            data_loc=data_loc,
            config=self.yaml_file,
            main=main,

            timeline=self.timeline)

        print("\n{} Activated".format(self.yaml_file["sampling_type"]))
        optimizer.start_study(self.yaml_file["n_study_trials"])


if __name__ == "__main__":

    import random
    import numpy as np
    # torch.manual_seed(0)
    # random.seed(0)
    # np.random.seed(0)
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        device = "cuda:0"
        torch.cuda.empty_cache()
        print("\nRunning on gpu")

    else:
        device = "cpu"
        print("\nRunning on cpu")


    with open("config.yaml", "r") as configuration:
        yaml_file = yaml.load(configuration, Loader=yaml.FullLoader)

    start = datetime.datetime.now()

    # Assigning the location to read the data
    if yaml_file['mode']=='hyd':
        reading_all_sensors()
        data_loc = "hyd_data"

    elif yaml_file['mode']=='TE':
        data_loc = "te_data"

    timeline = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    switch = yaml_file["quick_check"]

    # To set location of save under quick check
    if switch is True:
        sampler_loc = "quick_check"

    # To set location of save under TPE sampler
    else:
        sampler_loc = yaml_file["sampling_type"]

    # Save location along with data type, sampling type and timeline.
    loc_main = os.path.join("parameters_and_graphs", data_loc,
                            sampler_loc, timeline)

    if not os.path.exists(loc_main):
        os.makedirs(loc_main)

    # Saving yaml file
    with open(os.path.join(loc_main, 'config.yaml'), "w") as target_file:
        yaml.dump(yaml_file, target_file)
        target_file.close()

    val_acc_list = []

    # getting data and labels according to the dataset specified in yaml file
    if data_loc == "hyd_data":

        data = HydDataLoader(yaml_file)
        (train_data, train_label, test_data, test_label, val_data,
         val_label) = data.data_set()

    else:
        data = TeData(root=os.getcwd(), device=device, yaml=yaml_file)
        train_data, train_label, test_data, test_label = data.get_data()
        val_data, val_label = test_data, test_label

    hydraulic_te = HydraulicTE(
        device, train_data, train_label, test_data, test_label, val_data,
        val_label, loc_main, yaml_file, val_acc_list, timeline)

    # To perform quick check
    if switch:
        hydraulic_te.train(hydraulic_te.main, data_loc)

    # To perform TPE Sampler tune
    else:
        hydraulic_te.tune(Optimizer, hydraulic_te.main, data_loc)

    end = datetime.datetime.now()
    torch.cuda.empty_cache()

    time = end - start
    print('The total time: {}'.format(time))




'''
Try changing this in the later versions
UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse
 their shape is deprecated and it will throw an error in a future release.

Consider `x.mT` to transpose batches of matricesor 
`x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor.
'''