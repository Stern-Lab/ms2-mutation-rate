# WARNING: This file is made mostly for reference and understanding the pipeline.
#          Actually running it should take several days WITH dozens of CPU cores available...
#          You can get all outputs of this script on Zenodo as described in the README

import os
from sbi_simulate import main as simulate
from add_seq_errs_to_sims import main as add_seq_errs
from train_density_estimator_ensemble import main as train_ensemble
from train_big_estimator import main as train_big_estimator
from test_ensemble import main as test_estimator

#TODO: 
#    - both train and test in simulate_and_add_seq_errs?
#    - if so, add rej-abc to test

def simulate_and_add_seq_errs(output_dir, ensemble_size=8, simulations_per_batch=10000, 
                              seq_error_rate=0.00005):
     # This function simulates and adds sequencing errors to the simulations ending up with 2 folders,
     # one with simulations without sequencing errors and one with sequencing errors.
     training_simulations_no_errs_path = os.path.join(output_dir, 'training_sims_no_errs')
     simulate(training_simulations_no_errs_path, ensemble_size, simulations_per_batch)
     training_simulations_with_errs_path = os.path.join(output_dir, 'training_sims_with_errs')
     add_seq_errs(training_simulations_no_errs_path, training_simulations_with_errs_path,
                  seq_error_rate)     
     
def train(training_simulations_path, estimators_path):
     # This function trains both the big estimator and the ensemble estimators
     ensemble_estimators_path = os.path.join(estimators_path, 'ensembles')
     big_estimators_path = os.path.join(estimators_path, 'big')
     for replica in ['A', 'B', 'C']:
          for summmary_statistic in ['long', 'short', 'man']:
               training_set_path = os.path.join(training_simulations_path,
                                                replica)
               trained_ensemble_path = os.path.join(ensemble_estimators_path, replica, 
                                                    summmary_statistic)
               train_ensemble(training_set_path, summmary_statistic, trained_ensemble_path)
               trained_big_path = os.path.join(big_estimators_path, replica, 
                                               summmary_statistic)
               train_big_estimator(training_set_path, summmary_statistic, trained_big_path)
     
def test(estimators_path, test_simulations_path, test_results_path,
         samples_per_estimator=1000):
     # test ensemble density estimators, big density estimators
     for replica in ['A', 'B', 'C']:
          for summmary_statistic in ['long', 'short', 'man']:
               for model_type in ['big', 'ensembles']:
                    estimator_path = os.path.join(estimators_path, model_type)
                    trained_ensemble_path = os.path.join(estimator_path, replica, 
                                             summmary_statistic)
                    test_sims_path = os.path.join(test_simulations_path, replica)
                    specific_test_res_path = os.path.join(test_results_path, model_type, 
                                                          replica, summmary_statistic)
                    test_estimator(trained_ensemble_path, test_sims_path, specific_test_res_path,
                                   samples_per_estimator)
               
               


