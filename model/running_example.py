# WARNING: This file is made mostly for reference and understanding the inference pipeline.
#          Actually running it should take several days WITH dozens of CPU cores available...!
#          You can get all outputs of this script on Zenodo as described in the README

import argparse
from os.path import join
from warnings import warn
from sbi_simulate import main as simulate
from add_seq_errs_to_sims import main as add_seq_errs
from train_density_estimator_ensemble import main as train_ensemble
from train_big_estimator import main as train_big_estimator
from test_ensemble import main as test_estimator
from test_REJ_ABC import main as test_REJ_ABC

def simulate_and_add_seq_errs(simulations_path, ensemble_size, simulations_per_batch, 
                              seq_error_rate):
     # simulates and adds sequencing errors to the simulations
     # simulations without errors are kept for testing models with higher/lower errors
     training_simulations_no_errs_path = join(simulations_path, 'sims_before_errs')
     simulate(training_simulations_no_errs_path, ensemble_size, simulations_per_batch)
     add_seq_errs(training_simulations_no_errs_path, simulations_path, seq_error_rate)          

def simulate_train_and_test(simulations_path, train_ensemble_size, train_simulations_per_batch,
                            test_size, seq_error_rate):
     train_path = join(simulations_path, 'train')
     simulate_and_add_seq_errs(train_path, train_ensemble_size, train_simulations_per_batch, 
                              seq_error_rate)
     test_path = join(simulations_path, 'test')
     TEST_ENSEMBLE_SIZE = 1  # test set doesn't need an ensemble.
     simulate_and_add_seq_errs(test_path, TEST_ENSEMBLE_SIZE, test_size, seq_error_rate)

def train(training_simulations_path, estimators_path):
     # trains both the big estimator and the ensemble estimators
     ensemble_estimators_path = join(estimators_path, 'ensembles')
     big_estimators_path = join(estimators_path, 'big')
     for replica in ['A', 'B', 'C']:
          for summmary_statistic in ['long', 'short', 'man']:
               training_set_path = join(training_simulations_path,
                                                replica)
               trained_ensemble_path = join(ensemble_estimators_path, replica, 
                                                    summmary_statistic)
               train_ensemble(training_set_path, summmary_statistic, trained_ensemble_path)
               trained_big_path = join(big_estimators_path, replica, 
                                               summmary_statistic)
               train_big_estimator(training_set_path, summmary_statistic, trained_big_path)
     
def test(estimators_path, simulations_path, test_results_path,
         samples_per_estimator, rej_abc_acceptance_rate):
     # test ensemble density estimators, big density estimators and REJ-ABC
     test_simulations_path = join(simulations_path, 'test')
     for replica in ['A', 'B', 'C']:
          for summmary_statistic in ['long', 'short', 'man']:
               for model_type in ['big', 'ensembles']:
                    estimator_path = join(estimators_path, model_type)
                    trained_ensemble_path = join(estimator_path, replica, 
                                                 summmary_statistic)
                    test_sims_path = join(test_simulations_path, replica)
                    specific_test_res_path = join(test_results_path, model_type, 
                                                  replica, summmary_statistic)
                    test_estimator(trained_ensemble_path, test_sims_path, specific_test_res_path,
                                   samples_per_estimator)
               rej_abc_out_path = join(test_results_path, 'REJ-ABC', replica, summmary_statistic)
               test_REJ_ABC(simulations_path, summmary_statistic, rej_abc_out_path, 
                            rej_abc_acceptance_rate)

def main(output_path, train_ensemble_size=8, train_simulations_per_ensemble=10000,
         test_size=2000, seq_error_rate=0.00005, samples_per_estimator=1000, 
         rej_abc_acceptance_rate=0.01):
     warn("with default values this should take several days to run on dozens of CPU cores...")
     simulations_path = join(output_path, 'simulations')
     simulate_train_and_test(simulations_path, train_ensemble_size, train_simulations_per_ensemble, 
                             test_size, seq_error_rate)
     training_simulations = join(simulations_path, 'train')
     estimators_path = join(output_path, 'estimators')
     train(training_simulations, estimators_path)
     test_results_path = join(output_path, 'test_results')
     test(estimators_path, simulations_path, test_results_path,
          samples_per_estimator, rej_abc_acceptance_rate)
               
if __name__ == "__main__":
     parser = argparse.ArgumentParser()
     parser.add_argument("-o", "--output_path", required=True,
                         help="Path to output everything")
     parser.add_argument("-te", "--train_ensemble_size", required=True, default=8, type=int,
                         help='number of estimators in the ensemble')
     parser.add_argument("-tse", "--train_simulations_per_ensemble", required=True, 
                         default=10000, type=int,
                         help='number of simulations in each ensemble training set')
     parser.add_argument("-ts", "--test_size", required=True, 
                         default=2000, type=int,
                         help='number of simulations in test set')
     parser.add_argument("-ser", "--seq_error_rate", required=True, type=float, default=0.00005,
                         help='estimated sequencing error rate')
     parser.add_argument("-spe", "--samples_per_estimator", required=True, type=int, default=1000,
                         help='number of samples drawn from each estimator in test time\
                               for creating the posterior')
     parser.add_argument("-r", "--rej_abc_acceptance_rate", required=True, type=float, default=0.01,
                         help='REJ-ABC posterior will be created from this fraction of simulations')
                              
     args = vars(parser.parse_args())
     main(output_path=args['output_path'], train_ensemble_size=args['train_ensemble_size'], 
          train_simulations_per_ensemble=args['train_simulations_per_ensemble'], 
          test_size=args['test_size'], seq_error_rate=args['seq_error_rate'], 
          samples_per_estimator=args['samples_per_estimator'], 
          rej_abc_acceptance_rate=['rej_abc_acceptance_rate'])


