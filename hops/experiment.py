"""
Utility functions to retrieve information about available services and setting up security for the Hops platform.

These utils facilitates development by hiding complexity for programs interacting with Hops services.
"""

from hops import hdfs as hopshdfs

from hops import differential_evolution as diff_evo
from hops import grid_search as gs
from hops import launcher as launcher

from hops import util

from datetime import datetime
import atexit
import json

elastic_id = 1
app_id = None
experiment_json = None
running = False

def launch(spark, map_fun, args_dict=None, name='no-name'):
    """ Run the wrapper function with each hyperparameter combination as specified by the dictionary

    Args:
      :spark_session: SparkSession object
      :map_fun: The TensorFlow function to run
      :args_dict: (optional) A dictionary containing hyperparameter values to insert as arguments for each TensorFlow job
    """
    try:
        global app_id
        global experiment_json
        global elastic_id
        global running
        running = True

        sc = spark.sparkContext
        app_id = str(sc.applicationId)

        launcher.run_id = launcher.run_id + 1

        experiment_json = None
        experiment_json = util.populate_experiment(sc, name, 'experiment', 'launcher', launcher.get_logdir(app_id))

        util.put_elastic(hopshdfs.project_name(), app_id, elastic_id, experiment_json)

        tensorboard_logdir = launcher.launch(sc, map_fun, args_dict, name)

        experiment_json = util.finalize_experiment(experiment_json, '', '')

        util.put_elastic(hopshdfs.project_name(), app_id, elastic_id, experiment_json)

    except:
        exception_handler()
        raise
    finally:
        elastic_id +=1
        running = False

    return tensorboard_logdir


def evolutionary_search(spark, objective_function, search_dict, direction = 'max', generations=10, popsize=10, mutation=0.5, crossover=0.7, cleanup_generations=False, name='no-name'):
    """ Run the wrapper function with each hyperparameter combination as specified by the dictionary

    Args:
      :spark_session: SparkSession object
      :map_fun: The TensorFlow function to run
      :search_dict: (optional) A dictionary containing differential evolutionary boundaries
    """
    try:
        global app_id
        global experiment_json
        global elastic_id
        global running
        running = True

        sc = spark.sparkContext
        app_id = str(sc.applicationId)

        diff_evo.run_id = diff_evo.run_id + 1

        experiment_json = None
        experiment_json = util.populate_experiment(sc, name, 'experiment', 'evolutionary_search', diff_evo.get_logdir(app_id))

        util.put_elastic(hopshdfs.project_name(), app_id, elastic_id, experiment_json)

        tensorboard_logdir, best_param, best_metric = diff_evo._search(spark, objective_function, search_dict, direction=direction, generations=generations, popsize=popsize, mutation=mutation, crossover=crossover, cleanup_generations=cleanup_generations)

        experiment_json = util.finalize_experiment(experiment_json, best_param, best_metric)

        util.put_elastic(hopshdfs.project_name(), app_id, elastic_id, experiment_json)

    except:
        exception_handler()
        raise
    finally:
        elastic_id +=1
        running = False

    return tensorboard_logdir

def grid_search(spark, map_fun, args_dict, direction='max', name='no-name'):
    """ Run the wrapper function with each hyperparameter combination as specified by the dictionary

    Args:
      :spark_session: SparkSession object
      :map_fun: The TensorFlow function to run
      :args_dict: (optional) A dictionary containing hyperparameter values to insert as arguments for each TensorFlow job
    """
    try:
        global app_id
        global experiment_json
        global elastic_id
        global running
        running = True

        sc = spark.sparkContext
        app_id = str(sc.applicationId)

        gs.run_id = gs.run_id + 1

        experiment_json = util.populate_experiment(sc, name, 'experiment', 'grid_search', gs.get_logdir(app_id))

        util.put_elastic(hopshdfs.project_name(), app_id, elastic_id, experiment_json)

        grid_params = util.grid_params(args_dict)

        tensorboard_logdir, param, metric = gs._grid_launch(sc, map_fun, grid_params, direction=direction)

        experiment_json = util.finalize_experiment(experiment_json, param, metric)

        util.put_elastic(hopshdfs.project_name(), app_id, elastic_id, experiment_json)
    except:
        exception_handler()
        raise
    finally:
        elastic_id +=1
        running = False

    return tensorboard_logdir

def exception_handler():
    global experiment_json
    if running:
        experiment_json = json.loads(experiment_json)
        experiment_json['status'] = "FAILED"
        experiment_json['finished'] = datetime.now().isoformat()
        experiment_json = json.dumps(experiment_json)
        util.put_elastic(hopshdfs.project_name(), app_id, elastic_id, experiment_json)

def exit_handler():
    global experiment_json
    if running:
        experiment_json = json.loads(experiment_json)
        experiment_json['status'] = "KILLED"
        experiment_json['finished'] = datetime.now().isoformat()
        experiment_json = json.dumps(experiment_json)
        util.put_elastic(hopshdfs.project_name(), app_id, elastic_id, experiment_json)

atexit.register(exit_handler)
