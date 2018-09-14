"""
Utility functions to retrieve information about available services and setting up security for the Hops platform.
These utils facilitates development by hiding complexity for programs interacting with Hops services.
"""

import subprocess
import os
import stat
import sys
import threading
import socket

from hops import hdfs as hopshdfs
from hops import tensorboard
from hops import devices
from hops import coordination_server
from hops import mpi_service

run_id = 0


def launch(spark_session, notebook, args):
    """ Run notebook pointed to in HopsFS as a python file in mpirun
    Args:
      :spark_session: SparkSession object
      :notebook: The path in HopsFS to the notebook
      :args: Program arguments given as list
    """
    global run_id

    print('\nStarting TensorFlow job, follow your progress on TensorBoard in Jupyter UI! \n')
    sys.stdout.flush()

    sc = spark_session.sparkContext
    app_id = str(sc.applicationId)

    conf_num = int(sc._conf.get("spark.executor.instances"))
    exec_mem = sc._conf.get("spark.executor.memory")

    # Each TF task should be run on 1 executor
    nodeRDD = sc.parallelize(range(conf_num), conf_num)

    server = coordination_server.Server(conf_num)
    server_addr = server.start()
    print("Server address: ", server_addr)

    # Force execution on executor, since GPU is located on executor
    rdd_thread = threading.Thread(target=wait_on_executors, args=(nodeRDD, notebook, server_addr,))
    rdd_thread.start()

    clusterspec = server.await_reservations()

    hdfs_exec_logdir, hdfs_appid_logdir = hopshdfs.create_directories(app_id, run_id, None, 'horovod')
    tb_hdfs_path, tb_pid = tensorboard.register(hdfs_exec_logdir, hdfs_appid_logdir, 0)

    program = os.environ['PYSPARK_PYTHON']
    envs = {"HOROVOD_TIMELINE": tensorboard.logdir() + '/timeline.json',
            "TENSORBOARD_LOGDIR": tensorboard.logdir(),
            "HADOOP_VERSION": os.environ['HADOOP_VERSION'],
            "HADOOP_HOME": os.environ['HADOOP_HOME'],
            "CLASSPATH": '$(${HADOOP_HOME}/bin/hadoop classpath --glob):${HADOOP_HOME}/share/hadoop/hdfs/hadoop'
                         '-hdfs-${HADOOP_VERSION}.jar'}
    nodes = get_nodes(clusterspec, program=program, args=args)
    mpi_cmd = mpi_service.MPIRunCmd(app_id, os.environ['HADOOP_USER_NAME'], get_num_ps(clusterspec), exec_mem,
                                    envs=envs, nodes=nodes)
    mpi = mpi_service.MPIService()
    mpi.mpirun_and_wait(payload=mpi_cmd, stdout=sys.stdout, stderr=sys.stderr)

    exit_code = mpi.get_exit_code()

    server.reservations.set_finished()

    cleanup(tb_hdfs_path)
    try:
        exit_code = int(exit_code)
        if exit_code < 0:
            print("Failed to get exit code.")
        elif exit_code > 0:
            raise Exception("mpirun FAILED, look in the logs for the full error \n")
    except ValueError:
        print(exit_code)

    rdd_thread.do_run = False
    rdd_thread.join()
    print('Finished TensorFlow job \n')
    print('Make sure to check /Logs/TensorFlow/' + app_id + '/runId.' + str(run_id) + ' for full log and TensorBoard '
                                                                                      'log dir')


def get_logdir(app_id):
    global run_id
    return hopshdfs.project_path() + '/Logs/TensorFlow/' + app_id + '/horovod/run.' + str(run_id)


def wait_on_executors(node_rdd, notebook, server_addr):
    node_rdd.foreachPartition(prepare_func(notebook, server_addr))


def prepare_func(nb_path, server_addr):

    def _wrapper_fun(iter):

        for i in iter:
            executor_num = i

        client = coordination_server.Client(server_addr)

        py_runnable = localize_scripts(nb_path)
        node_meta = {'host': get_ip_address(),
                     'executor_cwd': os.getcwd(),
                     'cuda_visible_devices_ordinals': devices.get_gpu_uuid(),
                     'py_runnable': py_runnable}
        print(node_meta)

        client.register(node_meta)

        t_gpus = threading.Thread(target=devices.print_periodic_gpu_utilization)
        if devices.get_num_gpus() > 0:
            t_gpus.start()

        # Only spark executor with index 0 should create necessary HDFS directories and start mpirun
        # Other executors simply block until index 0 reports mpirun is finished

        clusterspec = client.await_reservations()
        print(executor_num, clusterspec)

        gpu_str = '\n\nChecking for GPUs in the environment\n' + devices.get_gpu_info()
        print(gpu_str)

        client.await_mpirun_finished()

        if devices.get_num_gpus() > 0:
            t_gpus.do_run = False
            t_gpus.join()

    return _wrapper_fun


def get_nodes(clusterspec, program=None, args=None):
    nodes = []
    envs = ['HOROVOD_TIMELINE', 'TENSORBOARD_LOGDIR']

    for node in clusterspec:
        runnable = program + ' ' + node['py_runnable']
        n = mpi_service.Node(node['host'], len(node['cuda_visible_devices_ordinals']), node['executor_cwd'],
                             program=runnable, args=args, envs=envs, gpus=node['cuda_visible_devices_ordinals'])
        nodes.append(n)
    return nodes


def cleanup(tb_hdfs_path):
    hopshdfs.log('Performing cleanup')
    handle = hopshdfs.get()
    if tb_hdfs_path is not None and not tb_hdfs_path == '' and handle.exists(tb_hdfs_path):
        handle.delete(tb_hdfs_path)
    hopshdfs.kill_logger()


def get_ip_address():
    """Simple utility to get host IP address"""
    return socket.gethostbyname(socket.gethostname())


def get_hosts_string(clusterspec):
    hosts_string = ''
    for host in clusterspec:
        hosts_string = hosts_string + ' ' + host['host'] + ':' + str(len(host['cuda_visible_devices_ordinals']))


def get_num_ps(clusterspec):
    num = 0
    for host in clusterspec:
        num += len(host['cuda_visible_devices_ordinals'])
    return num


def get_hosts_file(clusterspec):
    hf = ''
    host_file = os.getcwd() + '/host_file'
    for host in clusterspec:
        hf = hf + '\n' + host['host'] + ' ' + 'slots=' + str(len(host['cuda_visible_devices_ordinals']))
    with open(host_file, 'w') as hostfile: hostfile.write(hf)
    return host_file


def find_host_in_clusterspec(clusterspec, host):
    for h in clusterspec:
        if h['name'] == host:
            return h


def localize_scripts(nb_path):

    # 1. Download the notebook as a string
    fs_handle = hopshdfs.get_fs()
    fd = fs_handle.open_file(nb_path, mode='r')
    note = fd.read()
    fd.close()

    path, filename = os.path.split(nb_path)
    f_nb = open(filename, 'wb')
    f_nb.write(note)
    f_nb.flush()
    f_nb.close()

    # 2. Convert notebook to py file
    jupyter_runnable = os.path.abspath(os.path.join(os.environ['PYSPARK_PYTHON'], os.pardir)) + '/jupyter'
    conversion_cmd = jupyter_runnable + ' nbconvert --to python ' + filename
    conversion = subprocess.Popen(conversion_cmd,
                                  shell=True,
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE)
    conversion.wait()
    stdout, stderr = conversion.communicate()
    print(stdout)
    print(stderr)

    py_runnable = os.getcwd() + '/' + filename.split('.')[0] + '.py'
    assert os.path.isfile(py_runnable)

    st = os.stat(py_runnable)
    os.chmod(py_runnable, st.st_mode | stat.S_IEXEC)

    return py_runnable