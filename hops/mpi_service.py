"""
MPI service helper classes that can be used to spawn mpi processes and retrieve log and status.
"""

import requests
import json
import time
import os
try:
    import thread as thread
except ModuleNotFoundError:
    import threading as thread
try:
    from urlparse import urlparse
except ModuleNotFoundError:
    from urllib.parse import urlparse

STOP_STATES = ['Stopped', 'Unknown']
POLLING_DELAY = 2
NO_LOG = 'No log found for appid.'
os.environ['MPI_REST_ENDPOINT'] = 'http://bbc4.sics.se:62668/mpi-service'
if 'MPI_REST_ENDPOINT' not in os.environ:
    raise EnvironmentError('Environment variable MPI_REST_ENDPOINT not set.')
MPI_REST_ENDPOINT = os.environ['MPI_REST_ENDPOINT']

try:
    url_ = urlparse(MPI_REST_ENDPOINT)
except:
    raise EnvironmentError('Malformed url: ' + MPI_REST_ENDPOINT)

if not url_.netloc:
    raise EnvironmentError('Malformed url: ' + MPI_REST_ENDPOINT)

if 'SSL_ENABLED' not in os.environ:
    os.environ['SSL_ENABLED'] = 'false'

if os.environ['SSL_ENABLED'] == 'true' and url_.scheme != 'https':
    MPI_REST_ENDPOINT = 'https://' + url_.netloc + url_.path
elif os.environ['SSL_ENABLED'] == 'false' and url_.scheme != 'http':
    MPI_REST_ENDPOINT = 'http://' + url_.netloc + url_.path


class MPIService:

    def __init__(self, session_id=None, token=None, appid=None, pid=None, num_stdout_lines_to_save=50):
        self._base_url = MPI_REST_ENDPOINT.strip('/')
        self.cookies = {}
        self.headers = {}
        self.appid = appid
        self.pid = pid
        self._num_stdout_lines_to_save = num_stdout_lines_to_save
        self._stdout = ''
        if session_id is not None:
            self.cookies = {'SESSIONID': session_id}
        elif token is not None:
            self.headers = {'Authorization': token}

    def get_session(self):
        s = requests.Session()
        return s

    def mpirun(self, payload=None):
        if payload is None:
            raise ValueError('Can not start mpi with empty payload.')
        elif not isinstance(payload, MPIRunCmd):
            raise ValueError('Payload needs to be an instance of MPIRunCmd.')
        self.appid = payload.appid
        with requests.Session() as session:
            self.pid = self.mpirun_(self._base_url, session, payload.toJSON(), headers=self.headers,
                                    cookies=self.cookies)
        return self.pid

    @staticmethod
    def mpirun_(base_url, session, payload, headers={}, cookies={}):
        session.headers.update({'Content-type': 'application/json'})
        url = base_url + '/jobs'
        response = session.post(url, headers=headers, cookies=cookies, data=payload)
        return MPIService.handel_response(response)

    def get_log(self, log_type='stdout', offset=0, length=-1):
        with requests.Session() as session:
            log = self.get_log_(self._base_url, session, self.appid, self.pid, log_type, offset, length,
                                headers=self.headers, cookies=self.cookies)
        return log

    @staticmethod
    def get_log_(base_url, session, appid, pid, log_type='stdout', offset=0, length=-1, headers={}, cookies={}):
        url = base_url + '/jobs/' + appid + '/' + str(pid) + '/log?' + 'type=' + log_type + '&offset=' + str(offset) + \
              '&length=' + str(length)
        response = session.get(url, headers=headers, cookies=cookies)
        return MPIService.handel_response(response)

    def get_status(self):
        with requests.Session() as session:
            status = self.get_status_(self._base_url, session, self.appid, self.pid, headers=self.headers,
                                      cookies=self.cookies)
        return status

    @staticmethod
    def get_status_(base_url, session, appid, pid, headers={}, cookies={}):
        url = base_url + '/jobs/' + appid + '/' + str(pid) + '/status'
        response = session.get(url, headers=headers, cookies=cookies)
        return MPIService.handel_response(response)

    def get_exit_code(self):
        with requests.Session() as session:
            status = self.get_exit_code_(self._base_url, session, self.appid, self.pid, headers=self.headers, cookies=self.cookies)
        return status

    @staticmethod
    def get_exit_code_(base_url, session, appid, pid, headers={}, cookies={}):
        url = base_url + '/jobs/' + appid + '/' + str(pid) + '/exit-code'
        response = session.get(url, headers=headers, cookies=cookies)
        return MPIService.handel_response(response)

    def mpi_top(self):
        with requests.Session() as session:
            top = self.mpi_top_(self._base_url, session, self.appid, self.pid, headers=self.headers,
                                cookies=self.cookies)
        return top

    @staticmethod
    def mpi_top_(base_url, session, appid, pid, headers={}, cookies={}):
        url = base_url + '/jobs/' + appid + '/' + str(pid) + '/mpi-top'
        response = session.get(url, headers=headers, cookies=cookies)
        return MPIService.handel_response(response)

    def is_done(self):
        return self.get_status() in STOP_STATES

    def wait(self):
        done = self.is_done()
        while not done:
            done = self.is_done()
            time.sleep(POLLING_DELAY)

    def redirect_log(self, log_type, stream):
        if stream is None or not hasattr(stream, 'write'):
            raise ValueError('Stream needs to be writable.')
        if not log_type:
            log_type = 'stdout'
        done = self.is_done()
        offset_ = 0
        while not done:
            done = self.is_done()
            out_ = self.get_log(log_type=log_type, offset=offset_)
            if out_ != NO_LOG:
                offset_ = offset_ + len(out_)
                if len(out_) > 0:
                    stream.write(out_)
                    self._get_log_tail(out_)
            time.sleep(POLLING_DELAY)

    def mpirun_and_wait(self, payload={}, stdout=None, stderr=None):
        self.mpirun(payload=payload)
        try:
            if stdout is not None and hasattr(stdout, 'write'):
                thread.start_new_thread(self.redirect_log, ('stdout', stdout,))
            if stderr is not None and hasattr(stdout, 'write'):
                thread.start_new_thread(self.redirect_log, ('stderr', stderr,))
        except:
            print("Error: unable to start thread")
        self.wait()

    @staticmethod
    def handel_response(response):
        response.raise_for_status()
        content = response.content
        return content.decode("utf-8")

    def _get_log_tail(self, lines):
        lines = self._stdout + lines
        lines = lines.split('\n')
        lines = "\n".join(lines[-self._num_stdout_lines_to_save:])
        self._stdout = lines

    def get_saved_log(self):
        return self._stdout


class Node:
    def __init__(self, hostname, num_processes, wdir, program=None, args=None, envs=None, gpus=[]):
        self.hostname = hostname
        self.numProcesses = num_processes
        self.wdir = wdir
        self.envs = envs
        self.gpus = gpus
        self.program = program
        self.args = args

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__)


class MPIRunCmd:
    def __init__(self, appid, username, executor_cores, executor_memory, program=None, args=None, envs=None, nodes=[]):
        self.appid = appid
        self.username = username
        self.executorCores = executor_cores
        self.executorMemory = executor_memory
        self.envs = envs
        self.program = program
        self.args = args
        self.nodes = nodes

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__)
