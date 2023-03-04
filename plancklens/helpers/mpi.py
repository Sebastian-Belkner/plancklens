"""mpi4py wrapper module.

"""

from __future__ import print_function
import os

verbose = False


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False
    
has_key = lambda key : key in os.environ.keys()
cond4mpi4py = not has_key('NERSC_HOST') or (has_key('NERSC_HOST') and has_key('SLURM_SUBMIT_DIR'))

if not is_notebook() and cond4mpi4py:
    try:
        from mpi4py import MPI

        rank = MPI.COMM_WORLD.Get_rank()
        size = MPI.COMM_WORLD.Get_size()
        barrier = MPI.COMM_WORLD.Barrier
        ANY_SOURCE = MPI.ANY_SOURCE
        send = MPI.COMM_WORLD.send
        receive = MPI.COMM_WORLD.recv
        bcast = MPI.COMM_WORLD.bcast
        status = MPI.Status
        iprobe = MPI.COMM_WORLD.Iprobe
        Ibcast = MPI.COMM_WORLD.Ibcast
        Isend = MPI.COMM_WORLD.Isend
        Ireceive = MPI.COMM_WORLD.Irecv
        finalize = MPI.Finalize
        if verbose: print('mpi.py : setup OK, rank %s in %s' % (rank, size))
    except:
        rank = 0
        size = 1
        barrier = lambda: -1
        finalize = lambda: -1
        bcast = lambda _: 0
        if verbose: print('mpi.py: unable to import mpi4py\n')
else:
    rank = 0
    size = 1
    barrier = lambda: -1
    bcast = lambda _: 0
    finalize = lambda: -1