import numpy as np
import os.path as osp, time, atexit, os
class logger:

    def __init__(self, exp_name, seed, output_dir, output_fname='progress.txt'):
        self.epoch_dict = dict()
        self.first_row = True
        self.log_headers = []
        self.log_current_row = {}
        self.exp_name = exp_name
        self.output_dir = output_dir
        subdir = ''.join([exp_name, '_seed_', str(seed)])
        relpath = osp.join(output_dir, subdir)
        if osp.exists(relpath):
            print("Warning: Log dir %s already exists! Storing info there anyway." % relpath)
        else:
            os.makedirs(relpath)
        self.output_file = open(osp.join(relpath, output_fname), 'w')


    def store(self, **kwargs):
        for k, v in kwargs.items():
            if not(k in self.epoch_dict.keys()):
                self.epoch_dict[k] = []
            self.epoch_dict[k].append(v)

    def log_tabular(self, key, val=None, with_min_and_max=False, average_only=False):
        """
        Log a value or possibly the mean/std/min/max values of a diagnostic.

        Args:
            key (string): The name of the diagnostic. If you are logging a
                diagnostic whose state has previously been saved with
                ``store``, the key here has to match the key you used there.

            val: A value for the diagnostic. If you have previously saved
                values for this key via ``store``, do *not* provide a ``val``
                here.

            with_min_and_max (bool): If true, log min and max values of the
                diagnostic over the epoch.

            average_only (bool): If true, do not log the standard deviation
                of the diagnostic over the epoch.
        """
        if val is not None:
            self.log_tabular_origin(key,val)
        else:
            v = self.epoch_dict[key]
            vals = np.concatenate(v) if isinstance(v[0], np.ndarray) and len(v[0].shape)>0 else v
            stats = self.statistic_scalar(vals, with_min_and_max=with_min_and_max)
            self.log_tabular_origin(key if average_only else 'Average' + key, stats[0])
            if not(average_only):
                self.log_tabular_origin('Std'+key, stats[1])
            if with_min_and_max:
                self.log_tabular_origin('Max'+key, stats[3])
                self.log_tabular_origin('Min'+key, stats[2])
        self.epoch_dict[key] = []

    def dump_tabular(self):
        """
        Write all of the diagnostics from the current iteration.

        Writes both to stdout, and to the output file.
        """
        vals = []
        key_lens = [len(key) for key in self.log_headers]
        max_key_len = max(15,max(key_lens))
        keystr = '%'+'%d'%max_key_len
        fmt = "| " + keystr + "s | %15s |"
        n_slashes = 22 + max_key_len
        print("-"*n_slashes)

        for key in self.log_headers:
            val = self.log_current_row.get(key, "")
            valstr = "%8.3g"%val if hasattr(val, "__float__") else val
            print(fmt%(key, valstr))
            vals.append(val)

        print("-"*n_slashes)
        if self.output_file is not None:
            if self.first_row:
                self.output_file.write("\t".join(self.log_headers) + "\n")
            self.output_file.write("\t".join(map(str, vals)) + "\n")
            self.output_file.flush()

        self.log_current_row.clear()
        self.first_row=False

    def log_tabular_origin(self, key, val):
        """
        Log a value of some diagnostic.

        Call this only once for each diagnostic quantity, each iteration.
        After using ``log_tabular`` to store values for each diagnostic,
        make sure to call ``dump_tabular`` to write them out to file and
        stdout (otherwise they will not get saved anywhere).
        """
        if self.first_row:
            self.log_headers.append(key)
        else:
            assert key in self.log_headers, "Trying to introduce a new key %s that you didn't include in the first iteration"%key
        assert key not in self.log_current_row, "You already set %s this iteration. Maybe you forgot to call dump_tabular()"%key
        self.log_current_row[key] = val

    def statistic_scalar(self, x, with_min_and_max=False):
        """
        Get mean/std and optional min/max of scalar x across MPI processes.
        Args:
            x: An array containing samples of the scalar to produce statistics
                for.
            with_min_and_max (bool): If true, return min and max of x in
                addition to mean and std.
        """
        x = np.array(x, dtype=np.float32)
        global_sum, global_n = np.sum(x), len(x)
        mean = global_sum / global_n

        std = np.std(x) # compute global std
        if with_min_and_max:
            global_min = np.min(x)
            global_max = np.max(x)
            return mean, std, global_min, global_max
        return mean, std