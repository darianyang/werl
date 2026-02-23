import numpy as np
import h5py

# ps
tau=20

#f = h5py.File('west.h5', 'r')
#f = h5py.File('atb_bp_eg5_west.h5', 'r')
f = h5py.File('west_lowS.h5', 'r')

walltime = f['summary']['walltime'][:].sum()
aggtime = f['summary']['n_particles'][:].sum()

def count_events(h5):
    """
    Check if the target state was reached, given the data in a WEST H5 file.

    Parameters
    ----------
    h5 : h5py.File
        west.h5 file
    """
    events = 0
    # Get the key to the final iteration. 
    # Need to do -2 instead of -1 because there's an empty-ish final iteration written.
    for iteration_key in list(h5['iterations'].keys())[-2:0:-1]:
        endpoint_types = h5[f'iterations/{iteration_key}/seg_index']['endpoint_type']
        if 3 in endpoint_types:
            #print(f"recycled segment found in file {h5_filename} at iteration {iteration_key}")
            # count the number of 3s
            events += np.count_nonzero(endpoint_types == 3)
    return events

print("walltime: ", walltime, "seconds")
print("walltime: ", walltime/60, "minutes")
print("walltime: ", walltime/60/60, "hours")
print("walltime: ", walltime/60/60/24, "days")
print("aggtime: ", aggtime, "units")
print("aggtime: ", (aggtime * tau)/1000, "ns")
print("events:", count_events(f))
