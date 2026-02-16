# calc_aggr_time.py
# 
# Calculated aggregate simulation time from a WESTPA simulation

import h5py


def calc_aggr_sim_time(h5file='west.h5', tau=100, units='ps'):
    count = 0
    with h5py.File(h5file,'r') as f:
        for idx in range(len(f['summary'][:])):
            if f['summary'][idx,'walltime']>0:
                count += f['summary']['n_particles'][idx]

    print(f'Aggregate: {count} segments; Total of {count*tau} {units} at {tau} {units}/seg.')

def calc_clock_time(h5file='west.h5'):
    cpu_time = 0
    wall_time = 0
    with h5py.File(h5file,'r') as f:
        for idx in range(len(f['summary'][:])):
            wall_time += f['summary']['walltime'][idx]
            cpu_time += f['summary']['cputime'][idx]

    print(f'Wall Clock: {wall_time:.3f} secs / {wall_time/60:.3f} mins / {wall_time/60/60:.3f} hours\n\
CPU Clock: {cpu_time:.3f} secs / {cpu_time/60:.3f} mins / {cpu_time/60/60:.3f} hours')



if __name__ == "__main__":
    calc_aggr_sim_time('west.h5', tau=50)
    calc_clock_time('west.h5')
