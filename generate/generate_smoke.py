import jax
jax.config.update("jax_platform_name", "cpu")
from phi.flow import math
from phi.jax.flow import *
from tqdm import trange
import numpy as np
import h5py
import jax.numpy as jnp
import time
math.set_global_precision(64) 


nt = 100
nx = 64
ny = 64
inflow_rate = 0.3 #0.2
inflow_size = 2.05
buoyancy_force = 0.5
# nx = 32
# ny = 32
# inflow_rate = 0.3 #0.2
# inflow_size = 1.05
# buoyancy_force = 0.5
print(f"JAX devices: {jax.devices()}")

def random_inflow_loc(batch_size):
    x = tensor(np.random.uniform(inflow_size+1, nx-inflow_size-1, size=batch_size), batch('batch_size'))
    y = tensor(np.random.uniform(inflow_size+1, ny-inflow_size-1, size=batch_size), batch('batch_size'))
    return Sphere(x=x, y=y, radius=inflow_size)

@jit_compile
def step(v, s, p, dt, inflow):
    s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
    buoyancy = resample(s * (0, buoyancy_force), to=v)
    v = advect.semi_lagrangian(v, v, dt) + buoyancy * dt
    v, p = fluid.make_incompressible(v, (), Solve(x0=p))
    return v, s, p

def step_with_inflow(inflow):
    def step_fn(v, s, p, dt):
        return step(v, s, p, dt, inflow)
    return step_fn

def generate_smoke(mode, num_samples, batch_size):

    print(f'\nMode: {mode}; Number of samples: {num_samples}')

    domain = Box(x=nx, y=ny)

    batched_s_trj = []
    for b in range(0, num_samples, batch_size):
        print(f'Batch {int(b/batch_size) + 1}/{int(num_samples/batch_size)}:')
        inflow = random_inflow_loc(batch_size)
        # print(inflow._center)
        v0 = StaggeredGrid(0.0, 0.0, domain, x=nx, y=ny)
        smoke0 = CenteredGrid(0.0, ZERO_GRADIENT, domain, x=nx, y=ny)
        step_fn = step_with_inflow(inflow)
        v_trj, s_trj, p_trj = iterate(
            step_fn,
            batch(time=nt),
            v0, smoke0, None,
            dt=1.0, range=trange
        )

        batched_s_trj.append(s_trj.numpy('batch_size, time, y, x')[:,1:]) # Omit the first timestep when all is 0

    all_s_trj = np.concatenate(batched_s_trj, axis=0) 
    print('dtype: ', all_s_trj.dtype) #Should be float64

    output_path = f"data/fs_2d_pde_{nx}_{mode}_dataset.h5"
    with h5py.File(output_path, 'w') as f:
        group = f.create_group(mode)

        dataset_name = f"pde_{nt}-{nx}-{ny}"
        dset = group.create_dataset(dataset_name, data=all_s_trj, dtype='float64')

        # Add attributes
        dset.attrs['nt'] = nt
        dset.attrs['dx'] = 1.0 / nx
        dset.attrs['dt'] = 1.0 
        dset.attrs['tmin'] = 0.0 
        dset.attrs['tmax'] = 100.0


if __name__ == "__main__":
    print("\nGenerating Smoke Inflow Data...")
    batch_size = 128
    modes = {("train", 2048)} #,("valid", 128), ("test", 128)}
    # modes = {('train', 16)}

    for mode, num_samples in modes:
        t1 = time.time()
        generate_smoke(mode, num_samples, batch_size)
        print(f'Took {time.time()-t1} seconds')

    # t1 = time.time()
    # generate_smoke('train', 2, 2)
    # print(f'Took {time.time()-t1} seconds')

    print('\nData generation completed!')