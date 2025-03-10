import jax
# jax.config.update("jax_platform_name", "cpu")
from phi.flow import math
from phi.jax.flow import *
from tqdm import trange
import numpy as np
import h5py
import jax.numpy as jnp
import time
# math.set_global_precision(32) 


nt = 100
nx = 128
ny = 128
inflow_rate = 0.6 #0.2
inflow_size = 8
buoyancy_force = 0

# nt = 100
# nx = 32
# ny = 32
# inflow_rate = 0.3 #0.2
# inflow_size = 2
# buoyancy_force = 0.7/4

# nx = 32
# ny = 32
# inflow_rate = 0.3 #0.2
# inflow_size = 1.05
# buoyancy_force = 0.5
print(f"JAX devices: {jax.devices()}")

class InflowLocation:
    def __init__(self, nx, ny, inflow_size, all_possible_inflow_loc=True):
        self.inflow_loc_set = set()
        if all_possible_inflow_loc:
            self.populate_inflow_loc(nx, ny, inflow_size)

    def populate_inflow_loc(self, nx, ny, inflow_size):
        for i in range(inflow_size, nx-inflow_size, nx//32):
            for j in range(inflow_size, ny-inflow_size, ny//32):
                self.inflow_loc_set.add((i, j))
    
    def get_inflow_loc(self, batch_size):
        batch_x = []
        batch_y = []
        for b in range(batch_size):
            x, y = self.inflow_loc_set.pop()
            batch_x.append(x)
            batch_y.append(y)
        rand_x = tensor(batch_x, batch('batch_size'))
        rand_y = tensor(batch_y, batch('batch_size'))
        return Sphere(x=rand_x, y=rand_y, radius=inflow_size)


    def generate_random_inflow_loc(self, batch_size):
        batch_x = []
        batch_y = []
        while len(batch_x) < batch_size:
            x = np.float32(np.random.randint(inflow_size, nx-inflow_size))
            y = np.float32(np.random.randint(inflow_size, ny-inflow_size))
            if (x, y) not in self.inflow_loc_set:
                self.inflow_loc_set.add((x, y))
                batch_x.append(x)
                batch_y.append(y)
        
        rand_x = tensor(batch_x, batch('batch_size'))
        rand_y = tensor(batch_y, batch('batch_size'))
        return Sphere(x=rand_x, y=rand_y, radius=inflow_size)


@jit_compile
def step(v, s, p, dt, inflow):
    s = advect.mac_cormack(s, v, dt) + inflow_rate * resample(inflow, to=s, soft=True)
    s = diffuse.explicit(s, 0.3, dt)
    buoyancy = resample(s * (0, buoyancy_force), to=v)
    v = advect.semi_lagrangian(v, v, dt)  + buoyancy * dt
    v = diffuse.explicit(v, 0.3, dt)
    v, p = fluid.make_incompressible(v, (), Solve(x0=p))
    return v, s, p

def step_with_inflow(inflow):
    def step_fn(v, s, p, dt):
        return step(v, s, p, dt, inflow)
    return step_fn

def generate_smoke(mode, num_samples, batch_size, inflow_loc):
    print(f'\nMode: {mode}; Number of samples: {num_samples}')

    domain = Box(x=nx, y=ny)
    if inflow_loc is None:
        inflow_loc = InflowLocation(nx, ny, inflow_size)

    batched_s_trj = []
    for b in range(0, num_samples, batch_size):
        print(f'Batch {int(b/batch_size) + 1}/{int(num_samples/batch_size)}:')
        inflow = inflow_loc.get_inflow_loc(batch_size)
        # print(inflow._center)
        v0 = StaggeredGrid(np.float32(0.0), np.float32(0.0), domain, x=nx, y=ny)
        smoke0 = CenteredGrid(np.float32(0.0), ZERO_GRADIENT, domain, x=nx, y=ny)
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

    output_path = f"data/no_bouyancy_dif_fs_2d_pde_{nx}_{mode}_dataset.h5"
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

def main():
    inflow_loc = InflowLocation(nx, ny, inflow_size)
    batch_size = 4
    # modes = {("train", 528) ,("valid", 128), ("test", 128)}
    modes = {("train", 4)}

    print("\nGenerating Smoke Inflow Data...")
    for mode, num_samples in modes:
        t1 = time.time()
        generate_smoke(mode, num_samples, batch_size, inflow_loc)
        print(f'Took {time.time()-t1} seconds')
    print('\nData generation completed!')

if __name__ == "__main__":
    main()