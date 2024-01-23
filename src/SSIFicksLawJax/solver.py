import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import jax.numpy as jnp

class SSISolver(diffrax.AbstractSolver):
    """
    
    See:
    https://www.sciencedirect.com/science/article/pii/0021999185901561
    
    """

    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump

        δt = t1 - t0
        prev_δt = args['δt']
        # args['δt'] = δt
        δx = args['δx']

        # "Replenished" energy term
        cv = args['heat_capacity']
        cv = jnp.append(jnp.insert(cv,0,cv[0]),cv[-1])
        fik_left  = cv[1:-1]/(cv[:-2]+cv[1:-1])
        fik_right = cv[1:-1]/(cv[2:]+cv[1:-1])
        qδt = (fik_left*args['prev_flux_left']+fik_right*args['prev_flux_right'])*prev_δt
        
        y0_ghost = jnp.append(jnp.insert(y0,0,y0[0]),y0[-1])
        D = δt*jnp.ones(y0_ghost.shape[0]-1)
        Dbar = 0.5*(D[:-1]+D[1:])

        y1 = (y0+D[:-1]*y0_ghost[:-2]/δx**2+D[1:]*y0_ghost[2:]/δx**2+qδt)/(1+2*Dbar/δx**2)

        y1_ghost = jnp.append(jnp.insert(y1,0,y1[0]),y1[-1])
        # args['prev_flux_left']  = -D[:-1]*(y0_ghost[1:-1]-y1_ghost[:-2])/δx**2+D[:-1]*(y1_ghost[1:-1]-y0_ghost[:-2])/δx**2
        # args['prev_flux_right'] = -D[1:]*(y0_ghost[2:]-y1_ghost[1:-1])/δx**2+D[1:]*(y1_ghost[2:]-y0_ghost[1:-1])/δx**2

        euler_y1 = (y0+D[:-1]*y0_ghost[:-2]/δx**2+D[1:]*y0_ghost[2:]/δx**2-2*Dbar*y0/δx**2)

        y_error = y1 - euler_y1
        dense_info = dict(y0=y0, y1=y1)

        solver_state = None
        result = diffrax.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import time

    Nx = 200
    δx = 1.0

    Nt = 240
    y0 = jnp.zeros(Nx)
    y0 = y0.at[75:125].set(1.0)
    ts = jnp.linspace(0.0,100.0,Nt)

    args = {'δx' : δx, 'heat_capacity' : jnp.ones_like(y0), 'δt' : 0.0, 'prev_flux_left' : jnp.zeros_like(y0), 'prev_flux_right' : jnp.zeros_like(y0)}

    start = time.time()
    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t,y,args : -y),
        solver=SSISolver(),
        t0=ts[0],
        t1=ts[-1],
        dt0=1e-1,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=int(1e6)
    )
    end = time.time()
    print(f'Code completed: {end-start} s')

    plt.imshow(solution.ys)
    plt.colorbar()
    plt.show()