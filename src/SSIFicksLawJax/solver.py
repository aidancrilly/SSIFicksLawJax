import diffrax
import equinox as eqx  # https://github.com/patrick-kidger/equinox
import jax
import lineax as lx
import jax.numpy as jnp

class FTCSSolver(diffrax.AbstractSolver):
    """
    
    Forward time centred space

    y^{n+1}_{i} = y^{n}_{i}+r*(y^{n}_{i+1}-2*y^{n}_{i}+y^{n}_{i-1})

    """

    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump

        δt = t1 - t0
        δxc, areas, vols = args['δxc'],args['areas'],args['vols']
        
        y0_ghost = jnp.append(jnp.insert(y0,0,y0[0]),y0[-1])
        D = jnp.ones(y0_ghost.shape[0]-1)*(δt*areas)
        Dbar = 0.5*(D[:-1]+D[1:])
        y1 = (y0+D[:-1]*y0_ghost[:-2]/vols/δxc+D[1:]*y0_ghost[2:]/vols/δxc-2*Dbar*y0/vols/δxc)

        y_error = None
        dense_info = dict(y0=y0, y1=y1)
        solver_state = None
        result = diffrax.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        pass

class BackwardEuler(diffrax.AbstractSolver):
    """
    
    Backward time centred space

    y^{n+1}_{i} = y^{n}_{i}+r*(y^{n+1}_{i+1}-2*y^{n+1}_{i}+y^{n+1}_{i-1})

    Tridiagonal system

    """

    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump

        δt = t1 - t0
        δxc, areas, vols = args['δxc'],args['areas'],args['vols']
        
        y0_ghost = jnp.append(jnp.insert(y0,0,y0[0]),y0[-1])
        D = jnp.ones(y0_ghost.shape[0]-1)*(δt*areas)
        Dbar = 0.5*(D[:-1]+D[1:])

        diag = 2*Dbar/vols/δxc
        diag = (diag.at[0].set((2*Dbar[0]-D[0])/vols[0]/δxc)).at[-1].set((2*Dbar[-1]-D[-1])/vols[-1]/δxc)
        operator = lx.TridiagonalLinearOperator(1+diag,-D[1:-1]/vols[1:]/δxc,-D[1:-1]/vols[:-1]/δxc)

        y1 = lx.linear_solve(operator, y0).value

        y_error = None
        dense_info = dict(y0=y0, y1=y1)
        solver_state = None
        result = diffrax.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        pass

class CrankNicolson(diffrax.AbstractSolver):
    """
    
    CrankNicolson

    y^{n+1}_{i} = y^{n}_{i}+0.5*r*(y^{n+1}_{i+1}-2*y^{n+1}_{i}+y^{n+1}_{i-1})+0.5*r*(y^{n}_{i+1}-2*y^{n}_{i}+y^{n}_{i-1})

    Tridiagonal system

    """

    term_structure = diffrax.ODETerm
    interpolation_cls = diffrax.LocalLinearInterpolation

    def init(self, terms, t0, t1, y0, args):
        return None

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump

        δt = t1 - t0
        δxc, areas, vols = args['δxc'],args['areas'],args['vols']
        
        y0_ghost = jnp.append(jnp.insert(y0,0,y0[0]),y0[-1])
        D = jnp.ones(y0_ghost.shape[0]-1)*(δt*areas)
        Dbar = 0.5*(D[:-1]+D[1:])

        diag = 2*Dbar/vols/δxc
        diag = (diag.at[0].set((2*Dbar[0]-D[0])/vols[0]/δxc)).at[-1].set((2*Dbar[-1]-D[-1])/vols[-1]/δxc)
        operator = lx.TridiagonalLinearOperator(1+0.5*diag,-0.5*D[1:-1]/vols[1:]/δxc,-0.5*D[1:-1]/vols[:-1]/δxc)

        vector = y0+0.5*(D[:-1]*y0_ghost[:-2]/vols/δxc+D[1:]*y0_ghost[2:]/vols/δxc-2*Dbar*y0/vols/δxc)

        y1 = lx.linear_solve(operator, vector).value

        y_error = None
        dense_info = dict(y0=y0, y1=y1)
        solver_state = None
        result = diffrax.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        pass


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
        del made_jump

        δt = t1 - t0
        δxc, areas, vols = args['δxc'],args['areas'],args['vols']

        # "Replenished" energy term
        cv = args['heat_capacity']
        cv = jnp.append(jnp.insert(cv,0,cv[0]),cv[-1])
        fik_left  = cv[1:-1]/(cv[:-2]+cv[1:-1])
        fik_right = cv[1:-1]/(cv[2:]+cv[1:-1])
        qδt = (fik_left*solver_state['δ_left']+fik_right*solver_state['δ_right'])
        
        y0_ghost = jnp.append(jnp.insert(y0,0,y0[0]),y0[-1])
        D = jnp.ones(y0_ghost.shape[0]-1)*(δt*areas)
        Dbar = 0.5*(D[:-1]+D[1:])

        y1 = (y0+D[:-1]*y0_ghost[:-2]/vols/δxc+D[1:]*y0_ghost[2:]/vols/δxc+qδt)/(1+2*Dbar/vols/δxc)

        y1_ghost = jnp.append(jnp.insert(y1,0,y1[0]),y1[-1])
        
        solver_state['δ_right'] = D[1:]* (-(y0_ghost[2:]-y1_ghost[1:-1])+(y1_ghost[2:]-y0_ghost[1:-1])  )/vols/δxc
        solver_state['δ_left']  = D[:-1]*(-(y0_ghost[1:-1]-y1_ghost[:-2])+(y1_ghost[1:-1]-y0_ghost[:-2]))/vols/δxc

        euler_y1 = (y0+D[:-1]*y0_ghost[:-2]/vols/δxc+D[1:]*y0_ghost[2:]/vols/δxc-2*Dbar*y0/vols/δxc)

        y_error = y1 - euler_y1
        dense_info = dict(y0=y0, y1=y1)

        result = diffrax.RESULTS.successful
        return y1, y_error, dense_info, solver_state, result
    
    def func(self, terms, t0, y0, args):
        pass

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    """
    
    Simple 1D test problem with central blob of heat diffusing outwards with uniform diffusivity/heat capacity.
    
    """

    Nx = 200
    δx = 1.0
    A  = 1.0

    y0 = jnp.zeros(Nx)
    y0 = y0.at[75:125].set(1.0)

    Nt = 240
    ts = jnp.linspace(0.0,150.0,Nt)

    safety_factor = 0.9
    dt_explicit = safety_factor*0.5*1.0/δx**2

    args = {'δxc' : δx, 'areas' : A*jnp.ones(Nx+1), 'vols' : A*δx*jnp.ones(Nx), 'heat_capacity' : jnp.ones_like(y0)}
    solver_state = {'δ_left' : jnp.zeros_like(y0), 'δ_right' : jnp.zeros_like(y0)}

    SSI_solution1 = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t,y,args : -y),
        solver=SSISolver(),
        t0=ts[0],
        t1=ts[-1],
        dt0=10*dt_explicit,
        y0=y0,
        args=args,
        solver_state=solver_state,
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=int(1e6)
    )

    SSI_solution2 = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t,y,args : -y),
        solver=SSISolver(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt_explicit,
        y0=y0,
        args=args,
        solver_state=solver_state,
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=int(1e6)
    )

    FTCS_solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t,y,args : -y),
        solver=FTCSSolver(),
        t0=ts[0],
        t1=ts[-1],
        dt0=dt_explicit,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=int(1e6)
    )

    Imp_solution = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t,y,args : -y),
        solver=BackwardEuler(),
        t0=ts[0],
        t1=ts[-1],
        dt0=10*dt_explicit,
        y0=y0,
        args=args,
        stepsize_controller=diffrax.ConstantStepSize(),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps=int(1e6)
    )


    fig = plt.figure(figsize=(12,9))
    ax1 = fig.add_subplot(241)
    ax2 = fig.add_subplot(242)
    ax3 = fig.add_subplot(243)
    ax4 = fig.add_subplot(244)
    ax5 = fig.add_subplot(245)
    ax6 = fig.add_subplot(246)

    ax1.imshow(SSI_solution1.ys)
    ax2.imshow(SSI_solution2.ys)
    ax3.imshow(FTCS_solution.ys)
    ax4.imshow(Imp_solution.ys)
    ax1.set_title("SSI dt  = 10 x dt_exp")
    ax2.set_title("SSI dt  = dt_exp")
    ax3.set_title("FTCS dt = dt_exp")
    ax4.set_title("Implicit dt = 10 x dt_exp")

    ax5.plot(SSI_solution1.ts,jnp.sum(SSI_solution1.ys,axis=1))
    ax5.plot(SSI_solution2.ts,jnp.sum(SSI_solution2.ys,axis=1))
    ax5.plot(FTCS_solution.ts,jnp.sum(FTCS_solution.ys,axis=1))
    ax5.plot(Imp_solution.ts,jnp.sum(Imp_solution.ys,axis=1))
    ax5.set_xlabel("t")
    ax5.set_ylabel(r"$\int$ y(x,t) $dx$")

    ax6.plot(SSI_solution1.ys[-1,:],alpha=0.5,lw=3,label="SSI dt  = 10 x dt_exp")
    ax6.plot(SSI_solution2.ys[-1,:],alpha=0.5,lw=3,label="SSI dt  = dt_exp")
    ax6.plot(FTCS_solution.ys[-1,:],alpha=0.5,lw=3,label="FTCS dt = dt_exp")
    ax6.plot(Imp_solution.ys[-1,:],alpha=0.5,lw=3,label="Implicit dt  = 10 x dt_exp")
    ax6.set_xlabel("x")
    ax6.set_ylabel("y(t=t1)")

    ax6.legend(frameon=False)

    fig.tight_layout()

    plt.show()