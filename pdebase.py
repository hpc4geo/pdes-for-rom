
import numpy as np

class PDEBase:
  """
  Base class for describing discrete PDEs.
  """
  
  def __init__(self, name, parameter_names, parameter_vals, checkpoint_frequency=1000):
    """
    Input
      name:                 Textual name of the PDE. Should not contain white space characters.
      parameter_names:      List of textual names of the parameters.
      parameter_vals:       List of parameter values.
      checkpoint_frequency: How often a checkpoint file should be emitted during time-stepping.
    """
    self.model_name = name
    
    if len(parameter_names) != len(parameter_vals):
      raise RuntimeError('len(parameter_names) does not equal len(parameter_vals)')
    # Create a dict using keys from parameter_names[] and values from parameter_vals[]
    self.parameter = dict(zip(parameter_names, parameter_vals))

    self.checkpoint_frequency = checkpoint_frequency
    self.checkpoints = list() # will contain tuple : (step_counter, time)

    self.time = 0.0
    self.time_step_count = 0

    # PDE needs to define these values
    self.dt = 0.0
    self.phi = None
    self.coor = None


    def define_param_id(self):
      """
      Returns a string encoding parameter_name|values.
      """
      # Define a list of strings given by param_name|param_value.
      # Concatenate strings, placing an underscore between them.
      p = list()
      for k in self.parameter:
        p.append(k + str(self.parameter[k]))
      name = "_".join(p)
      return name
    self.param_id = define_param_id(self)


  # Standard python method for viewing an object
  def __str__(self):
    v = 'name:\n  ' + self.model_name
    v += '\n' + 'params:\n  ' + str(self.parameter)
    v += '\n' + 'param_id:\n  ' + self.param_id
    v += '\n' + 'time: ' + str(self.time)
    v += '\n' + 'time step: ' + str(self.time_step_count)
    v += '\n' + 'time step size: ' + str(self.dt)
    v += '\n' + 'checkpoint frequency: ' + str(self.checkpoint_frequency)
    v += '\n' + 'checkpoints: ' + str(self.checkpoints)
    return v


  def get_grid(self):
    """
    Returns the coordinates associated with the PDE.
    """
    return self.coor


  def get_solution(self):
    """
    Returns the solution vector associated with the PDE.
    """
    return self.phi


  # [name]-[param_id]-step[step].bin
  def create_output_name(self, step=None):
    """
    Returns a filename encoding the PDE name, parameter name/values and the current step.
    """
    if step is None:
      step = self.time_step_count
    fn = [self.model_name, self.param_id, "step" + str(step) + ".bin"]
    fn = "_".join(fn)
    return fn


  def write_solution(self):
    """
    Writes the current solution to file (currently ascii).
    """
    filename = self.create_output_name()
    v = self.get_solution()
    print('writing file:', filename)
    if v is not None:
      np.savetxt(filename, v)


  def load_solution(self, n):
    """
    Load a solution associated with step n into a vector.
    
    Input
      n: Step index.
      
    Output
      v: Solution vector associated with step `n`.
    """
    filename = self.create_output_name(step=n)
    print('loading file:', filename)
    try:
      v = np.loadtxt(filename)
    except:
      v = None
    return v
  
  
  def dump(self):
    """
    Dump the full state of PDBase to binary using pickle.
    """
    import pickle as pkl
    
    filename = self.create_output_name(step=self.time_step_count)
    filename = filename.replace('.bin','.pkl')
    file = open(filename, "wb")
    pkl.dump(self, file)
    file.close()


  def load(self, fname):
    """
    Create a PDBase object from a binary file.
    The file should be created using PDEBase.dump().
    
    Output
      pde: The object loaded from file.
    """
    import pickle as pkl
    file = open(fname, "rb")
    pde = pkl.load(file)
    file.close()
    return pde
  

  def write_checkpoint(self):
    """
    Write a solution vector to file and cache step index and time values.
    """
    self.write_solution()

    # If the checkpoint list is empty, insert value
    # If it's not empty, check the current step is larger than the last value in self.checkpoints
    # This will ensure that the in-memory record is monotonically increasing
    if len(self.checkpoints) != 0:
      last, last_time = self.checkpoints[-1]
      if self.time_step_count > last:
        self.checkpoints.append( (self.time_step_count, self.time) ) # (step_counter, time)
    else:
      self.checkpoints.append( (self.time_step_count, self.time) ) # (step_counter, time)
  
    self.dump()

  
  def step(self):
    """
    Updates the solution of the PDE. Step performs the following
    (i) Internally creates any checkpoint files;
    (ii) Increments the time step counter;
    (iii) Increments the value of time.
    By default, the initial condition (solution vector associated with time step index = 0), will be checkpointed.
    
    The method `step()` will attempt to call a user call-back `PDEBase.advance()` to perform the numerical
    operations required to update the discrete solution of the PDE.
    
    The call-back should have the signature
      advance(phi_k)
    where `phi_k` is the value of the phi from the previous time step (or the initial condition).
    
    Output
      phi: The updated solution of the PDE.
    """
    # write initial condition
    if self.time_step_count == 0:
      self.write_checkpoint()
    
    try:
      self.phi = self.advance(self.phi)
    except:
      pass
    
    self.time += self.dt
    self.time_step_count += 1
    if self.time_step_count % self.checkpoint_frequency == 0:
      self.write_checkpoint()

    return self.phi


  def load_nearest_checkpoint(self, target_step):
    """
    Loads a checkpointed file which is near to `target_step`.
    
    PDEs cannot be advanced backwards in time (in a stable manner), hence in order
    to obtain a solution at target_step we search for a checkpoint file associated 
    with a previously computed time step index which is less than or equal to `target_step`.

    This function does not modify the state of `self` (PDEBase), that is, neither the time step counter,
    the value of time or the solution vector `phi` are modified.

    If a valid checkpoint files cannot be found, None is returned for `v`, `step_i` and `step_t`.

    Output
      v: A solution vector associated with the nearest checkpoint file.
      step_i: The time step index associated with `v`.
      step_t: The time value associated with `v`.
    """
    last, last_time = self.checkpoints[-1]
    if last < target_step:
      print('Cannot load checkpoint for target step',str(target_step),'- last checkpoint written was',str(last))
      return None, None, None

    ckp_i = -1
    min_sep = 1e32
    for i in range(0, len(self.checkpoints)):
      step_i, step_t = self.checkpoints[i]
      delta = target_step - step_i
      if delta > 0:
        min_sep = min(min_sep, delta)
        ckp_i = i
      elif delta == 0:
        ckp_i = i
        break

    if ckp_i == -1:
      print('Failed to locate suitable checkpoint file for target step',str(target_step))
      return None, None, None

    step_i, step_t = self.checkpoints[ckp_i]
    v = self.load_solution(step_i)
    
    return v, step_i, step_t


  def rollback_to(self, target_step):
    """
    Moves the state of `self` (PDEBase) to a time step counter corresponding to `target_step`
    via a checkpoint file.

    Essentially this loads the nearest checkpoint file via `PDEBase.load_nearest_checkpoint()`
    and then calls `PDEBase.step()` an appropriate number of times in order that the state of
    `self` (PDEBase) is consistent with the time step index equal to `target_step`.
    """
    x, i, t = self.load_nearest_checkpoint(target_step)
    if i is None:
      msg = 'Cannot restart from target step ' + str(target_step)
      raise RuntimeError(msg)

    print('[restart] loaded nearest step', i)
          
    self.phi = x
    self.time_step_count = i
    self.time = t

    for k in range(i+1,target_step+1):
      print('[restart] updating to target - performing additional step', k)
      self.step()



def PDEBaseLoadCheckpointFile(filename):
  """
  Helper method to load a checkpoint file.
  """
  pde = PDEBase('empty', ['p'], [0])
  pde_from_file = pde.load(filename)
  return pde_from_file


def example_base_class():
  
  pde = PDEBase('base-ex1', ['kappa', 'g'], [1.0, 33.3], 4)
  pde.dt = 0.1
  
  print(pde)

  pde.write_solution()

  pde.load_solution(n=10)

  for i in range(9):
    pde.step()
  print(pde)

  print('-- load checkpoint --')
  x, i, t = pde.load_nearest_checkpoint(6)
  print(pde)

  print('-- restart from checkpoint --')
  pde.rollback_to(8)
  print(pde)

  pde.dump()

  pde2 = pde.load("base-ex1_kappa1.0_g33.3_step9.pkl")
  print(pde2)


class ShellFD(PDEBase):
  """
  A shell class which inherits from PDEBase.
  Definition[shell]: An object which is empty (hollow), and thus on its own does nothing useful.
  This shell is intended to illustrate how a real PDE discretization could be defined using PDEBase.
  
  In ShellFD, it is assumed we always will solve the PDE on the domain x \in [-1,1].
  ShellFD defines a single parameter, `a`.
  """
  
  def __init__(self, npoints, p_val_a, chp_freq = 100):
    """
    ShellFD requires two essential argument to create it:
    (1) `npoints`: the number of points used to define the finite difference grid;
    (2) `p_val_a`: the parameter value.
    An optional checkpoint frequency arg can also be provided, 
    otherwise ShellFD will use a default value of 100.
    """
    
    # Call the constructor (__init__() function) associated with PDEBase.
    # Note that even though a single parameter is defined, it is provided as a list.
    # Also note that the arg p_val_a is passed to the constructor for PDEBase.__int__()
    PDEBase.__init__(self, "ShellFD", ['a'], [p_val_a], chp_freq)

    # ===================================================
    # Declare variables unique to ShellFD implementation
    self.npoints = npoints
    self.dx = 2.0 / float(npoints-1)
    
    # ===================================================
    # Assign values to variables defined within PDEBase
    #   self.coor, self.dt, self.phi
    
    # Define the domain to be x \in [-1, 1]
    self.coor = np.zeros((self.npoints, 1))
    for i in range(self.npoints):
      self.coor[i, 0] = -1.0 + i * self.dx

    # Set the time step size
    self.dt = 1.0e-1

    # Allocate the solution vector (do not initialize it)
    self.phi = np.zeros(self.npoints)

    # (i) If only initial condition was ever to be used, we could define it directly here.
    # (ii) Alternatively, we can use the "getter" functions PDEBase.get_solution(), PDEBase.get_grid()
    #      and define the IC in the calling code. We use option (ii) for the ShellFD implementation.

  # ===========================================================
  # Define the ShellFD specific method to update the solution
  # which will be called by PDEBase.step()
  def advance(self, phi_k):
    """
    Returns phi = phi + dt * 1.2345. This is not a soluition to a PDE.
    """
    phi = phi_k + self.dt * 1.2345
    return phi


def example_shell_rollback():
  # Instantiate ShellFD with: 12 grid points, a = 0 and checkpoint frequency of 2
  pde = ShellFD(12, 0, chp_freq = 2)
  print(pde)

  # Define IC, using the coordinates
  x = pde.get_grid()
  u = pde.get_solution()
  # Note we cannot do u = 3.3 * x[:,0]**2 + 2.0 as this will allocate new memory
  # Options are
  #   u = pde.get_solution()
  #   u[:] = 3.3 * x[:,0]**2 + 2.0
  # or
  #   u = 3.3 * x[:,0]**2 + 2.0
  #   pde.phi = u
  #
  u[:] = 3.3 * x[:,0]**2 + 2.0
  
  # Print time step and u associated with the IC
  print('step',pde.time_step_count,u)

  # Perform a few time steps, printing values of u to the screen
  u1 = pde.step()
  print('step',pde.time_step_count,u1)

  u1 = pde.step()
  print('step',pde.time_step_count,u1)

  u1 = pde.step()
  print('step',pde.time_step_count,u1)

  u1 = pde.step()
  print('step',pde.time_step_count,u1)

  u1 = pde.step()
  print('step',pde.time_step_count,u1)

  # Resest the state of pde to that corresponding to step = 3.
  # This will be peformed by loading the checkpoint file from step = 2,
  # and then advance by a single call to step().
  pde.rollback_to(3)
  
  # Get the solution, print to screen, compare with value of u from step 3 printed above
  u1 = pde.get_solution()
  print('step',pde.time_step_count,u1)

  # Advance to step 4 (again), view u, compare with previous result on the screen
  u1 = pde.step()
  print('step',pde.time_step_count,u1)

  pde2 = pde.load("ShellFD_a0_step4.pkl")
  print(pde2)

def example_shell_restart_from_file():
  # Instantiate ShellFD with: 12 grid points, a = 0 and checkpoint frequency of 2
  pde = ShellFD(12, 0, chp_freq = 2)
  print(pde)
  
  # Define IC, using the coordinates
  x = pde.get_grid()
  u = pde.get_solution()
  u[:] = 3.3 * x[:,0]**2 + 2.0
  
  # Print time step and u associated with the IC
  print('step',pde.time_step_count,u)
  
  # Perform a few time steps, printing values of u to the screen
  for i in range(14):
    u1 = pde.step()
    print('step',pde.time_step_count,u1)

  print('Simulation finished')
  print(pde)


  chkpoint_filename = pde.create_output_name(step=8)
  chkpoint_filename = chkpoint_filename.replace(".bin",".pkl")

  # Option 1
  pde_restart = pde.load(chkpoint_filename)
  pde = None
  pde = pde_restart

  # Option 2
  pde = PDEBaseLoadCheckpointFile(chkpoint_filename)

  print('Simulation restarted')
  print(pde)


  for i in range(6):
    u1 = pde.step()
    print('step',pde.time_step_count,u1)

  print('Simulation finished')
  print(pde)


if __name__ == '__main__':
  example_base_class()
  #example_shell_rollback()
  example_shell_restart_from_file()


