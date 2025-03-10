import taichi as ti
import argparse
import os
import math
import numpy as np
import random
import matplotlib.pyplot as plt
import noise


real = ti.f32
ti.init(default_fp=real, arch=ti.gpu, flatten_if=True)

# Simulation parameters
dim = 2
n_particles = 8192
n_solid_particles = 0
n_actuators = 0
n_grid = 128
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = 1
E = 5
mu = E
la = E
max_steps = 2048
steps = 1024
gravity = 3.8
target = [0.8, 0.2]
bound = 3
coeff = 0.01
damping = 0.95

# Fields
scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

actuator_id = ti.field(ti.i32)
particle_type = ti.field(ti.i32)
x, v = vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
C, F = mat(), mat()

loss = scalar()

# Actuation parameters
n_sin_waves = 4
weights = scalar()
bias = scalar()
x_avg = vec()

actuation = scalar()
actuation_omega = 30  # Frequency of actuation
act_strength = 20  # Strength of actuation

# Metrics
shape_overlap = scalar()
deformation = scalar()
stability = scalar()
speed = scalar()
shape_penalty_value = scalar()
cluster_penalty = scalar()
target_x, target_y = scalar(), scalar()

# Shape parameters
shape_params = ti.Vector.field(4, dtype=real)  # [n_actuators, 4]
shape_params_grad = ti.Vector.field(4, dtype=real)

def allocate_fields(n_actuators):
    ti.root.dense(ti.ij, (n_actuators, n_sin_waves)).place(weights)
    ti.root.dense(ti.i, n_actuators).place(bias)

    ti.root.dense(ti.ij, (max_steps, n_actuators)).place(actuation)
    ti.root.dense(ti.i, n_particles).place(actuator_id, particle_type)
    ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, C, F)
    ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in, grid_v_out)
    ti.root.place(loss, x_avg)
    ti.root.place(shape_overlap, deformation, stability, speed)
    ti.root.place(shape_penalty_value, cluster_penalty, target_x, target_y)
    # Allocate fields for shape parameters and their gradients
    ti.root.dense(ti.i, n_actuators).place(shape_params, shape_params_grad)

    ti.root.lazy_grad()

@ti.kernel
def initialize_x():
    for i in range(n_particles):
        for f in range(max_steps):
            x[f, i] = x[0, i]  # Copy initial positions to all time steps

@ti.kernel
def initialize_actuator_id():
    for i in range(n_particles):
        actuator_id[i] = -1  # Default value for fluid particles

@ti.kernel
def initialize_F():
    for i in range(n_particles):
        F[0, i] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]])

@ti.kernel
def clear_grid():
    for i, j in grid_m_in:
        grid_v_in[i, j] = [0, 0]
        grid_m_in[i, j] = 0
        grid_v_in.grad[i, j] = [0, 0]
        grid_m_in.grad[i, j] = 0
        grid_v_out.grad[i, j] = [0, 0]

@ti.kernel
def clear_particle_grad():
    for f, i in x:
        x.grad[f, i] = [0, 0]
        v.grad[f, i] = [0, 0]
        C.grad[f, i] = [[0, 0], [0, 0]]
        F.grad[f, i] = [[0, 0], [0, 0]]

@ti.kernel
def clear_actuation_grad():
    for t, i in actuation:
        actuation[t, i] = 0.0

@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        J = (new_F).determinant()
        if particle_type[p] == 0:  # fluid
            sqrtJ = ti.sqrt(J)
            new_F = ti.Matrix([[sqrtJ, 0], [0, sqrtJ]])

        F[f + 1, p] = new_F
        r, s = ti.polar_decompose(new_F)
        act_id = actuator_id[p]
        act = actuation[f, ti.max(0, act_id)] * act_strength
        if act_id == -1:
            act = 0.0
        A = ti.Matrix([[0.0, 0.0], [0.0, 1.0]]) * act
        cauchy = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        mass = 0.0
        if particle_type[p] == 0:
            mass = 4
            cauchy = ti.Matrix([[1.0, 0.0], [0.0, 0.1]]) * (J - 1) * E
        else:
            mass = 1
            cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                     ti.Matrix.diag(2, la * (J - 1) * J)
        cauchy += new_F @ A @ new_F.transpose()
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + mass * C[f, p]
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[base + offset] += weight * (mass * v[f, p] + affine @ dpos)
                grid_m_in[base + offset] += weight * mass

@ti.kernel
def grid_op():
    for i, j in grid_m_in:
        inv_m = 1 / (grid_m_in[i, j] + 1e-10)
        v_out = inv_m * grid_v_in[i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
            v_out[1] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
            v_out[1] = 0
        if j < bound and v_out[1] < 0:
            v_out[0] = 0
            v_out[1] = 0
            normal = ti.Vector([0.0, 1.0])
            lsq = (normal**2).sum()
            if lsq > 0.5:
                if ti.static(coeff < 0):
                    v_out[0] = 0
                    v_out[1] = 0
                else:
                    lin = v_out.dot(normal)
                    if lin < 0:
                        vit = v_out - lin * normal
                        lit = vit.norm() + 1e-10
                        if lit + coeff * lin <= 0:
                            v_out[0] = 0
                            v_out[1] = 0
                        else:
                            v_out = (1 + coeff * lin / lit) * vit
        if j > n_grid - bound and v_out[1] > 0:
            v_out[0] = 0
            v_out[1] = 0

        grid_v_out[i, j] = v_out

@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C

@ti.kernel
def compute_actuation(t: ti.i32):
    for i in range(n_actuators):
        act = 0.0
        for j in ti.static(range(n_sin_waves)):
            amplitude = 1.0
            wave_len = 0.5 
            phase_diff = actuation_omega * t * dt - wave_len * i
            act += amplitude * weights[i, j] * ti.sin(phase_diff)
        act += bias[i]
        actuation[t, i] = ti.tanh(act) * 1.5 

@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        contrib = 0.0
        if particle_type[i] == 1:
            contrib = 1.0 / n_solid_particles
        ti.atomic_add(x_avg[None], contrib * x[steps - 1, i])

@ti.kernel
def reset_metrics():
    shape_overlap[None] = 0.0
    deformation[None] = 0.0
    stability[None] = 0.0
    shape_penalty_value[None] = 0.0
    x_avg[None] = [0, 0]
    cluster_penalty[None] = 0.0
    target_x[None], target_y[None] = 0.5, 0.5

# Rest of the code remains unchanged...     
@ti.kernel
def compute_shape_penalty():

    # Penalize shapes that are too large or too small
    for i in range(n_actuators):
        width = shape_params[i][2]  # Width of the shape
        height = shape_params[i][3]  # Height of the shape

        # Penalize if width or height is outside the desired range [0.1, 0.5]
        if width < 0.1 or width > 0.5:
            shape_penalty_value[None] += (0.1 - width) ** 2 if width < 0.1 else (width - 0.5) ** 2
        if height < 0.1 or height > 0.5:
            shape_penalty_value[None] += (0.1 - height) ** 2 if height < 0.1 else (height - 0.5) ** 2

    # Penalize breaking (deformation and separation)
    for i in range(n_particles):
        if particle_type[i] == 1: 
            initial_pos = x[0, i]
            current_pos = x[steps - 1, i]
            deformation = (current_pos - initial_pos).norm()

            shape_penalty_value[None] += deformation * 0.1  # Scale the penalty

    # Penalize disconnected clusters (breaking into multiple parts)
    for i in range(n_particles):
        if particle_type[i] == 1: 
            for j in range(i + 1, n_particles):
                if particle_type[j] == 1:
                    distance = (x[steps - 1, i] - x[steps - 1, j]).norm()
                    if distance > 0.1: 
                        cluster_penalty[None] += distance * 0.1 

    # Penalize shapes that are too far from the target area
    for i in range(n_actuators):
        x, y, w, h = shape_params[i]
        shape_center_x = x + w / 2
        shape_center_y = y + h / 2
        distance = ((shape_center_x - target_x[None]) ** 2 + (shape_center_y - target_y[None]) ** 2) ** 0.5
        shape_penalty_value[None] += distance * 0.1  


@ti.kernel
def compute_shape_metrics():
    for i in range(n_particles):
        if particle_type[i] == 1:  
            orig_pos = x[0, i]
            final_pos = x[steps - 1, i]
            deform = ti.sqrt((final_pos - orig_pos).dot(final_pos - orig_pos))
            ti.atomic_add(deformation[None], deform)
            
            avg_y = 0.0
            for t in range(steps - 10, steps):
                avg_y += x[t, i][1]
            avg_y /= 10.0
            
            variation = 0.0
            for t in range(steps - 10, steps):
                diff = x[t, i][1] - avg_y
                variation += diff * diff
            ti.atomic_add(stability[None], ti.sqrt(variation))

@ti.kernel
def compute_loss():
    w_distance = 5.0 
    w_speed = 1.0
    w_stability = 0.1 
    w_deform = 0.2
    w_shape_penalty = 0.005

    distance = x_avg[None][0] 
    speed = distance / (steps * dt) 
    norm_stability = stability[None] / (steps * n_actuators)
    norm_deform = deformation[None] / n_actuators + cluster_penalty[None] * 0.1
    shape_penalty = shape_penalty_value[None]
    previous_loss = loss[None]

    momentum = 0.9 * previous_loss if previous_loss != 0 else 0  

    # Loss function: maximize distance and speed, minimize instability and deformation
    loss[None] = -w_distance * distance - w_speed * speed + \
                 w_stability * norm_stability + \
                 w_deform * norm_deform + \
                 w_shape_penalty * shape_penalty + momentum

    

@ti.ad.grad_replaced
def advance(s):
    clear_grid()
    compute_actuation(s)
    p2g(s)
    grid_op()
    g2p(s)

@ti.ad.grad_for(advance)
def advance_grad(s):
    clear_grid()
    p2g(s)
    grid_op()

    g2p.grad(s)
    grid_op.grad()
    p2g.grad(s)
    compute_actuation.grad(s)

def forward(total_steps=steps):
    for s in range(total_steps - 1):
        advance(s)
        
    reset_metrics()
    compute_x_avg()
    compute_shape_penalty()
    compute_shape_metrics()
    compute_loss()

class Scene:
    def __init__(self):
        self.n_particles = 0
        self.n_solid_particles = 0
        self.x = []
        self.actuator_id = []
        self.particle_type = []
        self.offset_x = 0
        self.offset_y = 0

    def add_rect(self, x, y, w, h, actuation, ptype=1):
        if ptype == 0:
            assert actuation == -1
        global n_particles
        w_count = int(w / dx) * 2
        h_count = int(h / dx) * 2
        real_dx = w / w_count
        real_dy = h / h_count
        for i in range(w_count):
            for j in range(h_count):
                self.x.append([
                    x + (i + 0.5) * real_dx + self.offset_x,
                    y + (j + 0.5) * real_dy + self.offset_y
                ])
                self.actuator_id.append(actuation)
                self.particle_type.append(ptype)
                self.n_particles += 1
                self.n_solid_particles += int(ptype == 1)

    def add_organic_shape(self, shape_params, actuation, ptype=1, irregularity=0.8, density=1.5):
        if ptype == 0:
            assert actuation == -1

        global n_particles
        x,y,w,h = shape_params[actuation+1]
        print(f"Adding shape with params: x={x:.4f}, y={y:.4f}, w={w:.4f}, h={h:.4f}")

        w_count = int(w / dx * density)
        h_count = int(h / dx * density)
        real_dx = w / w_count
        real_dy = h / h_count

        def parametric_shape(t):
            noise_factor = noise.pnoise1(t * 10, repeat=10) * irregularity * random.uniform(-1, 1)
            radius_x = w / 2 * (1 + noise_factor)
            radius_y = h / 2 * (1 + noise_factor)
            px = x + radius_x * math.cos(t)
            py = y + radius_y 
            return px, py

        def point_in_parametric_shape(px, py):
            dx = px - x
            dy = py - y
            r = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            boundary_x, boundary_y = parametric_shape(theta)
            boundary_r = math.hypot(boundary_x - x, boundary_y - y)
            return r <= boundary_r

        for i in range(w_count):
            for j in range(h_count):
                px = x + (i + 0.5) * real_dx + self.offset_x 
                py = y + (j + 0.5) * real_dy + self.offset_y
                if point_in_parametric_shape(px, py):
                    self.x.append([px, py])
                    self.actuator_id.append(actuation)
                    self.particle_type.append(ptype)
                    self.n_particles += 1
                    self.n_solid_particles += int(ptype == 1)

    def set_offset(self, x, y):
        self.offset_x = x
        self.offset_y = y

    def finalize(self):
        global n_particles, n_solid_particles
        n_particles = self.n_particles
        n_solid_particles = self.n_solid_particles
        print('n_particles', n_particles)
        print('n_solid', n_solid_particles)

    def set_n_actuators(self, n_act):
        global n_actuators
        n_actuators = n_act

def robot(scene, shape_params):
    scene.set_n_actuators(4)
    scene.set_offset(0.0, 0.0)
    scene.add_organic_shape(shape_params, -1)
    scene.add_organic_shape(shape_params, 0)
    scene.add_organic_shape(shape_params, 1)
    scene.add_organic_shape(shape_params, 2)
    # scene.add_organic_shape(shape_params, 3)


gui = ti.GUI("Differentiable MPM", (640, 640), background_color=0xFFFFFF)

def visualize(s, folder):
    aid = actuator_id.to_numpy()
    
    particles = x.to_numpy()[s]
    actuation_ = actuation.to_numpy()
    colors = np.empty(particles.shape[0], dtype=np.uint32)

    for i in range(n_particles):
        color = 0x111111
        if aid[i] != -1:
            if 0 <= aid[i] < n_actuators:
                act = actuation_[s - 1, int(aid[i])]
                color = ti.rgb_to_hex((0.5 - act, 0.5 - abs(act), 0.5 + act))
            else:
                print(f"Warning: Invalid actuator_id {aid[i]} for particle {i} there are only {n_actuators} actuators")
        colors[i] = color

    # print(f"Colors shape: {colors.shape}")
    gui.circles(pos=particles, color=colors, radius=1.5)
    gui.line((0.05, 0.02), (0.95, 0.02), radius=3, color=0x0)
    os.makedirs(folder, exist_ok=True)
    gui.show(f'{folder}/{s:04d}.png')


def optimize_iteration(scene, iter, learning_rate):
    print("optimize iter")
    with ti.ad.Tape(loss):
        forward()
    l = loss[None]

    if np.isnan(l):
        raise ValueError("Invalid NaN")
    print('i=', iter, 'loss=', l)


    # Gradient clipping
    clip_value = 1.0  # Set a threshold for gradient clipping
    for i in range(n_actuators):
        for j in range(n_sin_waves):
            grad = weights.grad[i, j]
            if abs(grad) > clip_value:
                grad = clip_value * grad / abs(grad)
            weights[i, j] -= learning_rate * grad
        grad_bias = bias.grad[i]
        if abs(grad_bias) > clip_value:
            grad_bias = clip_value * grad_bias / abs(grad_bias)
        bias[i] -= learning_rate * grad_bias

    # Update shape parameters with gradient clipping
    for i in range(n_actuators):
        for j in range(4):  # x, y, width, height
            grad = shape_params.grad[i][j]
            if abs(grad) > clip_value:
                grad = clip_value * grad / abs(grad)
            shape_params[i][j] += learning_rate * grad * 10
            if j < 2:  # x, y positions
                shape_params[i][j] = ti.max(0.0, ti.min(1.0, shape_params[i][j]))
            else:
                shape_params[i][j] = ti.max(0.05, ti.min(0.5, shape_params[i][j]))


    print(f"shape_penalty_value = {shape_penalty_value[None]}")

    return l

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', type=int, default=100)
    parser.add_argument('--shape', type=int, default=3, choices=[1, 2, 3])
    options = parser.parse_args()

    allocate_fields(4)

    # # Initialize shape parameters (x, y, width, height for each actuator)
    # shape_params[0] = [0.0, 0.1, 0.6, 0.3]  # Base shape
    # shape_params[1] = [0.0, 0.02, 0.1, 0.3]  # Actuator 1
    # shape_params[2] = [0.2, 0.05, 0.03, 0.3]  # Actuator 2
    # shape_params[3] = [0.255, 0.05, 0.03, 0.2]  # Actuator 3
    if options.shape == 1:
        # Shape 1: A simple rectangular shape
        shape_params[0] = [0.0, 0.1, 0.6, 0.3]  # Base shape
        shape_params[1] = [0.0, 0.02, 0.1, 0.3]  # Actuator 1
        shape_params[2] = [0.2, 0.05, 0.03, 0.3]  # Actuator 2
        shape_params[3] = [0.255, 0.05, 0.03, 0.2]  # Actuator 3
    elif options.shape == 2:
        # Shape 2: A more complex shape with wider base
        shape_params[0] = [0.0, 0.1, 0.8, 0.4]  # Base shape
        shape_params[1] = [0.0, 0.02, 0.2, 0.4]  # Actuator 1
        shape_params[2] = [0.3, 0.05, 0.05, 0.4]  # Actuator 2
        shape_params[3] = [0.355, 0.05, 0.05, 0.3]  # Actuator 3
    elif options.shape == 3:
        # Shape 3: Default shape (same as original)
        shape_params[0] = [0.0, 0.1, 0.6, 0.3]  # Base shape
        shape_params[1] = [0.0, 0.02, 0.1, 0.3]  # Actuator 1
        shape_params[2] = [0.2, 0.05, 0.03, 0.3]  # Actuator 2
        shape_params[3] = [0.255, 0.05, 0.03, 0.2]  # Actuator 3
 
    initialize_actuator_id()
    initialize_x()

    losses = []
    for iter in range(options.iters):

        # make scene with new param
        scene = Scene()
        robot(scene, shape_params)
        scene.finalize()
        print("built scene")
        
        for i in range(scene.n_particles):
            x[0, i] = scene.x[i]
            if iter == 0:
                F[0, i] = [[1, 0], [0, 1]]
                actuator_id[i] = scene.actuator_id[i]
                particle_type[i] = scene.particle_type[i]

        l = optimize_iteration(scene, iter, 0.01)
        losses.append(l)

        # if iter % 1 == 0:
        print(f"Shape params: {shape_params}")
        print(f"Shape params grad: {shape_params.grad}")
        forward(1500)
        for s in range(15, 1500, 16):
            visualize(s, 'diffmpm/iter{:03d}/'.format(iter))

    plt.title("Optimization of Shape Parameters")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.show()

if __name__ == '__main__':
    main()