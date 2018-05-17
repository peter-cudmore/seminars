import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import networkx as nx


class OscillatorModel(animation.TimedAnimation):
    def __init__(self,
                 nodes=100,
                 coupling_network=nx.complete_graph,
                 distribution=np.random.standard_normal,
                 timesteps=500,
                 t_end=25,
                 symmetric=False,
                 **osc_params):

        self.n = nodes
        if symmetric:
            freq = list(distribution(int(nodes / 2)))
            freq += [-f for f in freq]
            if nodes % 2 == 1:
                freq.append(0)
            self.omega = np.array(sorted(freq))
        else:
            self.omega = np.array(sorted(distribution(int(nodes))))
        self.graph = coupling_network(nodes)
        self.t = np.linspace(0, t_end, timesteps).T
        self.z = np.zeros((nodes, timesteps), dtype=np.complex)

        self.z[:, 0] = np.random.uniform(low=-1, high=1, size=self.z[:,
                                                              0].shape) + 1j * np.random.uniform(
            low=-1, high=1, size=self.z[:, 0].shape)
        self._solve_ode(**osc_params)
        max_z = abs(self.z).max()
        fig = plt.figure()
        gs = GridSpec(2, 5)
        self.ax_phase = plt.subplot(gs[0, :3])
        self.ax_complex = plt.subplot(gs[0, 3:])
        self.ax_r = plt.subplot(gs[1, :])
        self._init_phase_axis()
        self._init_complex_plane(extent=max_z)
        self._init_r_axis(t_end, max_z)
        gs.tight_layout(fig)

        self.o_line = Line2D([], [], color='none', marker='.',
                             markerfacecolor='r')
        self.ax_phase.add_line(self.o_line)
        self.c_line = Line2D([], [], color='none', marker='.',
                             markerfacecolor='r')
        self.cz_line = Line2D([], [], color='none', marker='o',
                              markeredgecolor='b')
        self.ax_complex.add_line(self.c_line)
        self.ax_complex.add_line(self.cz_line)

        self.r_line = Line2D([], [], color='red')
        self.ax_r.add_line(self.r_line)

        animation.TimedAnimation.__init__(self, fig, interval=25, blit=True)

    def _solve_ode(self, alpha=1, omega=0, beta=1, k=1):

        L = np.diag(alpha + 1j * (self.omega + omega)) + k * nx.to_numpy_array(
            self.graph, dtype=np.complex) / self.n
        NL = -beta * np.eye(self.n, dtype=np.complex)

        for i in range(1, self.t.size):
            dt = (self.t[i] - self.t[i - 1])
            zs = np.power(np.abs(self.z[:, i - 1]), 2)
            dz = (L + NL * zs).dot(self.z[:, i - 1])
            self.z[:, i] = self.z[:, i - 1] + dt * dz

    def _init_phase_axis(self):
        ax = self.ax_phase

        ax.set_xticks([0, int(self.n / 2), self.n])
        ax.set_xlim(0, self.n)

        ax.set_ylim(-np.pi, np.pi)
        ax.set_yticks([-np.pi, 0, np.pi])
        ax.set_yticklabels(["$-\pi$", "$0$", "$\pi$"])
        ax.set_title("Oscillator Phase")
        ax.set_ylabel("Phase")
        ax.set_xlabel("Osc. Index")

    def _init_complex_plane(self, extent=1):
        ax = self.ax_complex
        l = int(np.ceil(abs(extent)))
        ax.set_ylim(-1.2 * l, 1.2 * l)
        ax.set_xlim(-1.2 * l, 1.2 * l)
        y_major = range(-l, l + 1)
        x_major = range(-l, l + 1)
        ax.set_xticks(x_major)
        ax.set_yticks(y_major)
        ax.set_ylabel("$\Im$")
        ax.set_xlabel("$\Re$")
        ax.set_title("Complex Plane")
        ax.yaxis.tick_right()
        ax.yaxis.label_position = 'right'
        ax.set_aspect('equal')

    def _init_r_axis(self, t_max=10, r_max=1):
        ax = self.ax_r
        ax.set_xlim(0, t_max)
        ax.set_ylim(0, r_max)
        ax.set_yticks([0, 0.5, 1])
        ax.set_xlabel('time $t$')
        ax.set_ylabel('Order $r$')

    def _init_draw(self):
        lines = [self.r_line, self.cz_line, self.c_line, self.o_line]
        for line in lines:
            line.set_data([], [])

    def new_frame_seq(self):
        return iter(range(self.t.size))

    def _draw_frame(self, framedata):
        i = framedata
        z = self.z[:, i]
        zsum = np.sum(z) / self.n
        theta_avg = np.angle(zsum)
        theta = np.remainder(np.pi + np.angle(z) - theta_avg, 2 * np.pi) - np.pi

        self.o_line.set_data(list(range(self.n)), theta)

        self.c_line.set_data(np.real(z), np.imag(z))
        self.cz_line.set_data([np.real(zsum)], [np.imag(zsum)])

        rx, ry = self.r_line.get_data()
        rx.append(self.t[i])
        ry.append(abs(zsum))

        self.r_line.set_data(rx, ry)
        self._drawn_artists = [self.o_line, self.c_line, self.cz_line,
                               self.r_line]