import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
import networkx as nx


class OscillatorModel(animation.TimedAnimation):
    def __init__(self,
                 nodes=50,
                 coupling_network=nx.complete_graph,
                 distribution=np.random.standard_normal,
                 timesteps=200,
                 t_end=None,
                 symmetric=False,
                 interval=25,
                 **osc_params):
        """
        :param nodes: The number of oscillators to simulate (default 50)
        :param coupling_network: Networkx graph construction function
        representing the coupling topology (default is nx.complete_graph). The
        function should take a single argument, the number of nodes, and return
        a graph.
        :param distribution: Function to generate the frequency distribution.
        It should take an single argument, the number of nodes, and return a
        numpy array containing the frequency   
        :param timesteps:
        :param t_end:
        :param symmetric:
        :param interval:
        :param osc_params:
        """

        self.t_end = t_end
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
        if t_end:
            self.t = np.linspace(0, t_end, timesteps).T
            self.z = np.zeros((nodes, timesteps), dtype=np.complex)
        else:
            self.z = np.zeros((nodes, 1), dtype=np.complex)

        self.z[:, 0] = np.random.uniform(low=-1, high=1, size=self.z[:,
                                                              0].shape) + 1j * np.random.uniform(
            low=-1, high=1, size=self.z[:, 0].shape)
        if t_end:
            self._solve_ode(**osc_params)

        max_z = max(min(abs(self.z).max(), 5), 1)
        #max_z = 2
        fig = plt.figure()
        gs = GridSpec(2, 5)
        self.ax_phase = plt.subplot(gs[0, :3])
        self.ax_complex = plt.subplot(gs[0, 3:])
        self.ax_r = plt.subplot(gs[1, :])
        self._init_phase_axis()
        self._init_complex_plane(extent=max_z)
        if not t_end:
            self.t_max = 50
        else:
            self.t_max = t_end
        self.t_max_step = 25

        self._init_r_axis(t_max=self.t_max, r_max=max_z)

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

        self._sn = 1./np.sqrt(nodes)
        if t_end:
            self.rn_line = Line2D([0, t_end],[self._sn, self._sn], linestyle=':')
        else:
            self.rn_line = Line2D([0, self.t_max], [self._sn, self._sn],
                                  linestyle=':')
        self._running = False
        self.ax_r.add_line(self.rn_line)
        self._gen_args = osc_params
        self._timestep = 2 * interval / 1000.0
        animation.TimedAnimation.__init__(self, fig, interval=interval, blit=True)

    def _ode_generator(self, alpha=1, omega=0, beta=1, k=1):
        L = np.diag(alpha + 1j * (self.omega + omega)) + k * nx.to_numpy_array(
            self.graph, dtype=np.complex) / self.n
        NL = -beta * np.eye(self.n, dtype=np.complex)
        self._running = True

        z = self.z[:,0]
        dt = 10 ** (-3)
        t = 0
        while self._running:
            t_next = t + self._timestep
            while t < t_next:
                F1 = (L + NL * np.power(np.abs(z), 2)).dot(z)
                k1 = dt*F1
                k2 = dt * (L + NL * np.power(np.abs(z+k1/2), 2)).dot(z + k1/2)
                k3 = dt * (L + NL * np.power(np.abs(z + k2 / 2), 2)).dot(z + k2 / 2)
                k4 = dt * (L + NL * np.power(np.abs(z + k3), 2)).dot(z + k3)
                z += (k1 + 2*k2 + 2*k3 + k4)/6
                t += dt

            yield (t, z)

    def _solve_ode(self, alpha=1, omega=0, beta=1, k=1):
        L = np.diag(alpha + 1j * (self.omega + omega)) + k * nx.to_numpy_array(
            self.graph, dtype=np.complex) / self.n
        NL = -beta * np.eye(self.n, dtype=np.complex)
        for i in range(1, self.t.size):
            t = self.t[i - 1]
            t_end = self.t[i]
            z = self.z[:, i - 1]
            dt = 10**(-3)
            while t < t_end:
                F1 = (L + NL * np.power(np.abs(z), 2)).dot(z)
                k1 = dt*F1
                k2 = dt * (L + NL * np.power(np.abs(z+k1/2), 2)).dot(z + k1/2)
                k3 = dt * (L + NL * np.power(np.abs(z + k2 / 2), 2)).dot(z + k2 / 2)
                k4 = dt * (L + NL * np.power(np.abs(z + k3), 2)).dot(z + k3)
                z += (k1 + 2*k2 + 2*k3 + k4)/6
                t += dt

            self.z[:, i] = z

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
        try:
            l = int(np.ceil(abs(extent)))
        except ValueError:
            l = 2
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

    # def new_frame_seq(self):
    #     return iter(range(self.t.size))

    def new_frame_seq(self):
        if self.t_end:
            return iter((self.t[i], self.z[:,i]) for i in range(self.t.size))
        else:
            return self._ode_generator(**self._gen_args)

    def _draw_frame(self, framedata):

        t, z = framedata

        zsum = np.sum(z) / self.n

        theta_avg = np.angle(zsum)
        theta = np.remainder(np.pi + np.angle(z) - theta_avg, 2 * np.pi) - np.pi

        self.o_line.set_data(list(range(self.n)), theta)

        self.c_line.set_data(np.real(z), np.imag(z))
        self.cz_line.set_data([np.real(zsum)], [np.imag(zsum)])

        rx, ry = self.r_line.get_data()

        rx.append(t)
        ry.append(abs(zsum))

        if t + 10 > self.t_max :
            self.ax_r.set_xlim(self.t_max-self.t_max_step, self.t_max+self.t_max_step)
            self.t_max += self.t_max_step
            slice = int(self.t_max_step*self._timestep *1000)
            rx = rx[slice:]
            ry = ry[slice:]
            self.rn_line.set_data([rx[0], rx[-1]], [self._sn, self._sn])
            self._drawn_artists = [self.rn_line]
        else:
            self._drawn_artists = []

        self.r_line.set_data(rx, ry)

        self._drawn_artists += [self.o_line, self.c_line, self.cz_line,
                                self.r_line]


class NonlinearOscillatorModel(OscillatorModel):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)    

    def _ode_generator(self, alpha=-1, omega=0, beta=0, k=1):

        L = np.diag(alpha + 1j * (self.omega + omega))
        self._running = True

        z = self.z[:, 0]
        n = self.n
        dt = 10 ** (-3)
        t = 0
        step = self._timestep
        while self._running:
            t_next = t + step
            while t < t_next:
                z1 = np.sum(z)/n
                F1 = L.dot(z) + z1*k/(1+ np.power(abs(z1),2))
                k1 = dt * F1
                z2 = np.sum(z + k1/2)/n
                k2 = dt *(L.dot(z + k1 / 2) +
                          z2*k/(1 + np.power(abs(z2),2)))
                z3 = np.sum(z + k2 / 2)/n
                k3 = dt * (L.dot(z + k2 / 2) + z3*k/(1 + np.power(abs(z3),2)))
                z4 = np.sum(z + k3 / 2)/n
                k4 = dt * (L.dot(z + k3) + z4*k/(1 + np.power(abs(z4), 2)))
                z += (k1 + 2 * k2 + 2 * k3 + k4) / 6
                t += dt

            yield (t, z)

    def _init_draw(self):
        lines = [self.r_line, self.cz_line, self.c_line, self.o_line]
        for line in lines:
            line.set_data([], [])

        # def new_frame_seq(self):
        #     return iter(range(self.t.size))

    def new_frame_seq(self):
        if self.t_end:
            return iter((self.t[i], self.z[:, i]) for i in range(self.t.size))
        else:
            return self._ode_generator(**self._gen_args)

    def _draw_frame(self, framedata):

        t, z = framedata

        zsum = np.sum(z) / self.n

        theta_avg = np.angle(zsum)
        theta = np.remainder(np.pi + np.angle(z) - theta_avg, 2 * np.pi) - np.pi

        self.o_line.set_data(list(range(self.n)), theta)

        self.c_line.set_data(np.real(z), np.imag(z))
        self.cz_line.set_data([np.real(zsum)], [np.imag(zsum)])

        rx, ry = self.r_line.get_data()

        rx.append(t)
        ry.append(abs(zsum))

        if t + 10 > self.t_max:
            self.ax_r.set_xlim(self.t_max - self.t_max_step,
                               self.t_max + self.t_max_step)
            self.t_max += self.t_max_step
            slice = int(self.t_max_step * self._timestep * 1000)
            rx = rx[slice:]
            ry = ry[slice:]
            self.rn_line.set_data([rx[0], rx[-1]], [self._sn, self._sn])
            self._drawn_artists = [self.rn_line]
        else:
            self._drawn_artists = []

        self.r_line.set_data(rx, ry)

        self._drawn_artists += [self.o_line, self.c_line, self.cz_line,
                                self.r_line]

def plot_graph(g=None):
    if not g:
        g = nx.watts_strogatz_graph(10,5,0.2)
        labels = {i: str(i) for i in g.nodes}
        edge_labels = dict()
        for edge in g.edges:
            i,j = edge
            edge_labels[edge] = "$w_{{{i},{j}}}$".format(i=i,j=j)
        pos = nx.spring_layout(g, k=1)

        nodes =nx.draw_networkx_nodes(g,pos, node_color='white', edgecolors='red', node_size=400)
        edges = nx.draw_networkx_edges(g,pos, alpha=0.4)
        lbls = nx.draw_networkx_labels(g,pos)
        edge_labels = nx.draw_networkx_edge_labels(g,pos,edge_labels=edge_labels, font_size=10)
        ax = plt.gca()
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
        ax.set_aspect('equal')
        return plt.gcf()