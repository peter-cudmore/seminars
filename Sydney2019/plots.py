from IPython.core.display import HTML
from string import Template
# HTML('''require.config({paths: {d3: "http://d3js.org/d3.v4.min"}});''')
# HTML('<script src = "https://d3js.org/d3.v4.min.js"></script>')
html_template = Template('''
<style> $css_text</style>
<div id="graph-div"></div>
<script> 
require(["d3"], function(d3) {
      $js_text
    });</script>
''')

css_text = '''
.axis path,
.axis line {
    fill:none;
    stroke: #000;
    shape-rendering: crispEdges;
}

.dot {
    stroke: #000;
}

.circle path,
.circle line {
    fill:none;
    stroke: #000;
    shape-rendering: crispEdges;
}
'''


def add_svg(script):

    return HTML(html_template.substitute({
        "css_text":css_text,"js_text":script}))


class D3Plot:
    def __init__(self, container="graph-div"):
        self.container = container
        self.width = 300
        self.height = 300

    def __str__(self):
        canvas, script = self._build_canvas()

        return "\n".join([script, self._draw_unit_circle(canvas)])

    def html(self):
        return HTML(str(self))

    def _build_canvas(self):
        canvas_name = "svg"
        script_string = f'''
        var {canvas_name} = d3.select("#{self.container}")
            .append("svg")
            .attr("width", {self.width})
            .attr("height", {self.height});
        '''
        return canvas_name, script_string

    def _draw_unit_circle(self, canvas_name):
        radius = 0.9 * min(self.width/2, self.height/2)
        return f"""
        {canvas_name}.append("circle")
                     .attr("cx", {self.width/2})
                     .attr("cy", {self.width/2})
                     .attr("r", {radius})
                     .attr("stroke", "black")
                     .attr("fill", "none");
        """

def van_der_pol():

    width = 300
    height = 300

    js_text = f'''
    var data = [
        {{"x": 1, "y":1}},
        {{"x": 0, "y":1}}
    ];
    var period = 100;
    var dt = 0.01;
    var mu = 0; 
    
    var running = 0;
    
    var svg = d3.select("#graph-div")
        .append("svg")
        .attr("width", {width})
        .attr("height", {height});
    
    var pointgroup = svg.append("g")
        .attr("transform", "translate({width/2}, {height/2})");
    
    var axis = svg.append("g")
        .attr("transform", "translate({width/2}, {height/2})");
    
    var unit_circle = axis.append("circle")
        .attr("fill", "none")
        .attr("r",{0.9 * min(width/2, height/2)})
        .attr("cx", 0)
        .attr("cy", 0)
        .attr("stroke","grey");
    
    function update(data) {{
        
        var points = pointgroup.selectAll("circle")
            .data(data);
    
        points.attr("class", "update");
    
        points.enter().append("circle")
                .attr("r", 2)
            .merge(points)
                .attr("cx", function(d, i) {{return 100 * d.x;}})
                .attr("cy", function(d, i) {{return 100 * d.y;}});
                

        points.exit().remove();
    }}

    update(data);
    
    d3.interval(function() {{
        if (running == 1) {{console.log("udated");}}
        
    }}, period);
    '''

    return js_text


# to do
# 1. add a path
# 2. connect path to data
# 3. add a timer
# 4. add a function that starts or stops a timer on mouse click
# 5. add a method to update data on timer tick


import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.widgets import Slider
import matplotlib.animation as animation
import numpy as np
import networkx as nx

class BasicOscillator(animation.TimedAnimation):
    def __init__(self, extent=1, n=5, vector_field=None,scale=2):

        interval = 10
        self._N = n
        fig = plt.figure(figsize=(scale*4,scale*4))
        self._scale = scale
        self.ax_complex = fig.gca()
        self._init_complex_plane(extent)

        self.c_line = Line2D([], [], color='none', marker='.', markersize=scale*10.0,
                             markerfacecolor='r')
        self.c_line_2 = Line2D([], [], color='none', marker='.', markersize=scale*8,
                               markerfacecolor='r', alpha=0.5)
        self.c_line_3 = Line2D([], [], color='none', marker='.', markersize=scale*6,
                               markerfacecolor='r', alpha=0.3)
        self._init_draw()
        self.ax_complex.add_line(self.c_line)
        self.ax_complex.add_line(self.c_line_2)
        self.ax_complex.add_line(self.c_line_3)
        self._timestep = 2 * interval / 1000.0

        animation.TimedAnimation.__init__(self, fig, interval=interval, blit=True)

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
        ax.yaxis.tick_right()
        ax.yaxis.label_position = 'left'
        ax.set_aspect('equal')

    def _ode_generator(self):
        raise NotImplementedError

    def new_frame_seq(self):
        return self._ode_generator()

    def _draw_frame(self, framedata):

        t, z = framedata
        self.c_line_3.set_data(self.c_line_2.get_data())
        self.c_line_2.set_data(self.c_line.get_data())
        self.c_line.set_data(np.real(z), np.imag(z))
        self._drawn_artists = [self.c_line, self.c_line_2, self.c_line_3]


class LimitCycleModel(BasicOscillator):
    def __init__(self, n=5, extent=1.25, omega=4, **kwargs):
        self._mu = 0

        self._z0 = np.random.uniform(low=-1, high=1, size=(n,)) \
                   + 1j * np.random.uniform(low=-1, high=1, size=(n,))

        super().__init__(n=n, extent=extent, vector_field=self._vector_field, **kwargs)
        self._init_quiver(self._vector_field, extent=extent)
        self.fig = plt.gcf()
        self.param_axis = self.fig.add_axes([0.2, 0.9, 0.6, 0.025])
        self.param_slider = Slider(self.param_axis, 'u', -1, 1, valinit=0)

        self.redaw_quiver = False
        self.param_slider.on_changed(self.update)

    def update(self, val):
        self._mu = self.param_slider.val
        self.redaw_quiver = True
        self.fig.canvas.draw_idle()

    def _draw_frame(self, framedata):
        super()._draw_frame(framedata)
        if self.redaw_quiver:
            U, V = self._vector_field(self._X, self._Y)
            self._quiver.set_UVC(U, V)
            self._drawn_artists += [self._quiver]
            self.redaw_quiver = False

    def _init_quiver(self, vector_field, extent=1):
        X, Y = np.meshgrid(
            np.linspace(-1.5*extent, 1.5*extent, 25),
            np.linspace(-1.5*extent, 1.5*extent, 25)
        )
        self._X = X
        self._Y = Y
        U, V = vector_field(X, Y)

        self._quiver = self.ax_complex.quiver(X,Y,U,V)

    def _vector_field(self, X, Y):
        Z = X *1j*Y
        FZ = (self._mu - abs(Z)**2) * Z + 4j*Z
        return np.real(FZ), np.imag(FZ)

    def _ode_generator(self):
        self._running = True

        z = np.array(self._z0)
        dt = 10 ** (-3)
        t = 0
        while self._running:
            mu = self._mu
            t_next = t + self._timestep
            while t < t_next:
                t += dt
                z = [z_i + dt*((mu - abs(z_i)**2) * z_i + 4j*z_i)
                      if 100 > abs(z_i) > 0.01
                      else np.random.uniform(low=-1, high=1) + 1j * np.random.uniform(low=-1, high=1)
                      for i, z_i in enumerate(z)]
            yield (t, z)

class VanDerPolModel(BasicOscillator):

    def __init__(self, mu, n=5, extent=2.5):
        self._mu = mu

        self._z0 = np.random.uniform(low=-1, high=1, size=(n,)) \
                   + 1j * np.random.uniform(low=-1, high=1, size=(n,))

        super().__init__(n=n, extent=extent, vector_field=self._vector_field)
        self._init_quiver(self._vector_field, extent=extent)
        self.fig = plt.gcf()
        self.param_axis = self.fig.add_axes([0.2, 0.9, 0.6, 0.025])
        self.param_slider = Slider(self.param_axis, 'u', -1, 1, valinit=0)

        self.redaw_quiver = False
        self.param_slider.on_changed(self.update)

    def update(self, val):
        self._mu = self.param_slider.val
        self.redaw_quiver = True
        self.fig.canvas.draw_idle()

    def _draw_frame(self, framedata):
        super()._draw_frame(framedata)
        if self.redaw_quiver:
            U, V = self._vector_field(self._X, self._Y)
            self._quiver.set_UVC(U, V)
            self._drawn_artists += [self._quiver]
            self.redaw_quiver = False

    def _init_quiver(self, vector_field, extent=1):
        X, Y = np.meshgrid(
            np.linspace(-1.5*extent, 1.5*extent, 25),
            np.linspace(-1.5*extent, 1.5*extent, 25)
        )
        self._X = X
        self._Y = Y
        U, V = vector_field(X, Y)

        self._quiver = self.ax_complex.quiver(X,Y,U,V)


    def _vector_field(self, X, Y):
        return -Y, X + self._mu * (1 - X ** 2) * Y

    def _ode_generator(self):
        self._running = True

        z = np.array(self._z0)
        dt = 10 ** (-3)
        t = 0
        while self._running:
            mu = self._mu
            t_next = t + self._timestep
            while t < t_next:
                t += dt
                z = [z_i + 1j * dt * (z_i + (mu - np.real(z_i) ** 2) * np.imag(z_i))
                      if 100 > abs(z_i) > 0.001
                      else np.random.uniform(low=-1, high=1) + 1j * np.random.uniform(low=-1, high=1)
                      for i, z_i in enumerate(z)]
            yield (t, z)


class KuramotoModel(BasicOscillator):

    def __init__(self,extent=1, omega=None,**kwargs):
        self._omega = omega
        self._K = 0

        n = len(omega)
        self._theta0 = np.pi*np.random.uniform(low=-1, high=1, size=(n,))

        super().__init__(n=n, extent=extent, **kwargs)

        self.ax_complex.add_artist(
            plt.Circle((0, 0), 1, color='k', alpha=0.3, fill=False)
        )
        self.ax_complex.add_line(
            Line2D([-extent, extent], [0, 0], color='k', linestyle=':', zorder=0))

        self.ax_complex.add_line(
            Line2D([0, 0], [-extent, extent], color='k', linestyle=':', zorder=0))

        self.fig = plt.gcf()
        self.param_axis = self.fig.add_axes([0.2, 0.9, 0.6, 0.025])
        self.param_slider = Slider(self.param_axis, 'K', 0, n, valinit=0)

        self.param_slider.on_changed(self.update)

    def update(self, val):
        self._K = self.param_slider.val
        self.fig.canvas.draw_idle()

    def _ode_generator(self):
        self._running = True

        theta = self._theta0
        omega = self._omega

        dt = 10 ** (-3)
        t = 0
        while self._running:
            t_next = t + self._timestep
            KonN = self._K / self._N
            while t < t_next:
                t += dt
                theta = [theta_i + dt*omega[i] +
                         dt*KonN * sum([np.sin(theta_j - theta_i) for theta_j in theta])
                         for i, theta_i in enumerate(theta)]

            yield (t, [np.exp(1j * theta_j) for theta_j in theta])


class KuramotoOrderModel(KuramotoModel):

    def __init__(self, omega,**kwargs):

        super().__init__(omega=omega, **kwargs)

        self.ax_complex.add_artist(
            plt.Circle((0, 0), 1/np.sqrt(len(self._omega)), color='k', alpha=0.3,
                       fill=False, linestyle=':')
        )
        self.meanfield = [
            Line2D([], [], color='none', marker='o', markersize=self._scale*10.0, markerfacecolor='k'),
            Line2D([], [], color='none', marker='o', markersize=self._scale*8.0, markerfacecolor='k', alpha=0.5),
            Line2D([], [], color='none', marker='o', markersize=self._scale*6.0, markerfacecolor='k', alpha=0.3)
        ]
        for line in self.meanfield:
            self.ax_complex.add_line(line)

    def _draw_frame(self, framedata):
        super()._draw_frame(framedata)
        _, z = framedata
        z_bar = sum(z)/len(z)
        self.meanfield[2].set_data(self.meanfield[1].get_data())
        self.meanfield[1].set_data(self.meanfield[0].get_data())
        self.meanfield[0].set_data(np.real(z_bar), np.imag(z_bar))
        self._drawn_artists += self.meanfield


class DampedCoupledOsc(BasicOscillator):

    def __init__(self, omega,extent=1, **kwargs):
        self._omega = omega
        n = len(omega)
        self._K = 0
        super().__init__(extent=extent, **kwargs)
        self._z0 = np.random.uniform(low=-1, high=1, size=(n,)) \
                   + 1j * np.random.uniform(low=-1, high=1, size=(n,))

        self._N = n
        self.ax_complex.add_line(
            Line2D([-extent, extent], [0, 0], color='k', linestyle=':', zorder=0))

        self.ax_complex.add_line(
            Line2D([0, 0], [-extent, extent], color='k', linestyle=':', zorder=0))

        self.meanfield = [
            Line2D([], [], color='none', marker='o', markersize=self._scale * 10.0, markerfacecolor='k'),
            Line2D([], [], color='none', marker='o', markersize=self._scale * 8.0, markerfacecolor='k', alpha=0.5),
            Line2D([], [], color='none', marker='o', markersize=self._scale * 6.0, markerfacecolor='k', alpha=0.3)
        ]
        for line in self.meanfield:
            self.ax_complex.add_line(line)
        self.fig = plt.gcf()
        self.param_axis = self.fig.add_axes([0.2, 0.9, 0.6, 0.025])
        self.param_slider = Slider(self.param_axis, 'K', 0, n, valinit=0)

        self.param_slider.on_changed(self.update)

    def update(self, val):
        self._K = self.param_slider.val
        self.fig.canvas.draw_idle()

    def _ode_generator(self):
        self._running = True
        omega = self._omega
        z = np.array(self._z0)
        dt = 10 ** (-3)
        t = 0
        while self._running:
            t_next = t + self._timestep
            F0 = self._K
            while t < t_next:
                t += dt
                zbar = sum(z)/len(z)
                zfz = zbar*F0/(1 + abs(zbar)**2)
                z = [z_i + dt *((-1+ 1j*omega[i])*z_i + zfz) for i, z_i in enumerate(z)]
            yield (t, z)

    def _draw_frame(self, framedata):
        super()._draw_frame(framedata)
        _, z = framedata
        z_bar = sum(z)/len(z)
        self.meanfield[2].set_data(self.meanfield[1].get_data())
        self.meanfield[1].set_data(self.meanfield[0].get_data())
        self.meanfield[0].set_data(np.real(z_bar), np.imag(z_bar))
        self._drawn_artists += self.meanfield

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
        # max_z = 2
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

        self._sn = 1. / np.sqrt(nodes)
        if t_end:
            self.rn_line = Line2D([0, t_end], [self._sn, self._sn], linestyle=':')
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

        z = self.z[:, 0]
        dt = 10 ** (-3)
        t = 0
        while self._running:
            t_next = t + self._timestep
            while t < t_next:
                F1 = (L + NL * np.power(np.abs(z), 2)).dot(z)
                k1 = dt * F1
                k2 = dt * (L + NL * np.power(np.abs(z + k1 / 2), 2)).dot(z + k1 / 2)
                k3 = dt * (L + NL * np.power(np.abs(z + k2 / 2), 2)).dot(z + k2 / 2)
                k4 = dt * (L + NL * np.power(np.abs(z + k3), 2)).dot(z + k3)
                z += (k1 + 2 * k2 + 2 * k3 + k4) / 6
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
            dt = 10 ** (-3)
            while t < t_end:
                F1 = (L + NL * np.power(np.abs(z), 2)).dot(z)
                k1 = dt * F1
                k2 = dt * (L + NL * np.power(np.abs(z + k1 / 2), 2)).dot(z + k1 / 2)
                k3 = dt * (L + NL * np.power(np.abs(z + k2 / 2), 2)).dot(z + k2 / 2)
                k4 = dt * (L + NL * np.power(np.abs(z + k3), 2)).dot(z + k3)
                z += (k1 + 2 * k2 + 2 * k3 + k4) / 6
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
            self.ax_r.set_xlim(self.t_max - self.t_max_step, self.t_max + self.t_max_step)
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
                z1 = np.sum(z) / n
                F1 = L.dot(z) + z1 * k / (1 + np.power(abs(z1), 2))
                k1 = dt * F1
                z2 = np.sum(z + k1 / 2) / n
                k2 = dt * (L.dot(z + k1 / 2) +
                           z2 * k / (1 + np.power(abs(z2), 2)))
                z3 = np.sum(z + k2 / 2) / n
                k3 = dt * (L.dot(z + k2 / 2) + z3 * k / (1 + np.power(abs(z3), 2)))
                z4 = np.sum(z + k3 / 2) / n
                k4 = dt * (L.dot(z + k3) + z4 * k / (1 + np.power(abs(z4), 2)))
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

#
# def plot_graph(g=None):
#     if not g:
#         g = nx.watts_strogatz_graph(10, 5, 0.2)
#         labels = {i: str(i) for i in g.nodes}
#         edge_labels = dict()
#         for edge in g.edges:
#             i, j = edge
#             edge_labels[edge] = "$w_{{{i},{j}}}$".format(i=i, j=j)
#         pos = nx.spring_layout(g, k=1)
#
#         nodes = nx.draw_networkx_nodes(g, pos, node_color='white', edgecolors='red', node_size=400)
#         edges = nx.draw_networkx_edges(g, pos, alpha=0.4)
#         lbls = nx.draw_networkx_labels(g, pos)
#         edge_labels = nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=10)
#         ax = plt.gca()
#         ax.xaxis.set_ticks([])
#         ax.yaxis.set_ticks([])
#         ax.set_aspect('equal')
#         return plt.gcf()