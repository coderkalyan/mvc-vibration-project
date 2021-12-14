from manim import *
#from math import *
import numpy as np

class HarmonicMotion(Scene):
    def construct(self):
        amp = 1
        omega = 1
        zeta = 0.1
        omega_d = omega * np.sqrt(1 - zeta**2)

        axes = Axes(
                x_range=[-1, 7 * np.pi, np.pi / omega_d],
                y_range=[-1.25 * amp, 1.25 * amp, 0.25],
        )
        
        shm_graph = axes.plot(lambda t: amp * np.sin(omega_d * t), color=BLUE, x_range=[0, 6.75 * np.pi])

        self.play(Create(axes, run_time=2))
        self.wait()

        self.play(Create(shm_graph, run_time=3))
        self.wait()

        self.play(shm_graph.animate.set_color(color=GRAY))

        exp_graph_top = axes.plot(lambda t: amp * np.exp(-zeta * omega * t), color=RED, x_range=[0, 6.75 * np.pi])
        exp_graph_bottom = axes.plot(lambda t: -1 * amp * np.exp(-zeta * omega * t), color=RED, x_range=[0, 6.75 * np.pi])
        exp_graph = VGroup(exp_graph_top, exp_graph_bottom)
        self.play(Create(exp_graph, run_time=3))

        dhm_shm_graph = axes.plot(lambda t: amp * np.sin(omega_d * t), color=GRAY, x_range=[0, 6.75 * np.pi])
        self.add(dhm_shm_graph)
        dhm_graph = axes.plot(lambda t: amp * np.sin(omega_d * t) * np.exp(-zeta * omega * t), color=BLUE, x_range=[0, 6.75 * np.pi])
        self.play(ReplacementTransform(dhm_shm_graph, dhm_graph, run_time=2))
        self.play(dhm_shm_graph.animate.set_color(color=BLUE))


class DestructiveInterference(Scene):
    omega1 = ValueTracker(1)
    amp1 = ValueTracker(1)
    zeta1 = ValueTracker(0.1)
    t1 = ValueTracker(0)
    omega2 = ValueTracker(1)
    amp2 = ValueTracker(1)
    zeta2 = ValueTracker(0.1)
    t2 = ValueTracker(0)

    def dhm(self, t, amp, omega, zeta, shift):
        w = omega.get_value()
        d = zeta.get_value()
        a = amp.get_value()
        wd = w * np.sqrt(1 - d**2)
        tn = shift.get_value()
        if t < tn:
            return 0

        val = a * np.sin(wd * (t - tn)) * np.exp(-d * w * (t - tn))

        return val

    def construct(self):
        w1 = self.omega1.get_value()
        d1 = self.zeta1.get_value()
        wd1 = w1 * np.sqrt(1 - d1**2)
        a1 = self.amp1.get_value()
        axes = Axes(
                x_range=[-w1, 7 * np.pi / w1, np.pi / wd1],
                y_range=[-1.25 * a1, 1.25 * a1, 0.25],
        )
        
        self.add(axes)

        disp1 = always_redraw(lambda: axes.plot(lambda t: self.dhm(t, self.amp1, self.omega1, self.zeta1, self.t1), x_range=[self.t1.get_value(), 6.75 * np.pi / wd1], color=RED))
        self.add(disp1)
        disp2 = always_redraw(lambda: axes.plot(lambda t: self.dhm(t, self.amp2, self.omega2, self.zeta2, self.t2), x_range=[self.t2.get_value(), 6.75 * np.pi / wd1], color=BLUE))
        self.add(disp2)


        tot = always_redraw(lambda: axes.plot(lambda t: self.dhm(t, self.amp1, self.omega1, self.zeta1, self.t1) + self.dhm(t, self.amp2, self.omega2, self.zeta2, self.t2), x_range=[0, 6.75 * np.pi / wd1], color=GREEN))
        self.add(tot)

        self.wait()
        self.play(self.t2.animate.set_value(np.pi / wd1), run_time=2)
        self.wait()
        self.play(self.omega2.animate.set_value(wd1 * 1.2), run_time=2)
        self.play(self.omega2.animate.set_value(wd1 * 0.83), run_time=2)
        self.play(self.omega2.animate.set_value(wd1 * 1.0), run_time=2)
        self.wait()
        a_target = a1 * np.exp(-d1 * w1 * np.pi)
        self.play(self.amp2.animate.set_value(a1 * 1.2), run_time=2)
        self.play(self.amp2.animate.set_value(a_target * 0.83), run_time=2)
        self.play(self.amp2.animate.set_value(a_target * 1.0), run_time=2)
        # self.play(
        # omega_tracker = ValueTracker(self.omega)

        # vib_graph = axes.plot(lambda t: self.amp * np.sin(omega_d * t) * np.exp(-self.zeta * self.omega * t), color=RED, x_range=[0, 6.75 * np.pi])
        # # self.play(Create(vib_graph, run_time=3))

        # def get_dhm_graph():
        #     return axes.plot(lambda t: self.dhm(t, omega_tracker), color=BLUE, x_range=[0, 6.75 * np.pi])
        # dhm_graph = get_dhm_graph()
        # # self.play(Create(dhm_graph, run_time=3))
        # dhm_graph = always_redraw(get_dhm_graph)
        # self.add(dhm_graph)



class RightAngle(Scene):
    def construct(self):
        horizontal = Line(start=np.array([-2.5, 2, 0]), end=np.array([2.5, 2, 0]), color=BLUE)
        vertical = Line(start=np.array([2.5, 2, 0]), end=np.array([2.5, -2, 0]), color=BLUE)
        self.add(horizontal, vertical)
        # self.wait()

        toolhead = Dot(point=np.array([-2.5, 2, 0]), color=RED)
        self.play(Create(toolhead))
        self.play(MoveAlongPath(toolhead, horizontal, run_time=2.5, rate_func=linear))

        tracker = ValueTracker(0)

        def vibrate(dot, dt):
            amp = 0.2
            omega = 40
            zeta = 0.1
            omega_d = omega * np.sqrt(1 - zeta**2)
            posx = amp * np.sin(omega_d * tracker.get_value()) * np.exp(-zeta * omega * tracker.get_value())
            posx += 2.5
            dot.match_y(Dot(vertical.point_from_proportion(tracker.get_value())))
            dot.set_x(posx)

        trace = TracedPath(toolhead.get_center, color=RED)
        self.add(trace)
        toolhead.add_updater(vibrate)
        self.play(tracker.animate.set_value(1), run_time=2.5, rate_func=linear, dt=0.00001)
