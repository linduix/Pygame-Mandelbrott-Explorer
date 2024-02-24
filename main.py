import pygame as pg
import numpy as np
from numba import njit, prange

res = width, height = 1600, 900


@njit(fastmath=True)
def interpolate_color(start_color, end_color, t):
    if t >= .99:
        return 0

    # Extract RGB components from start and end colors
    r1, g1, b1 = start_color
    r2, g2, b2 = end_color

    # Interpolate RGB components
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)

    return (r << 16) + (g << 8) + b


class Fractal():
    def __init__(self, app):
        self.app: App = app
        self.surfarray: np.ndarray = np.zeros(res, dtype=float)
        self.offsetx, self.offsety = 1, 1
        self.zoom: float = 2.2
        self.scale: float = self.zoom / height
        self.increment = [0.5, 0.0]
        self.max_iter: float = 100.0

    def render(self):
        scale: float = 1+np.log(1/self.zoom)
        cap: int = 700
        scalediter: int = min(int(self.max_iter*scale), cap)
        self.surfarray = self.mandelbrott(self.surfarray, self.scale, self.increment, scalediter)

    @staticmethod
    @njit(fastmath=True, parallel=True)
    def mandelbrott(surfarray, scale, increment, max_iter):
        start_color =  (40, 10, 80) # Purple
        end_color = (255, 200, 20)  # Yellow
        for x in prange(width):
            for y in prange(height):
                c: np.complexfloating = (x * scale - increment[0] +
                                         1j * (y * scale - increment[1]))

                z: np.complexfloating = np.complexfloating(0)
                ix: int = 0
                for ix in range(max_iter):
                    z = z ** 2 + c
                    if z.real**2 + z.imag**2 > 4:
                        break

                norm = (ix+1) / max_iter
                col: int = interpolate_color(start_color, end_color, norm)
                surfarray[x, y] = col

        return surfarray

    def update(self):
        self.offset = np.array([self.offsetx * width, self.offsety * height]) // 2
        self.scale = self.zoom / height
        self.render()

    def draw(self):
        pg.surfarray.blit_array(self.app.screen, self.surfarray)

    def run(self):
        self.update()
        self.draw()


class App(object):
    def __init__(self):
        self.screen = pg.display.set_mode(res, pg.SCALED)
        self.clock = pg.time.Clock()
        self.fractal = Fractal(self)

    def run(self):
        dt: float = 1.0
        zoomin = False
        zoomout = False
        left = False
        right = False
        up = False
        down = False
        velocity = .25

        while True:
            self.screen.fill('black')
            self.fractal.run()
            pg.display.flip()

            for event in pg.event.get():
                match event.type:
                    case pg.QUIT:
                        exit()
                    case pg.KEYDOWN:
                        match event.key:
                            case pg.K_w:
                                zoomin = True
                            case pg.K_s:
                                zoomout = True
                            case pg.K_LEFT:
                                left = True
                            case pg.K_RIGHT:
                                right = True
                            case pg.K_UP:
                                up = True
                            case pg.K_DOWN:
                                down = True
                            case pg.K_i:
                                self.fractal.max_iter *= 2
                            case pg.K_o:
                                self.fractal.max_iter /= 2
                    case pg.KEYUP:
                        match event.key:
                            case pg.K_w:
                                zoomin = False
                            case pg.K_s:
                                zoomout = False
                            case pg.K_LEFT:
                                left = False
                            case pg.K_RIGHT:
                                right = False
                            case pg.K_UP:
                                up = False
                            case pg.K_DOWN:
                                down = False

            if zoomin:
                factor = (0.2 * dt)
                self.fractal.zoom *= 1 - factor
                velocity *= 1 - factor
            if zoomout:
                factor = (0.2 * dt)
                self.fractal.zoom *= 1 + factor
                velocity *= 1 + factor
            if left:
                self.fractal.increment[0] += velocity * dt
            if right:
                self.fractal.increment[0] -= velocity * dt
            if up:
                self.fractal.increment[1] += velocity * dt
            if down:
                self.fractal.increment[1] -= velocity * dt

            dt = self.clock.tick()/1000
            # print(dt)
            pg.display.set_caption(f'FPS: {self.clock.get_fps():.2f}')


if __name__ == '__main__':
    app = App()
    app.run()