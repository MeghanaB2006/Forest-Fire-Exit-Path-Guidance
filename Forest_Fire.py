import pygame
import heapq
import math
import time
from collections import deque
from enum import Enum
import random
import csv
from datetime import datetime

pygame.init()
pygame.mixer.init()

# Constants
WIDTH, HEIGHT, GRID_SIZE = 1400, 800, 20
GRID_WIDTH, GRID_HEIGHT, INFO_PANEL_WIDTH = 35, 30, 300

# Colors
WHITE, BLACK, GREEN, RED, BLUE = (255,255,255), (0,0,0), (0,255,0), (255,0,0), (0,0,255)
YELLOW, ORANGE, PURPLE, CYAN = (255,255,0), (255,165,0), (128,0,128), (0,255,255)
GRAY, DARK_RED, LIGHT_GRAY = (128,128,128), (139,0,0), (200,200,200)
DARK_GRAY, GOLD, DARK_GREEN = (64,64,64), (255,215,0), (0,128,0)

class CellType(Enum):
    EMPTY, OBSTACLE, FIRE, START, EXIT, SMOKE = range(6)

class Node:
    def __init__(self, pos, g=0, h=0, parent=None):
        self.pos, self.g, self.h, self.f, self.parent = pos, g, h, g+h, parent
    def __lt__(self, other): return self.f < other.f

class ForestFirePathfinder:
    def __init__(self):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Forest Fire Exit Path Guidance System - Enhanced AI")
        self.clock = pygame.time.Clock()
        self.font, self.small_font = pygame.font.Font(None, 24), pygame.font.Font(None, 18)
        self.title_font = pygame.font.Font(None, 28)
        
        # Grid initialization
        self.grid = [[CellType.EMPTY]*GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.fire_intensity = [[0]*GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.smoke_density = [[0]*GRID_WIDTH for _ in range(GRID_HEIGHT)]
        self.safe_zones = [[100]*GRID_WIDTH for _ in range(GRID_HEIGHT)]
        
        # State variables
        self.start, self.exits, self.obstacles, self.fires = None, [], [], []
        self.wind_direction, self.wind_strength = [0,0], 0.3
        self.paths, self.algorithm_stats = {}, {}
        self.current_algorithm, self.best_algorithm = None, None
        self.show_all_paths, self.path_animation_offset = False, 0
        self.performance_log, self.reroute_count = [], 0
        self.simulation_start_time = time.time()
        
        self.create_sound_effects()
        self.setup_environment()
        
    def create_sound_effects(self):
        self.sound_enabled = False
        try:
            pygame.mixer.quit()
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
            
            def gen_beep(freq, dur):
                sr, n = 22050, int(22050*dur)
                try:
                    import numpy as np
                    t = np.linspace(0, dur, n)
                    sig = (32767*0.3*np.sin(2*np.pi*freq*t)).astype(np.int16)
                    return pygame.sndarray.make_sound(np.column_stack((sig,sig)))
                except:
                    samps = [[int(32767*0.3*math.sin(2*math.pi*freq*i/sr))]*2 for i in range(n)]
                    return pygame.sndarray.make_sound(pygame.sndarray.array(samps).astype(pygame.int16))
            
            self.warning_sound, self.success_sound = gen_beep(880,0.2), gen_beep(440,0.3)
            self.alert_sound = gen_beep(660,0.15)
            for s in [self.warning_sound, self.success_sound, self.alert_sound]: s.set_volume(0.5)
            self.sound_enabled = True
            print("[SOUND] Initialized")
        except Exception as e:
            print(f"[SOUND] Init failed: {e}")
    
    def play_sound(self, stype):
        if not self.sound_enabled: return
        try:
            {'warning':self.warning_sound,'success':self.success_sound,'alert':self.alert_sound}[stype].play()
        except: pass
        
    def setup_environment(self):
        self.start = (5, 15)
        self.grid[15][5] = CellType.START
        
        for ex in [(32,5), (32,25), (17,2)]:
            self.exits.append(ex)
            self.grid[ex[1]][ex[0]] = CellType.EXIT
        
        for _ in range(80):
            x, y = random.randint(0,GRID_WIDTH-1), random.randint(0,GRID_HEIGHT-1)
            if self.grid[y][x] == CellType.EMPTY and (x,y) != self.start:
                self.grid[y][x] = CellType.OBSTACLE
                self.obstacles.append((x,y))
        
        for fx, fy in [(10,15), (15,12), (12,18), (20,15)]:
            if self.grid[fy][fx] == CellType.EMPTY:
                self.grid[fy][fx], self.fire_intensity[fy][fx] = CellType.FIRE, 100
                self.fires.append((fx,fy))
        self.wind_direction = [1,0]
    
    def shift_wind_direction(self):
        dirs = [[0,1],[1,0],[0,-1],[-1,0],[1,1],[1,-1],[-1,1],[-1,-1]]
        old = self.wind_direction.copy()
        self.wind_direction = random.choice(dirs)
        if self.wind_direction != old:
            print(f"[WIND] {old} → {self.wind_direction}")
            self.play_sound("alert")
    
    def update_safe_zones(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                if self.grid[y][x] != CellType.FIRE:
                    min_dist = min([math.sqrt((x-fx)**2+(y-fy)**2) for fx,fy in self.fires]+[999])
                    self.safe_zones[y][x] = max(0, self.safe_zones[y][x]-(5-min_dist)*5) if min_dist<5 else min(100, self.safe_zones[y][x]+0.5)
    
    def heuristic(self, p1, p2): return math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    def manhattan_distance(self, p1, p2): return abs(p1[0]-p2[0])+abs(p1[1]-p2[1])
    
    def calculate_path_safety(self, path):
        if not path: return 0
        danger = sum((5-min([math.sqrt((x-fx)**2+(y-fy)**2) for fx,fy in self.fires]+[999]))*10 if min([math.sqrt((x-fx)**2+(y-fy)**2) for fx,fy in self.fires]+[999])<5 else 0 + 
                     self.smoke_density[y][x]*0.3 + (100-self.safe_zones[y][x])*0.2 for x,y in path)
        return max(0, 100-danger/len(path))
    
    def is_path_blocked(self, path): return any(self.grid[y][x]==CellType.FIRE for x,y in path) if path else True
    
    def get_neighbors(self, pos):
        x, y = pos
        dirs = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        neighbors = []
        for dx, dy in dirs:
            nx, ny = x+dx, y+dy
            if 0<=nx<GRID_WIDTH and 0<=ny<GRID_HEIGHT and self.grid[ny][nx] not in [CellType.OBSTACLE, CellType.FIRE]:
                cost = 1.414 if abs(dx)+abs(dy)==2 else 1.0
                cost += self.smoke_density[ny][nx]*0.5 + (100-self.safe_zones[ny][nx])*0.01
                min_dist = min([math.sqrt((nx-fx)**2+(ny-fy)**2) for fx,fy in self.fires]+[999])
                cost += (3-min_dist)*2 if min_dist<3 else 0
                neighbors.append(((nx,ny), cost))
        return neighbors
    
    def a_star(self, start, goal):
        st = time.time()
        open_list, closed, came_from, g_score = [], set(), {start:None}, {start:0}
        heapq.heappush(open_list, Node(start, 0, self.heuristic(start, goal)))
        explored = 0
        
        while open_list:
            curr = heapq.heappop(open_list)
            explored += 1
            if curr.pos == goal:
                path = self.reconstruct_path(came_from, curr.pos)
                return path, {'nodes_explored':explored,'path_length':len(path),'path_cost':g_score[curr.pos],
                             'execution_time':time.time()-st,'safety_score':self.calculate_path_safety(path)}
            if curr.pos in closed: continue
            closed.add(curr.pos)
            for nb, cost in self.get_neighbors(curr.pos):
                tent_g = g_score[curr.pos] + cost
                if nb not in g_score or tent_g < g_score[nb]:
                    g_score[nb] = tent_g
                    heapq.heappush(open_list, Node(nb, tent_g, self.heuristic(nb, goal)))
                    came_from[nb] = curr.pos
        return None, {'nodes_explored':explored,'path_length':0,'path_cost':float('inf'),
                     'execution_time':time.time()-st,'safety_score':0}
    
    def bfs(self, start, goal):
        st, q, vis, cf, exp = time.time(), deque([start]), {start}, {start:None}, 0
        while q:
            curr = q.popleft()
            exp += 1
            if curr == goal:
                path = self.reconstruct_path(cf, curr)
                return path, {'nodes_explored':exp,'path_length':len(path),'path_cost':len(path),
                             'execution_time':time.time()-st,'safety_score':self.calculate_path_safety(path)}
            for nb, _ in self.get_neighbors(curr):
                if nb not in vis:
                    vis.add(nb)
                    q.append(nb)
                    cf[nb] = curr
        return None, {'nodes_explored':exp,'path_length':0,'path_cost':float('inf'),
                     'execution_time':time.time()-st,'safety_score':0}
    
    def dfs(self, start, goal):
        st, stk, vis, cf, exp = time.time(), [start], {start}, {start:None}, 0
        while stk:
            curr = stk.pop()
            exp += 1
            if curr == goal:
                path = self.reconstruct_path(cf, curr)
                return path, {'nodes_explored':exp,'path_length':len(path),'path_cost':len(path),
                             'execution_time':time.time()-st,'safety_score':self.calculate_path_safety(path)}
            for nb, _ in self.get_neighbors(curr):
                if nb not in vis:
                    vis.add(nb)
                    stk.append(nb)
                    cf[nb] = curr
        return None, {'nodes_explored':exp,'path_length':0,'path_cost':float('inf'),
                     'execution_time':time.time()-st,'safety_score':0}
    
    def greedy_best_first(self, start, goal):
        st, ol, cf, vis, exp = time.time(), [], {start:None}, set(), 0
        heapq.heappush(ol, (self.heuristic(start, goal), start))
        while ol:
            _, curr = heapq.heappop(ol)
            exp += 1
            if curr in vis: continue
            vis.add(curr)
            if curr == goal:
                path = self.reconstruct_path(cf, curr)
                return path, {'nodes_explored':exp,'path_length':len(path),'path_cost':len(path),
                             'execution_time':time.time()-st,'safety_score':self.calculate_path_safety(path)}
            for nb, _ in self.get_neighbors(curr):
                if nb not in vis:
                    heapq.heappush(ol, (self.heuristic(nb, goal), nb))
                    if nb not in cf: cf[nb] = curr
        return None, {'nodes_explored':exp,'path_length':0,'path_cost':float('inf'),
                     'execution_time':time.time()-st,'safety_score':0}
    
    def dijkstra(self, start, goal):
        st, ol, cf, cost, exp = time.time(), [], {start:None}, {start:0}, 0
        heapq.heappush(ol, (0, start))
        while ol:
            cc, curr = heapq.heappop(ol)
            exp += 1
            if curr == goal:
                path = self.reconstruct_path(cf, curr)
                return path, {'nodes_explored':exp,'path_length':len(path),'path_cost':cost[curr],
                             'execution_time':time.time()-st,'safety_score':self.calculate_path_safety(path)}
            if cc > cost.get(curr, float('inf')): continue
            for nb, c in self.get_neighbors(curr):
                new_c = cost[curr] + c
                if nb not in cost or new_c < cost[nb]:
                    cost[nb] = new_c
                    heapq.heappush(ol, (new_c, nb))
                    cf[nb] = curr
        return None, {'nodes_explored':exp,'path_length':0,'path_cost':float('inf'),
                     'execution_time':time.time()-st,'safety_score':0}
    
    def bidirectional_search(self, start, goal):
        st = time.time()
        qf, vf, qb, vb, exp = deque([start]), {start:None}, deque([goal]), {goal:None}, 0
        while qf and qb:
            if qf:
                cf = qf.popleft()
                exp += 1
                if cf in vb:
                    path = self.merge_paths(vf, vb, cf)
                    return path, {'nodes_explored':exp,'path_length':len(path),'path_cost':len(path),
                                 'execution_time':time.time()-st,'safety_score':self.calculate_path_safety(path)}
                for nb, _ in self.get_neighbors(cf):
                    if nb not in vf:
                        vf[nb] = cf
                        qf.append(nb)
            if qb:
                cb = qb.popleft()
                exp += 1
                if cb in vf:
                    path = self.merge_paths(vf, vb, cb)
                    return path, {'nodes_explored':exp,'path_length':len(path),'path_cost':len(path),
                                 'execution_time':time.time()-st,'safety_score':self.calculate_path_safety(path)}
                for nb, _ in self.get_neighbors(cb):
                    if nb not in vb:
                        vb[nb] = cb
                        qb.append(nb)
        return None, {'nodes_explored':exp,'path_length':0,'path_cost':float('inf'),
                     'execution_time':time.time()-st,'safety_score':0}
    
    def merge_paths(self, vf, vb, mp):
        pf, curr = [], mp
        while curr: pf.append(curr); curr = vf[curr]
        pf.reverse()
        pb, curr = [], vb[mp]
        while curr: pb.append(curr); curr = vb[curr]
        return pf + pb
    
    def reconstruct_path(self, cf, curr):
        path = []
        while curr: path.append(curr); curr = cf[curr]
        path.reverse()
        return path
    
    def find_best_exit(self):
        best_ex, best_path, best_cost = None, None, float('inf')
        for ex in self.exits:
            path, stats = self.a_star(self.start, ex)
            if path and stats['path_cost'] < best_cost:
                best_cost, best_ex, best_path = stats['path_cost'], ex, path
        return best_ex, best_path
    
    def calculate_overall_score(self, st):
        if st['path_cost'] == float('inf'): return 0
        cs = 100/(1+st['path_cost'])
        ts = 100/(1+st['execution_time']*1000)
        efs = 100/(1+st['nodes_explored']/10)
        return st['safety_score']*0.4 + cs*0.3 + efs*0.2 + ts*0.1
    
    def calculate_evacuation_efficiency(self, st):
        return 0 if st['path_cost']==float('inf') or st['execution_time']==0 else st['safety_score']/(st['path_cost']*st['execution_time']*100)
    
    def find_best_algorithm(self):
        best_n, best_s = None, 0
        for n, st in self.algorithm_stats.items():
            sc = self.calculate_overall_score(st)
            st['overall_score'] = sc
            st['evacuation_efficiency'] = self.calculate_evacuation_efficiency(st)
            if sc > best_s: best_s, best_n = sc, n
        return best_n
    
    def run_all_algorithms(self):
        best_ex, _ = self.find_best_exit()
        if not best_ex:
            print("No exit!")
            return
        
        algs = {'A*':self.a_star,'BFS':self.bfs,'DFS':self.dfs,'Greedy':self.greedy_best_first,
                'Dijkstra':self.dijkstra,'Bidirectional':self.bidirectional_search}
        
        self.paths, self.algorithm_stats = {}, {}
        for n, alg in algs.items():
            path, stats = alg(self.start, best_ex)
            self.paths[n], self.algorithm_stats[n] = path, stats
        
        old = self.best_algorithm
        self.best_algorithm = self.find_best_algorithm()
        if old and old != self.best_algorithm:
            print(f"\n[SWITCH] {old} → {self.best_algorithm}")
            self.play_sound("alert")
        
        self.print_ai_summary()
        if self.best_algorithm and self.paths.get(self.best_algorithm): self.play_sound("success")
    
    def check_and_reroute(self):
        if not self.best_algorithm or self.best_algorithm not in self.paths: return
        if self.is_path_blocked(self.paths[self.best_algorithm]):
            print("\n[REROUTE] Blocked!")
            self.play_sound("warning")
            self.reroute_count += 1
            self.run_all_algorithms()
            print(f"[REROUTE] New: {self.best_algorithm}")
    
    def recalculate_safety_scores(self):
        for n, path in self.paths.items():
            if path:
                new_s = self.calculate_path_safety(path)
                old_s = self.algorithm_stats[n]['safety_score']
                self.algorithm_stats[n]['safety_score'] = new_s
                self.algorithm_stats[n]['overall_score'] = self.calculate_overall_score(self.algorithm_stats[n])
                self.algorithm_stats[n]['evacuation_efficiency'] = self.calculate_evacuation_efficiency(self.algorithm_stats[n])
                if old_s - new_s > 15:
                    print(f"[SAFETY] {n}: {old_s:.1f}→{new_s:.1f}")
                    if n == self.best_algorithm: self.play_sound("warning")
    
    def print_ai_summary(self):
        if not self.best_algorithm: return
        st = self.algorithm_stats[self.best_algorithm]
        et = time.time() - self.simulation_start_time
        print("\n"+"="*60+"\nAI EVACUATION ANALYSIS\n"+"="*60)
        print(f"Best: {self.best_algorithm}\nExits: {len(self.exits)}\nSafety: {st['safety_score']:.1f}%")
        print(f"EvacEff: {st['evacuation_efficiency']:.4f}\nCost: {st['path_cost']:.2f}")
        print(f"Time: {st['execution_time']*1000:.1f}ms\nReroutes: {self.reroute_count}\nSimTime: {et:.1f}s")
        status = "EXCELLENT" if st['safety_score']>75 else "MODERATE" if st['safety_score']>50 else "CRITICAL"
        print(f"Status: {status}\n"+"="*60)
    
    def log_performance_to_file(self):
        fn = f"pathfinding_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with open(fn, 'w', newline='') as f:
                w = csv.DictWriter(f, ['Algorithm','Nodes Explored','Path Length','Path Cost',
                                       'Execution Time (ms)','Safety Score','Overall Score',
                                       'Evacuation Efficiency','Is Best'])
                w.writeheader()
                for n, st in self.algorithm_stats.items():
                    w.writerow({'Algorithm':n,'Nodes Explored':st['nodes_explored'],
                               'Path Length':st['path_length'],'Path Cost':f"{st['path_cost']:.2f}",
                               'Execution Time (ms)':f"{st['execution_time']*1000:.2f}",
                               'Safety Score':f"{st['safety_score']:.2f}",
                               'Overall Score':f"{st['overall_score']:.2f}",
                               'Evacuation Efficiency':f"{st['evacuation_efficiency']:.4f}",
                               'Is Best':'Yes' if n==self.best_algorithm else 'No'})
            print(f"[LOG] Saved: {fn}")
        except Exception as e: print(f"[LOG] Error: {e}")
    
    def spread_fire(self):
        new_f, hi = [], 0
        for fx, fy in self.fires:
            for dx, dy in [(0,1),(1,0),(0,-1),(-1,0),(1,1),(-1,-1),(1,-1),(-1,1)]:
                nx, ny = fx+dx, fy+dy
                wb = dx==self.wind_direction[0] and dy==self.wind_direction[1]
                sc = 0.2 if wb else 0.06
                if 0<=nx<GRID_WIDTH and 0<=ny<GRID_HEIGHT and self.grid[ny][nx]==CellType.EMPTY and random.random()<sc:
                    self.grid[ny][nx], self.fire_intensity[ny][nx] = CellType.FIRE, 50
                    new_f.append((nx,ny))
            self.fire_intensity[fy][fx] = min(100, self.fire_intensity[fy][fx]+5)
            if self.fire_intensity[fy][fx] > 80: hi += 1
            for dx in range(-3,4):
                for dy in range(-3,4):
                    sx, sy = fx+dx, fy+dy
                    if 0<=sx<GRID_WIDTH and 0<=sy<GRID_HEIGHT:
                        d = math.sqrt(dx**2+dy**2)
                        self.smoke_density[sy][sx] = min(100, self.smoke_density[sy][sx]+max(0,40-d*10))
        self.fires.extend(new_f)
        if hi > len(self.fires)*0.5: self.play_sound("warning")
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                self.smoke_density[y][x] = max(0, self.smoke_density[y][x]-2)
    
    def draw_grid(self):
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                r = pygame.Rect(x*GRID_SIZE, y*GRID_SIZE, GRID_SIZE, GRID_SIZE)
                ct = self.grid[y][x]
                col = {CellType.EMPTY:WHITE,CellType.OBSTACLE:DARK_GRAY,CellType.START:BLUE,
                       CellType.EXIT:DARK_GREEN}.get(ct)
                if ct == CellType.FIRE:
                    i = self.fire_intensity[y][x]
                    col = (255, max(0,int(165*(100-i)/100)), 0)
                if col: pygame.draw.rect(self.screen, col, r)
                if self.smoke_density[y][x] > 0:
                    s = pygame.Surface((GRID_SIZE,GRID_SIZE))
                    s.set_alpha(int(self.smoke_density[y][x]*1.2))
                    s.fill(GRAY)
                    self.screen.blit(s, r)
                pygame.draw.rect(self.screen, BLACK, r, 1)
    
    def draw_path(self, path, col, w=3, best=False):
        if path and len(path)>1:
            pts = [(p[0]*GRID_SIZE+GRID_SIZE//2, p[1]*GRID_SIZE+GRID_SIZE//2) for p in path]
            if best:
                for ww in range(w+4, w-1, -1):
                    a = int(100-(w+4-ww)*20)
                    gs = pygame.Surface((WIDTH,HEIGHT), pygame.SRCALPHA)
                    pygame.draw.lines(gs, (*GOLD[:3],a), False, pts, ww)
                    self.screen.blit(gs, (0,0))
                for i in range(0, len(pts), 5):
                    if (self.path_animation_offset+i)%10 < 5:
                        pygame.draw.circle(self.screen, GOLD, pts[i], 4)
            else: pygame.draw.lines(self.screen, col, False, pts, w)
    
    def draw_info_panel(self):
        px = GRID_WIDTH*GRID_SIZE
        pygame.draw.rect(self.screen, LIGHT_GRAY, (px,0,INFO_PANEL_WIDTH,HEIGHT))
        y = 10
        
        t = self.title_font.render("Algorithm Analysis", True, BLACK)
        self.screen.blit(t, (px+10, y))
        y += 35
        
        if self.best_algorithm:
            br = pygame.Rect(px+10, y, INFO_PANEL_WIDTH-20, 28)
            pygame.draw.rect(self.screen, GOLD, br)
            pygame.draw.rect(self.screen, BLACK, br, 2)
            bt = self.font.render(f"★ BEST: {self.best_algorithm}", True, BLACK)
            self.screen.blit(bt, (px+20, y+4))
            y += 35
        
        cols = {'A*':RED,'BFS':BLUE,'DFS':GREEN,'Greedy':YELLOW,'Dijkstra':PURPLE,'Bidirectional':CYAN}
        for n, st in sorted(self.algorithm_stats.items(), key=lambda x:x[1].get('overall_score',0), reverse=True):
            cr = pygame.Rect(px+15, y, 12, 12)
            pygame.draw.rect(self.screen, cols[n], cr)
            pygame.draw.rect(self.screen, BLACK, cr, 1)
            nt = self.small_font.render(n, True, BLACK)
            self.screen.blit(nt, (px+35, y-2))
            y += 18
            for l in [f"Score: {st.get('overall_score',0):.1f}",f"Safety: {st['safety_score']:.1f}",
                     f"EvacEff: {st.get('evacuation_efficiency',0):.4f}",f"Cost: {st['path_cost']:.2f}",
                     f"Time: {st['execution_time']*1000:.1f}ms"]:
                txt = self.small_font.render(l, True, BLACK)
                self.screen.blit(txt, (px+35, y))
                y += 15
            y += 5
        
        y += 5
        pygame.draw.line(self.screen, BLACK, (px+10,y), (px+INFO_PANEL_WIDTH-10,y), 2)
        y += 10
        st = self.font.render("System Info", True, BLACK)
        self.screen.blit(st, (px+10, y))
        y += 22
        for i in [f"Reroutes: {self.reroute_count}",f"Fires: {len(self.fires)}",
                 f"Wind: {self.wind_direction}",f"Sound: {'ON' if self.sound_enabled else 'OFF'}"]:
            txt = self.small_font.render(i, True, BLACK)
            self.screen.blit(txt, (px+15, y))
            y += 16
        
        y += 8
        pygame.draw.line(self.screen, BLACK, (px+10,y), (px+INFO_PANEL_WIDTH-10,y), 2)
        y += 10
        lt = self.font.render("Legend", True, BLACK)
        self.screen.blit(lt, (px+10, y))
        y += 22
        for col, lbl in [(BLUE,"Start"),(DARK_GREEN,"Exit"),(ORANGE,"Fire"),(GRAY,"Smoke"),
                        (DARK_GRAY,"Obstacle"),(GOLD,"Best Path")]:
            pygame.draw.rect(self.screen, col, (px+15,y,12,12))
            pygame.draw.rect(self.screen, BLACK, (px+15,y,12,12), 1)
            txt = self.small_font.render(lbl, True, BLACK)
            self.screen.blit(txt, (px+35, y-2))
            y += 16
        
        y += 10
        pygame.draw.line(self.screen, BLACK, (px+10,y), (px+INFO_PANEL_WIDTH-10,y), 2)
        y += 10
        ct = self.font.render("Controls", True, BLACK)
        self.screen.blit(ct, (px+10, y))
        y += 22
        for ctrl in ["SPACE: Run All","B: BEST Path","A: ALL Paths","1-6: Individual",
                    "F: Fire Spread","R: Reset","L: Save Log","S: Toggle Sound","Q: Quit"]:
            txt = self.small_font.render(ctrl, True, BLACK)
            self.screen.blit(txt, (px+15, y))
            y += 16
        
        y += 10
        svw = "All Paths" if self.show_all_paths else self.current_algorithm if self.current_algorithm else "Best" if self.best_algorithm else "None"
        stxt = self.small_font.render(f"View: {svw}", True, DARK_RED)
        self.screen.blit(stxt, (px+15, y))
    
    def run(self):
        running, fsc, wcc, src, sbo = True, 0, 0, 0, False
        self.run_all_algorithms()
        
        while running:
            self.clock.tick(30)
            self.path_animation_offset = (self.path_animation_offset+1)%20
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q: running = False
                    elif event.key == pygame.K_SPACE:
                        self.run_all_algorithms()
                        self.show_all_paths, sbo, self.current_algorithm = True, False, None
                    elif event.key == pygame.K_b:
                        sbo, self.show_all_paths, self.current_algorithm = True, False, None
                    elif event.key == pygame.K_a:
                        self.show_all_paths, sbo, self.current_algorithm = True, False, None
                    elif event.key == pygame.K_f:
                        self.spread_fire()
                        self.update_safe_zones()
                        self.run_all_algorithms()
                    elif event.key == pygame.K_r:
                        self.__init__()
                        self.run_all_algorithms()
                    elif event.key == pygame.K_l: self.log_performance_to_file()
                    elif event.key == pygame.K_s:
                        self.sound_enabled = not self.sound_enabled
                        print(f"[SOUND] {'Enabled' if self.sound_enabled else 'Disabled'}")
                    elif event.key in [pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4, pygame.K_5, pygame.K_6]:
                        algs = {pygame.K_1:'A*',pygame.K_2:'BFS',pygame.K_3:'DFS',
                               pygame.K_4:'Greedy',pygame.K_5:'Dijkstra',pygame.K_6:'Bidirectional'}
                        self.current_algorithm = algs[event.key]
                        self.show_all_paths, sbo = False, False
            
            fsc += 1
            if fsc >= 150:
                self.spread_fire()
                self.update_safe_zones()
                self.check_and_reroute()
                fsc = 0
            
            wcc += 1
            if wcc >= 300:
                self.shift_wind_direction()
                wcc = 0
            
            src += 1
            if src >= 60:
                self.recalculate_safety_scores()
                src = 0
            
            self.screen.fill(WHITE)
            self.draw_grid()
            
            cols = {'A*':RED,'BFS':BLUE,'DFS':GREEN,'Greedy':YELLOW,'Dijkstra':PURPLE,'Bidirectional':CYAN}
            
            if sbo and self.best_algorithm:
                path = self.paths.get(self.best_algorithm)
                if path: self.draw_path(path, cols[self.best_algorithm], 5, True)
            elif self.show_all_paths:
                for n, path in self.paths.items():
                    if path:
                        if n == self.best_algorithm: self.draw_path(path, cols[n], 4, True)
                        else: self.draw_path(path, cols[n], 2)
            elif self.current_algorithm and self.current_algorithm in self.paths:
                path = self.paths[self.current_algorithm]
                if path:
                    ib = self.current_algorithm == self.best_algorithm
                    self.draw_path(path, cols[self.current_algorithm], 4, ib)
            
            self.draw_info_panel()
            pygame.display.flip()
        
        pygame.quit()

if __name__ == "__main__":
    game = ForestFirePathfinder()
    game.run()