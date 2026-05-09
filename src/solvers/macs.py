import os
import random
import time
import threading
import queue
import numpy as np
from numba import njit
from datetime import datetime, timedelta

from config.config import DC_CONFIG
from solvers.vnd import VND
from utils.early_stopper import EarlyStopper

# ─────────────────────────────────────────────────────────────────────────────
# Internal representation helpers
# ─────────────────────────────────────────────────────────────────────────────

def stores_to_internal(remaining_stores, distance_matrix, time_matrix, capacity, time_limit, depot_ready=0.0):
    """
    Convert the store-dict format into the internal node format
    used by the MACS-VRPTW solver.
    """
    base_dt = datetime(2024, 1, 1, 0, 0, 0)
    store_ids = sorted(remaining_stores, key=lambda s: int(s.get('store_id', 0)) if str(s.get('store_id')).isdigit() else str(s.get('store_id')))

    # Depot node
    depot = {
        'idx': 0,
        'sid': 'dc',
        'demand': 0.0,
        'ready_time': depot_ready,
        'due_date': time_limit,
        'service_time': 0.0,
    }
    nodes_internal = [depot]

    for k, s in enumerate(store_ids, start=1):
        sid = s['store_id']
        rt = datetime.fromisoformat(s['earliest_time'])
        dt = datetime.fromisoformat(s['latest_time'])
        ready  = (rt - base_dt).total_seconds() / 60
        due    = (dt - base_dt).total_seconds() / 60
        nodes_internal.append({
            'idx': k,
            'sid': sid,
            'demand': float(s['volume']),
            'ready_time': ready,
            'due_date': due,
            'service_time': float(s['dwell_time']),
            'store_info': s,
        })

    n = len(nodes_internal)
    dist = [[0.0] * n for _ in range(n)]
    time_mat = [[0.0] * n for _ in range(n)]
    for i, ni in enumerate(nodes_internal):
        for j, nj in enumerate(nodes_internal):
            if i != j:
                dist[i][j] = distance_matrix.get(ni['sid'], {}).get(nj['sid'], 0.0)
                time_mat[i][j] = time_matrix.get(ni['sid'], {}).get(nj['sid'], 0.0)

    return nodes_internal, dist, time_mat, float(capacity), time_limit

def solution_to_run_solomon_fmt(routes_internal, nodes_internal):
    result = {}
    for v_id, route in enumerate(routes_internal):
        stores_list = []
        for idx in route:
            ni = nodes_internal[idx]
            stores_list.append({'store_id': ni['sid']})
        result[v_id] = {'stores': stores_list}
    return result

# ─────────────────────────────────────────────────────────────────────────────
# MACS Core Implementation
# ─────────────────────────────────────────────────────────────────────────────

@njit(cache=True)
def _njit_eval_route(route, nodes_arr, dist_mat, time_mat, capacity, is_solomon):
    total_dist = 0.0
    cur_time = nodes_arr[0, 1] # ready_time
    load = 0.0
    prev = 0
    for idx in route:
        demand = nodes_arr[idx, 0]
        ready = nodes_arr[idx, 1]
        due = nodes_arr[idx, 2]
        service = nodes_arr[idx, 3]
        
        arrival = cur_time + time_mat[prev, idx]
        if arrival > due:
            return False, 0.0, load
        
        if is_solomon:
            start = max(arrival, ready)
        else:
            if arrival < ready:
                return False, 0.0, load
            start = arrival
        
        cur_time = start + service
        total_dist += dist_mat[prev, idx]
        load += demand
        if load > capacity:
            return False, 0.0, load
        prev = idx
    
    total_dist += dist_mat[prev, 0]
    if cur_time + time_mat[prev, 0] > nodes_arr[0, 2]: # nodes_arr[0, 2] is depot due_date
        return False, 0.0, load
    return True, total_dist, load

def _eval_route(route, nodes_arr, dist, time_mat, capacity, is_solomon):
    return _njit_eval_route(np.array(route, dtype=np.int32), nodes_arr, dist, time_mat, capacity, is_solomon)

def _solution_cost(routes, nodes_arr, dist, time_mat, capacity, is_solomon):
    total = 0.0
    feasible = True
    visited_count = 0
    for r in routes:
        f, d, _ = _njit_eval_route(np.array(r, dtype=np.int32), nodes_arr, dist, time_mat, capacity, is_solomon)
        if not f:
            feasible = False
        total += d
        visited_count += len(r)
    if visited_count != len(nodes_arr) - 1:
        feasible = False
    return feasible, total

def _nearest_neighbor(nodes, dist, time_mat, capacity, is_solomon):
    depot = nodes[0]
    unvisited = set(range(1, len(nodes)))
    routes = []

    while unvisited:
        route = []
        load = 0.0
        cur_time = nodes[0]['ready_time']
        cur = 0

        while True:
            best = None
            best_dist = float('inf')
            for cid in unvisited:
                nd = nodes[cid]
                if load + nd['demand'] > capacity:
                    continue
                travel_time = time_mat[cur][cid]
                arrival = cur_time + travel_time
                if arrival > nd['due_date']:
                    continue
                if is_solomon:
                    start = max(arrival, nd['ready_time'])
                else:
                    if arrival < nd['ready_time']:
                        continue
                    start = arrival
                
                if start + nd['service_time'] + time_mat[cid][0] > depot['due_date']:
                    continue
                if dist[cur][cid] < best_dist:
                    best_dist = dist[cur][cid]
                    best = cid
            if best is None:
                break
            nd = nodes[best]
            travel_time = time_mat[cur][best]
            arrival = cur_time + travel_time
            start = max(arrival, nd['ready_time']) if is_solomon else arrival
            cur_time = start + nd['service_time']
            load += nd['demand']
            route.append(best)
            unvisited.remove(best)
            cur = best

        if route:
            routes.append(route)
        elif unvisited:
            cid = next(iter(unvisited))
            routes.append([cid])
            unvisited.remove(cid)

    return routes

def _insertion(routes, nodes_arr, dist, time_mat, capacity, is_solomon):
    visited_count = sum(len(r) for r in routes)
    if visited_count == len(nodes_arr) - 1:
        return routes
        
    visited = set()
    for r in routes:
        for idx in r: visited.add(idx)
        
    unvisited = sorted(
        [c for c in range(1, len(nodes_arr)) if c not in visited],
        key=lambda c: -nodes_arr[c, 0] # demand
    )
    for cid in unvisited:
        best_cost = float('inf')
        best_r, best_p = -1, -1
        cid_arr = np.array([cid], dtype=np.int32)
        
        for ri, route in enumerate(routes):
            r_arr = np.array(route, dtype=np.int32)
            for pos in range(len(route) + 1):
                # Faster way to build new route for evaluation
                new_r = np.concatenate((r_arr[:pos], cid_arr, r_arr[pos:]))
                f, cost, _ = _njit_eval_route(new_r, nodes_arr, dist, time_mat, capacity, is_solomon)
                if f and cost < best_cost:
                    best_cost = cost
                    best_r, best_p = ri, pos
        if best_r != -1:
            routes[best_r].insert(best_p, cid)
    return routes

class _ACSColony:
    def __init__(self, nodes, nodes_arr, dist, time_mat, capacity, num_ants, beta, rho, q0, tau0, is_solomon, vnd=None, dc_config=None):
        self.nodes = nodes
        self.nodes_arr = nodes_arr
        self.dist = dist
        self.time_mat = time_mat
        self.capacity = capacity
        self.n = len(nodes)
        self.num_ants = num_ants
        self.beta = beta
        self.rho = rho
        self.q0 = q0
        self.tau0 = tau0
        self.is_solomon = is_solomon
        self.ph = np.full((self.n, self.n), tau0, dtype=np.float64)
        self.vnd = vnd
        self.dc_config = dc_config

    def _local_upd(self, i, j):
        self.ph[i][j] = (1 - self.rho) * self.ph[i][j] + self.rho * self.tau0

    def _global_upd(self, routes, cost):
        """ACS Global Update: Reinforced only on Global Best tour edges."""
        if cost <= 0 or cost == float('inf'):
            return
        delta = 1.0 / cost
        inc = self.rho * delta
        
        # Evaporation and reinforcement only on the best tour
        for route in routes:
            full = [0] + route + [0]
            for k in range(len(full) - 1):
                a, b = full[k], full[k + 1]
                self.ph[a][b] = (1.0 - self.rho) * self.ph[a][b] + inc
                self.ph[b][a] = (1.0 - self.rho) * self.ph[b][a] + inc

    def _eta(self, i, j, in_j, cur_time):
        nd = self.nodes[j]
        travel_time = self.time_mat[i][j]
        arrival = cur_time + travel_time
        delivery = max(arrival, nd['ready_time']) if self.is_solomon else arrival
        delta = delivery - cur_time
        d = delta * max(1.0, (nd['due_date'] - cur_time))
        d = max(1.0, d - in_j)
        return 1.0 / d

    def _choose(self, cur, feasible, in_vec, cur_time):
        if not feasible:
            return None
        if random.random() < self.q0:
            best_val, best = -1.0, None
            for j in feasible:
                val = self.ph[cur][j] * (self._eta(cur, j, in_vec[j], cur_time) ** self.beta)
                if val > best_val:
                    best_val, best = val, j
            return best
        nodes_list = list(feasible)
        weights = [self.ph[cur][j] * (self._eta(cur, j, in_vec[j], cur_time) ** self.beta)
                   for j in nodes_list]
        total = sum(weights)
        if total == 0:
            return random.choice(nodes_list)
        r = random.random() * total
        cum = 0.0
        for j, w in zip(nodes_list, weights):
            cum += w
            if cum >= r:
                return j
        return nodes_list[-1]

def _apply_vnd(routes, nodes, vnd, dc_config):
    routes_info = {}
    for v_id, route in enumerate(routes):
        r_id = f"9{v_id:02d}"
        stores = [nodes[idx]['store_info'].copy() for idx in route]
        dc_copy = dc_config.copy()
        dc_copy['route_id'] = r_id
        dc_copy['route_code'] = r_id
        for i, s in enumerate(stores):
            s['route_id'] = r_id
            s['route_code'] = f"{r_id}{i:02d}"
            
        routes_info[r_id] = {
            'dc': dc_copy,
            'stores': stores
        }
    
    opt_routes_info, _ = vnd.optimize(routes_info)
    
    sid_to_idx = {n['sid']: n['idx'] for n in nodes[1:]}
    new_routes = []
    for k, v in opt_routes_info.items():
        if not v['stores']: continue
        r = [sid_to_idx[s['store_id']] for s in v['stores']]
        new_routes.append(r)
    return new_routes

def _new_active_ant(colony, num_vehicles, in_vec, use_ls, nodes, nodes_arr, dist, capacity):
    unvisited = set(range(1, len(nodes)))
    routes = []
    cur_route = []
    load = 0.0
    cur_time = nodes_arr[0, 1] # ready_time
    cur = 0
    depots_left = num_vehicles

    while unvisited and depots_left > 0:
        feasible = set()
        for cid in unvisited:
            nd = nodes[cid]
            if load + nd['demand'] > capacity:
                continue
            travel_time = colony.time_mat[cur][cid]
            arrival = cur_time + travel_time
            if arrival > nd['due_date']:
                continue
            
            if colony.is_solomon:
                start = max(arrival, nd['ready_time'])
            else:
                if arrival < nd['ready_time']:
                    continue
                start = arrival
                
            if start + nd['service_time'] + colony.time_mat[cid][0] > nodes[0]['due_date']:
                continue
            feasible.add(cid)

        if not feasible:
            if cur_route:
                routes.append(cur_route)
            cur_route, load, cur_time, cur = [], 0.0, 0.0, 0
            depots_left -= 1
            continue

        chosen = colony._choose(cur, feasible, in_vec, cur_time)
        if chosen is None:
            if cur_route:
                routes.append(cur_route)
            cur_route, load, cur_time, cur = [], 0.0, 0.0, 0
            depots_left -= 1
            continue

        colony._local_upd(cur, chosen)
        nd = nodes[chosen]
        travel_time = colony.time_mat[cur][chosen]
        arrival = cur_time + travel_time
        start = max(arrival, nd['ready_time']) if colony.is_solomon else arrival
        cur_time = start + nd['service_time']
        load += nd['demand']
        cur_route.append(chosen)
        unvisited.remove(chosen)
        cur = chosen

    if cur_route:
        routes.append(cur_route)

    routes = _insertion(routes, nodes_arr, colony.dist, colony.time_mat, capacity, colony.is_solomon)

    return routes

class _ACSTime(_ACSColony):
    def __init__(self, nodes, nodes_arr, dist, time_mat, capacity, num_vehicles, **kw):
        super().__init__(nodes, nodes_arr, dist, time_mat, capacity, **kw)
        self.num_vehicles = num_vehicles
        self._in = [0.0] * len(nodes)

    def cycle(self, gb_routes, gb_cost):
        best_sol, best_cost = None, float('inf')
        max_visited = 0
        for _ in range(self.num_ants):
            routes = _new_active_ant(self, self.num_vehicles, self._in,
                                     False, self.nodes, self.nodes_arr, self.dist, self.capacity)
            
            visited_count = sum(len(r) for r in routes)
            if visited_count > max_visited:
                max_visited = visited_count
                
            f, cost = _solution_cost(routes, self.nodes_arr, self.dist, self.time_mat, self.capacity, self.is_solomon)
            if f and cost < best_cost:
                best_cost = cost
                best_sol = [r[:] for r in routes]
        
        # Apply VND only to the best candidate of this cycle (Massive speedup)
        if best_sol is not None and self.vnd is not None:
            best_sol = _apply_vnd(best_sol, self.nodes, self.vnd, self.dc_config)
            _, best_cost = _solution_cost(best_sol, self.nodes_arr, self.dist, self.time_mat, self.capacity, self.is_solomon)
                
        if gb_routes:
            self._global_upd(gb_routes, gb_cost)
        return best_sol, best_cost, max_visited

class _ACSVei(_ACSColony):
    def __init__(self, nodes, nodes_arr, dist, time_mat, capacity, num_vehicles, **kw):
        super().__init__(nodes, nodes_arr, dist, time_mat, capacity, **kw)
        self.num_vehicles = num_vehicles
        self._in = [0.0] * len(nodes)
        self._best_visited = 0
        self._best_routes = None
        self._best_cost = float('inf')

    def cycle(self, gb_routes, gb_cost):
        new_feasible = None
        for _ in range(self.num_ants):
            routes = _new_active_ant(self, self.num_vehicles, self._in,
                                     False, self.nodes, self.nodes_arr, self.dist, self.capacity)
            nv = sum(len(r) for r in routes)

            for cid in range(1, self.n):
                # Approximate tracking of unvisited
                is_visited = any(cid in r for r in routes)
                if not is_visited:
                    self._in[cid] += 1.0

            if nv > self._best_visited:
                self._best_visited = nv
                f, cost = _solution_cost(routes, self.nodes_arr, self.dist, self.time_mat, self.capacity, self.is_solomon)
                self._best_routes = [r[:] for r in routes]
                self._best_cost = cost if f else float('inf')
                self._in = [0.0] * self.n
                
                # Apply VND only when we find a new record in visited nodes
                if self.vnd is not None:
                    self._best_routes = _apply_vnd(self._best_routes, self.nodes, self.vnd, self.dc_config)
                    f_opt, cost_opt = _solution_cost(self._best_routes, self.nodes_arr, self.dist, self.time_mat, self.capacity, self.is_solomon)
                    self._best_cost = cost_opt if f_opt else float('inf')
                    if f_opt: f = True
                
                if f:
                    new_feasible = (self._best_routes, self._best_cost)

        if self._best_routes and self._best_cost < float('inf'):
            self._global_upd(self._best_routes, self._best_cost)
        if gb_routes:
            self._global_upd(gb_routes, gb_cost)

        return new_feasible

# ─────────────────────────────────────────────────────────────────────────────
# Multiprocessing Workers
# ─────────────────────────────────────────────────────────────────────────────

def _acs_time_worker(nodes, nodes_arr, dist, time_mat, capacity, num_vehicles,
                     colony_kw, gb_routes, gb_cost, stop_event, result_queue,
                     print_lock, turn_control):
    acs = _ACSTime(nodes=nodes, nodes_arr=nodes_arr, dist=dist, time_mat=time_mat,
                   capacity=capacity, num_vehicles=num_vehicles, **colony_kw)
    iter_num = 0
    while not stop_event.is_set():
        sol, cost, max_v = acs.cycle(gb_routes, gb_cost)
        is_improve = False
        if sol is not None:
            nv_new = len(sol)
            nv_best = len(gb_routes)
            # Better if fewer vehicles, or same vehicles with less distance
            if nv_new < nv_best or (nv_new == nv_best and cost < gb_cost):
                gb_routes, gb_cost = sol, cost
                is_improve = True
                result_queue.put(('TIME_IMPROVED', [r[:] for r in sol], cost))
        
        # Wait for turn to print
        while turn_control[0] != 1 and not stop_event.is_set():
            time.sleep(0.01)
        if stop_event.is_set(): break
        
        with print_lock:
            if is_improve:
                print(f"    [ACS-DIST] iter {iter_num}, vehicle={len(sol)}, dist={cost:.2f} (Improve!)")
            elif cost < float('inf'):
                nv_curr = len(sol) if sol else num_vehicles
                print(f"    [ACS-DIST] iter {iter_num}, vehicle={nv_curr}, dist={cost:.2f}")
            else:
                print(f"    [ACS-DIST] iter {iter_num}, nodes={max_v}/{len(nodes)-1}")
            turn_control[0] = 0
            
        iter_num += 1


def _acs_vei_worker(nodes, nodes_arr, dist, time_mat, capacity, num_vehicles_minus1,
                    colony_kw, gb_routes, gb_cost, stop_event, result_queue,
                    print_lock, turn_control):
    acs = _ACSVei(nodes=nodes, nodes_arr=nodes_arr, dist=dist, time_mat=time_mat,
                  capacity=capacity, num_vehicles=num_vehicles_minus1, **colony_kw)
    iter_num = 0
    while not stop_event.is_set():
        res = acs.cycle(gb_routes, gb_cost)
        is_feasible = (res is not None)
        is_improve = False
        nv_new = num_vehicles_minus1 + 1
        cost_new = gb_cost
        
        if is_feasible:
            r, c = res
            nv_new = len(r)
            nv_best = len(gb_routes)
            cost_new = c
            
            # Check if this feasible solution is better than our local best
            if nv_new < nv_best or (nv_new == nv_best and c < gb_cost):
                gb_routes, gb_cost = r, c
                is_improve = True
        
        # Wait for turn to print
        while turn_control[0] != 0 and not stop_event.is_set():
            time.sleep(0.01)
        if stop_event.is_set(): break
        
        with print_lock:
            if is_feasible:
                if nv_new < num_vehicles_minus1 + 1:
                    print(f"    [ACS-VEI]  iter {iter_num}, vehicle={nv_new}, dist={cost_new:.2f} (FEASIBLE!)")
                    result_queue.put(('VEI_FEASIBLE', [ri[:] for ri in r], c))
                elif is_improve:
                    print(f"    [ACS-VEI]  iter {iter_num}, vehicle={nv_new}, dist={cost_new:.2f} (Improve!)")
                    result_queue.put(('VEI_IMPROVED', [ri[:] for ri in r], c))
                else:
                    print(f"    [ACS-VEI]  iter {iter_num}, nodes={acs._best_visited}/{len(nodes)-1} (FEASIBLE)")
            else:
                print(f"    [ACS-VEI]  iter {iter_num}, nodes={acs._best_visited}/{len(nodes)-1}")
            turn_control[0] = 1
            
        if is_feasible and nv_new < num_vehicles_minus1 + 1:
            break
        iter_num += 1

# ─────────────────────────────────────────────────────────────────────────────
# MACSSolver 
# ─────────────────────────────────────────────────────────────────────────────

class MACSSolver:
    def __init__(self, remaining_stores, distance_matrix, time_matrix,
                 num_ants=10, time_limit=60,
                 support_capacity=200,
                 time_limit_per_route=100000.0,
                 is_solomon=False,
                 beta=1.0, rho=0.1, q0=0.9, early_stop_patience=10,
                 verbose=True, vnd_strategy='best'):

        self.remaining_stores = remaining_stores
        self.distance_matrix  = distance_matrix
        self.verbose          = verbose
        self.time_limit       = time_limit
        self.is_solomon       = is_solomon
        self.early_stop_patience = early_stop_patience
        self.vnd_strategy = vnd_strategy
        self.support_capacity = support_capacity
        depot_ready_val = 0.0
        if remaining_stores:
             # Just an estimate or 0.0 for Solomon
             depot_ready_val = 0.0 
        
        self.nodes, self.dist, self.time_mat, self.capacity, self.depot_due = stores_to_internal(
            remaining_stores, distance_matrix, time_matrix,
            support_capacity, time_limit_per_route,
            depot_ready=depot_ready_val
        )

        self.vnd = VND(distance_matrix, time_matrix, vehicle_cost=0, is_solomon=is_solomon, 
                       vnd_strategy=self.vnd_strategy, time_limit=time_limit_per_route)
        self.dc_config = DC_CONFIG.copy()
        self.dc_config['max_capacity'] = support_capacity

        self.num_ants = num_ants
        self.beta = beta
        self.rho  = rho
        self.q0   = q0
        self.tau0 = 1.0

        self._nodes_internal = self.nodes

    def run(self):
        t_start = time.time()
        nodes, dist, cap = self.nodes, np.array(self.dist), self.capacity
        time_mat = np.array(self.time_mat)
        
        # Convert nodes to array for Numba
        # 0: demand, 1: ready_time, 2: due_date, 3: service_time
        nodes_arr = np.zeros((len(nodes), 4), dtype=np.float64)
        for i, n in enumerate(nodes):
            nodes_arr[i, 0] = n['demand']
            nodes_arr[i, 1] = n['ready_time']
            nodes_arr[i, 2] = n['due_date']
            nodes_arr[i, 3] = n['service_time']

        gb_routes = _nearest_neighbor(nodes, dist, time_mat, cap, self.is_solomon)
        gb_routes = _insertion(gb_routes, nodes_arr, dist, time_mat, cap, self.is_solomon)
        gb_f, gb_cost = _solution_cost(gb_routes, nodes_arr, dist, time_mat, cap, self.is_solomon)
        gb_nv = len(gb_routes)

        self.tau0 = 1.0 / (len(nodes) * len(gb_routes))
        if self.verbose:
            print(f"    [MACS] Init: {gb_nv} vehicles, dist={gb_cost:.2f}, feasible={gb_f}")

        if not gb_f:
            gb_cost = float('inf')

        early_stopper = EarlyStopper(patience=self.early_stop_patience)

        colony_kw = dict(num_ants=self.num_ants, beta=self.beta,
                         rho=self.rho, q0=self.q0, tau0=self.tau0,
                         is_solomon=self.is_solomon,
                         vnd=self.vnd, dc_config=self.dc_config)

        iter_num = 0
        while (time.time() - t_start) < self.time_limit:
            if self.verbose:
                print(f"    [MACS] iter {iter_num}, elapsed={time.time()-t_start:.1f}s, vehicle={gb_nv}, DIST={gb_cost:.2f}")
            
            v = gb_nv
            result_queue = queue.Queue()
            stop_event = threading.Event()
            print_lock = threading.Lock()
            turn_control = [0] # 0: VEI's turn, 1: DIST's turn

            t_time = threading.Thread(
                target=_acs_time_worker,
                args=(nodes, nodes_arr, dist, time_mat, cap, v,
                      colony_kw,
                      [r[:] for r in gb_routes], gb_cost,
                      stop_event, result_queue,
                      print_lock, turn_control))
            t_vei = threading.Thread(
                target=_acs_vei_worker,
                args=(nodes, nodes_arr, dist, time_mat, cap, max(1, v - 1),
                      colony_kw,
                      [r[:] for r in gb_routes], gb_cost,
                      stop_event, result_queue,
                      print_lock, turn_control))
            
            t_time.start()
            t_vei.start()

            # Monitoring loop
            while t_time.is_alive() or t_vei.is_alive():
                if (time.time() - t_start) >= self.time_limit:
                    stop_event.set()
                    break
                
                try:
                    msg = result_queue.get(timeout=0.1)
                    source, sol, cost = msg

                    if source == 'VEI_FEASIBLE':
                        nv_new = len(sol)
                        if nv_new < gb_nv or (nv_new == gb_nv and cost < gb_cost):
                            gb_routes, gb_cost, gb_nv = sol, cost, nv_new
                            if self.verbose:
                                print(f"    [MACS Iter {iter_num}] VEI ✓ vehicles={gb_nv}, DIST={gb_cost:.2f}")
                            stop_event.set() # Rebuild colonies immediately
                            break
                    elif source == 'VEI_IMPROVED':
                        nv_new = len(sol)
                        if nv_new < gb_nv or (nv_new == gb_nv and cost < gb_cost):
                            old_nv = gb_nv
                            gb_routes, gb_cost = sol, cost
                            gb_nv = nv_new
                            if gb_nv < old_nv:
                                if self.verbose:
                                    print(f"    [MACS Iter {iter_num}] VEI reduced vehicles! vehicles={gb_nv}")
                                stop_event.set()
                                break
                    elif source == 'TIME_IMPROVED':
                        nv_new = len(sol)
                        if nv_new < gb_nv or (nv_new == gb_nv and cost < gb_cost):
                            old_nv = gb_nv
                            gb_routes, gb_cost = sol, cost
                            gb_nv = nv_new
                            if self.verbose:
                                print(f"    [MACS Iter {iter_num}] DIST ✓ DIST={gb_cost:.2f}")
                            if gb_nv < old_nv:
                                if self.verbose:
                                    print(f"    [MACS Iter {iter_num}] DIST reduced vehicles! vehicles={gb_nv}")
                                stop_event.set()
                                break
                except queue.Empty:
                    continue
            
            stop_event.set()
            t_time.join()
            t_vei.join()

            if early_stopper.check((gb_nv, gb_cost)):
                print(f"    [MACS] Early stop triggered at iteration {iter_num}.")
                break

            if self.verbose:
                elapsed = time.time() - t_start
                print(f"    [MACS Iter {iter_num}] elapsed={elapsed:.1f}s, vehicles={gb_nv}, dist={gb_cost:.2f}")

        elapsed = time.time() - t_start
        if self.verbose:
            print(f"    [MACS] Done in {elapsed:.1f}s → {gb_nv} vehicles, dist={gb_cost:.2f}")

        best_cost_tuple = (gb_nv, gb_cost)

        def _sol_to_dict():
            base_dt = datetime(2024, 1, 1, 0, 0, 0)
            res = {}
            for v_id, route in enumerate(gb_routes):
                r_id = f"V{v_id+1:02d}"
                
                total_dist = 0.0
                total_load = 0.0
                cur_time = 0.0
                prev = 0
                
                formatted_stores = []
                for i, idx in enumerate(route):
                    node = self._nodes_internal[idx]
                    s_info = node['store_info'].copy()
                    
                    travel_dist = self.dist[prev][idx]
                    travel_time = self.time_mat[prev][idx]
                    arrival = cur_time + travel_time
                    
                    if self.is_solomon:
                        start = max(arrival, node['ready_time'])
                    else:
                        start = arrival
                        
                    # Update times in ISO format (rounding to seconds to remove decimals)
                    # Update pred_time in ISO format (rounding to seconds)
                    s_info['pred_time'] = (base_dt + timedelta(minutes=start)).isoformat(timespec='seconds')
                    
                    # Update route codes
                    s_info['route_id'] = r_id
                    s_info['route_code'] = f"{r_id}{i+1:02d}"
                    
                    formatted_stores.append(s_info)
                    
                    # Advance simulation
                    cur_time = start + node['service_time']
                    total_dist += travel_dist
                    total_load += node['demand']
                    prev = idx
                    
                # Final leg back to depot
                total_dist += self.dist[prev][0]
                total_time = cur_time + self.time_mat[prev][0]
                
                dc = {
                    "route_id": r_id,
                    "route_code": r_id,
                    "store_id": self.dc_config.get('store_id', 'dc'),
                    "store_name": self.dc_config.get('store_name', 'Solomon Depot'),
                    "total_volume": float(total_load),
                    "load_rate": float(total_load / self.capacity) if self.capacity > 0 else 0.0,
                    "max_capacity": float(self.capacity),
                    "region": self.dc_config.get('region', 'unknown'),
                    "distance": float(total_dist),
                    "duration": float(total_time)
                }
                
                res[r_id] = {
                    "dc": dc,
                    "stores": formatted_stores
                }
            return res

        best_solution_dict = _sol_to_dict()
        return best_cost_tuple, best_solution_dict
